import json  # 导入序列化工具
from typing import List, Dict, Any  # 导入类型注解工具 List代码列表 Dict代表字典
from openai import OpenAI  # 导入OpenAI的官方SDK
from utils import function_to_json  # 导入自定义的工具转换函数

# 定义 Agent 的系统提示词，用于设定角色的行为准则
SYSTEM_PROMPT = """
你是一个叫JUN人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""


class Agent:
    def __init__(  # Python 类的初始化方法，创建对象时自动调用，初始化 Agent 对象
        self,  # 指向当前对象实例，用于访问和设置实例属性
        client: OpenAI,
        model: str = "Qwen/Qwen2.5-32B-Instruct",
        tools: List = None,
        verbose: bool = True,
    ):
        """
        初始化 Agent
        :param client: OpenAI 客户端实例
        :param model: 使用的模型名称
        :param tools: 可调用的工具函数列表
        :param verbose: 是否打印详细的调试信息（如工具调用过程）
        """
        self.client = client
        self.tools = tools or []
        self.model = model
        # 初始化对话历史，包含系统提示词
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose
        # 创建工具名称到函数的映射字典，方便后续通过名称查找并执行函数
        self.tool_map = {tool.__name__: tool for tool in self.tools}

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        """
        将 Python 函数列表转换为模型可识别的 JSON Schema 格式
        :return: 符合 OpenAI/DashScope 规范的工具定义列表
        """
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        """
        处理模型发出的单个工具调用请求
        :param tool_call: 模型返回的工具调用对象
        :return: 包含工具执行结果的消息字典
        """
        function_name = tool_call.function.name
        function_args_str = tool_call.function.arguments
        function_id = tool_call.id

        try:
            # 将模型生成的 JSON 字符串参数解析为 Python 字典
            function_args = json.loads(function_args_str)

            # 在工具映射表中查找对应的函数
            if function_name in self.tool_map:
                if self.verbose:
                    print(f">>> 正在执行工具: {function_name}, 参数: {function_args}")

                # 执行函数并获取结果
                func = self.tool_map[function_name]
                result = func(**function_args)
            else:
                result = f"错误: 找不到名为 {function_name} 的工具"
        except Exception as e:
            # 捕获执行过程中的任何异常并返回错误信息
            result = f"工具执行过程中出错: {str(e)}"

        # 构造符合 API 规范的工具响应消息
        return {
            "role": "tool",
            "content": str(result),
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt: str) -> str:
        """
        处理用户输入的核心方法，包含推理、工具调用和最终回复的完整循环
        :param prompt: 用户输入的文本
        :return: Agent 的最终回答
        """
        # 将用户问题添加到对话历史中
        self.messages.append({"role": "user", "content": prompt})

        # 1. 第一次请求模型，判断是否需要调用工具
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema() if self.tools else None,
            stream=False,
        )

        message = response.choices[0].message

        # 2. 循环处理工具调用（模型可能会连续请求多个工具，或根据工具结果再请求新工具）
        while message.tool_calls:
            # 必须先将模型包含 tool_calls 的回复加入历史，作为上下文的一部分
            self.messages.append(message)

            # 遍历并执行所有请求的工具调用
            for tool_call in message.tool_calls:
                tool_result_message = self.handle_tool_call(tool_call)
                # 将每个工具的执行结果加入历史
                self.messages.append(tool_result_message)

            # 带着工具执行结果再次请求模型，获取进一步的指令或最终结论
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema() if self.tools else None,
                stream=False,
            )
            message = response.choices[0].message

        # 3. 当模型不再请求工具时，得到最终回复，存入历史并返回
        self.messages.append({"role": "assistant", "content": message.content})
        return message.content or ""
