import os
import warnings

# 抑制 urllib3 的 OpenSSL 警告（macOS 使用 LibreSSL 是正常的）
# 这个警告不影响功能，只是版本兼容性提示
# 需要在导入可能使用 urllib3 的库之前设置
warnings.filterwarnings("ignore", message=".*urllib3.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*OpenSSL.*", category=UserWarning)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

# 抑制 LangChain 的弃用警告
# 注意：这是临时解决方案，建议未来迁移到新的 API
from langchain_core._api.deprecation import LangChainDeprecationWarning

from langchain.memory import ConversationBufferMemory
from tools.math_tools import add
from tools.time_tools import get_current_time

# 加载环境变量
load_dotenv()


def create_llm():  # 创建 LLM 模型实例
    api_key = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量中获取 API 密钥
    if not api_key:  # 如果 API 密钥未设置，则抛出错误
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云 DashScope API 的兼容模式 URL
        temperature=0.7,  # 用于控制生成文本的随机性
    )
    return llm


def create_tools():  # 创建工具列表
    tools = [add, get_current_time]
    return tools


def create_memory():  # 创建记忆模块
    # 使用新的方式创建 memory，避免弃用警告
    # 在新版本中，使用 return_messages=False 避免警告
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,  # 返回字符串格式，适合字符串模板
    )
    return memory


def create_agent_executor(llm, tools, memory):  # 创建Agent执行器
    prompt_template = """
    你是一个友好的AI助手，可以使用工具来回答用户问题
    用户的问题: {input}
    行动: {tool_names}
    思考过程: {agent_scratchpad}
    工具: {tools}
    记忆: {chat_history}
    回答:
    """
    prompt = PromptTemplate.from_template(prompt_template)  # 创建提示模板

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)  # 创建Agent

    agent_executor = AgentExecutor(
        agent=agent,  # Agent实例
        tools=tools,  # 工具列表
        memory=memory,  # 记忆模块
        verbose=True,  # 是否打印详细信息
        handle_parsing_errors=True,  # 处理解析错误
        max_iterations=5,  # 最大迭代次数
    )
    return agent_executor


# 创建主函数
def main():
    try:
        llm = create_llm()  # 创建 LLM 模型实例
        tools = create_tools()  # 创建工具列表
        memory = create_memory()  # 创建记忆模块
        agent_executor = create_agent_executor(llm, tools, memory)  # 创建Agent执行器
        while True:  # 循环输入问题
            prompt = input("请输入问题：")
            if prompt.lower() in ["exit", "quit", "退出"]:
                print("\n再见！")
                break

            if not prompt.strip():
                continue
            response = agent_executor.invoke({"input": prompt})  # 执行Agent
            print(response["output"])  # 打印响应
    except ValueError as e:  # 如果创建 LLM 模型实例失败，则抛出错误
        print(f"创建 LLM 模型实例失败: {e}")
        return


if __name__ == "__main__":
    main()
