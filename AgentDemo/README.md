# 从零开始构建一个 AI Agent：使用阿里云百炼实现工具调用

## 📖 前言：从 LLM 到 Agent

在了解 Agent 之前，我们先了解一下 LLM。

我们都知道 2022 年 11 月 30 日，OpenAI 发布了 ChatGPT，而 ChatGPT 就是 LLM 的典型应用实例之一，包括后面发布的文心一言、通义千问等大模型。

**LLM**（Large Language Model，大语言模型）是人工智能领域中自然语言处理（NLP）方向的核心技术，也是当前生成式 AI 的底层支撑。它的本质是**基于海量文本数据训练、具备强大语言理解与生成能力的深度学习模型**。

我们从以上的概念中很容易就能看到 LLM 的一些局限，比如：

- **语言理解**：只能基于训练过的文本数据去进行理解
- **语言生成**：只能以文本的方式去进行交互

这两个局限使得 LLM 更像是一个**建议者**，比如开车时的副驾驶，无法执行具体的动作。

而在 AI 的发展中，我们更需要 AI 去做一个**执行者**，去做司机，真正的融入到我们的物理世界中，由此，**Agent（智能体）**发展出来了。

---

## 🧠 Agent 核心概念

现在我们常常用这样的一个智能体架构公式去理解 Agent：

```js
Agent(智能体) = LLM(大语言模型) + Perception(感知) + Planning(规划) + Memory(记忆) + Tools(工具)
```

有了 Perception（感知）、Planning（规划）、Memory（记忆）、Tools（工具），智能体真正具备了五官与手脚，去理解、操作我们的物理世界：

### 1. Perception（感知）- Agent 与外部沟通的桥梁

通过多种渠道感知环境信息，类似于人的感官：

- **视觉感知（CV）**：通过摄像头或图像输入，看到并理解内容
- **听觉感知（ASR）**：通过麦克风或音频输入，听到并识别语音内容
- **其它感知**：集成传感器，感知环境的物理状态，如温度、湿度、味道等

### 2. Planning（规划）- Agent 的决策执行引擎

负责将 LLM 拆解的目标转化为可执行的分布计划，并根据反馈动态调整策略：

- **任务拆解**：将复杂目标拆解成一系列可执行的简单子任务
- **制定策略**：为每个子任务规划出具体的执行步骤和方法
- **动态调整**：根据执行反馈灵活调整计划，应对突发情况

### 3. Memory（记忆）- Agent 的长期知识库

解决 LLM 上下文窗口有限的问题，为 Agent 提供长期学习和经验积累的能力：

- **短期记忆**：类似工作记忆，存储当前任务上下文，支持多轮对话与任务连贯性
- **长期记忆**：类似知识库，持久化存储知识、偏好与经验，可随时检索利用

### 4. Tools（工具）- Agent 的双手

让 Agent 突破纯文本能力限制，实现"知行合一"的任务执行：

- **访问信息**：通过搜索引擎获取最新、最准确的信息
- **执行计算**：使用计算器或代码编辑器完成复杂运算和数据分析
- **控制设备**：通过 API 调用，控制智能家居、工业机器人等物理设备
- **操作软件**：自动操作电脑软件，如 Excel、Photoshop 等

---

## 🎯 项目简介

本项目是一个基于 **ReAct (Reasoning and Acting)** 架构的 AI Agent 示例，重点实现了 **Tools（工具）** 这一核心组件。它能够：

- 理解用户的自然语言问题
- 自动判断是否需要调用外部工具
- 执行 Python 函数（如获取时间、数学计算等）
- 将工具执行结果整合成人类可读的回答

**技术栈**：
- **大模型**：阿里云百炼（通义千问）
- **通信协议**：OpenAI 兼容模式
- **编程语言**：Python 3.8+

**在本项目中的实现**：
- **LLM**：使用阿里云百炼的通义千问模型作为"大脑"
- **Planning**：通过 ReAct 循环实现任务规划和动态调整
- **Memory**：通过 `messages` 列表维护对话上下文（短期记忆）
- **Tools**：通过 Python 函数实现工具调用能力（本项目重点）

---

## 🎯 项目目标

通过这个项目，你将学会：

1. 理解 Agent 的架构组成和工作原理
2. 如何将大模型与外部工具（Python 函数）结合
3. 理解 ReAct 循环的工作原理
4. 掌握 Function Calling（工具调用）的实现机制
5. 构建一个可扩展的 Agent 框架

---

## 📋 前置准备

### 1. 环境要求
- Python 3.8 或更高版本
- pip 包管理器
- 阿里云百炼 API Key（免费版即可）

### 2. 获取 API Key
1. 访问 [阿里云百炼控制台](https://bailian.console.aliyun.com/)
2. 注册/登录账号
3. 创建 API Key（格式类似：`sk-xxxxxxxxxxxxx`）

---

## 🚀 逐步实现

### 第一步：项目初始化

创建项目目录结构：

```bash
mkdir AgentDemo
cd AgentDemo
mkdir tools
```

### 第二步：安装依赖

创建 `requirements.txt` 文件：

```txt
openai
python-dotenv
```

安装依赖：

```bash
pip3 install -r requirements.txt
```

### 第三步：配置环境变量

创建 `.env` 文件（注意：不要将此文件提交到 Git）：

```env
DASHSCOPE_API_KEY=你的API_KEY
```

### 第四步：实现工具转换器（utils.py）

这是整个项目的"翻译官"，负责将 Python 函数自动转换为模型能理解的 JSON Schema。

**核心思路**：利用 Python 的 `inspect` 模块，读取函数的：
- 函数名
- 参数列表和类型注解
- 函数的 docstring（作为工具描述）

```python
import inspect

def function_to_json(func) -> dict:
    """
    通过 Python 反射机制，自动将一个 Python 函数转换为 
    OpenAI/DashScope 兼容的工具定义 (JSON Schema)
    """
    sig = inspect.signature(func)
    parameters = {}
    required = []

    # 遍历函数的所有参数
    for name, param in sig.parameters.items():
        # 根据类型注解推断 JSON Schema 类型
        if param.annotation == float:
            param_type = "number"
        elif param.annotation == int:
            param_type = "integer"
        elif param.annotation == str:
            param_type = "string"
        elif param.annotation == bool:
            param_type = "boolean"
        else:
            param_type = "string"

        parameters[name] = {"type": param_type}
        
        # 没有默认值的参数标记为必填
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func) or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

**为什么需要这个？**
- 大模型无法直接"看到"你的 Python 代码
- 它需要标准化的 JSON 格式来了解工具的功能和参数
- 手动编写 JSON Schema 容易出错且繁琐，自动化可以大大提升开发效率

### 第五步：定义工具函数（tools/）

#### 5.1 时间工具（tools/time_tool.py）

```python
from datetime import datetime

def get_current_datetime() -> str:
    """
    获取当前日期和时间。
    Agent 可以调用此工具来了解当前的实时时间。
    :return: 格式化后的日期时间字符串
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime
```

#### 5.2 数学工具（tools/math_tools.py）

```python
def add(a: int, b: int) -> int:
    """
    计算两个整数的和。
    Agent 会在需要进行加法运算时自动调用此工具。
    """
    return a + b

def compare(a: float, b: float) -> str:
    """
    比较两个数字的大小，并返回描述性结论。
    """
    if a > b:
        return f"{a} 大于 {b}"
    elif a < b:
        return f"{a} 小于 {b}"
    else:
        return f"{a} 等于 {b}"

def count_letter_in_string(letter: str, text: str) -> int:
    """
    统计指定字符在字符串中出现的次数。
    """
    return text.count(letter)
```

**关键点**：
- 每个函数必须有清晰的 **docstring**（模型用它来理解工具用途）
- 使用 **类型注解**（Type Hints）标注参数类型
- 函数名要有语义，便于模型选择

### 第六步：实现 Agent 核心逻辑（agent.py）

这是项目的"大脑"，负责：
1. 维护对话历史（Memory - 短期记忆）
2. 调用大模型（LLM）
3. 解析工具调用指令（Planning - 任务规划）
4. 执行工具并反馈结果（Tools - 工具执行）

#### 6.1 初始化

```python
import json
from typing import List, Dict, Any
from openai import OpenAI
from utils import function_to_json

SYSTEM_PROMPT = """
你是一个叫JUN人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""

class Agent:
    def __init__(self, client: OpenAI, model: str = "qwen-plus", 
                 tools: List = None, verbose: bool = True):
        self.client = client
        self.tools = tools or []
        self.model = model
        # Memory: 初始化对话历史，包含系统提示词
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.verbose = verbose
        # Tools: 创建工具映射：函数名 -> 函数对象
        self.tool_map = {tool.__name__: tool for tool in self.tools}
```

#### 6.2 工具 Schema 生成

```python
def get_tool_schema(self) -> List[Dict[str, Any]]:
    """将 Python 函数列表转换为模型可识别的 JSON Schema"""
    return [function_to_json(tool) for tool in self.tools]
```

#### 6.3 工具调用处理

```python
def handle_tool_call(self, tool_call):
    """处理模型发出的单个工具调用请求"""
    function_name = tool_call.function.name
    function_args_str = tool_call.function.arguments
    function_id = tool_call.id

    try:
        # 解析 JSON 字符串参数
        function_args = json.loads(function_args_str)
        
        # 从映射表中查找并执行函数
        if function_name in self.tool_map:
            if self.verbose:
                print(f">>> 正在执行工具: {function_name}, 参数: {function_args}")
            
            func = self.tool_map[function_name]
            result = func(**function_args)
        else:
            result = f"错误: 找不到名为 {function_name} 的工具"
    except Exception as e:
        result = f"工具执行过程中出错: {str(e)}"

    return {
        "role": "tool",
        "content": str(result),
        "tool_call_id": function_id,
    }
```

#### 6.4 ReAct 循环（核心逻辑 - Planning）

```python
def get_completion(self, prompt: str) -> str:
    """
    处理用户输入的核心方法，实现 ReAct 循环：
    Reasoning（推理）-> Acting（行动）-> Observing（观察）-> Reasoning（再推理）
    
    这体现了 Planning（规划）的核心思想：
    1. 任务拆解：将用户问题拆解为可执行的工具调用
    2. 制定策略：决定调用哪些工具、以什么顺序调用
    3. 动态调整：根据工具执行结果调整后续策略
    """
    # Memory: 添加用户问题到历史
    self.messages.append({"role": "user", "content": prompt})

    # Planning: 第一次请求 - LLM 判断是否需要工具
    response = self.client.chat.completions.create(
        model=self.model,
        messages=self.messages,
        tools=self.get_tool_schema() if self.tools else None,
        stream=False,
    )

    message = response.choices[0].message

    # Planning: 循环处理工具调用（支持多次调用或链式调用）
    while message.tool_calls:
        # Memory: 将模型的工具调用请求加入历史
        self.messages.append(message)

        # Tools: 执行所有请求的工具
        for tool_call in message.tool_calls:
            tool_result_message = self.handle_tool_call(tool_call)
            # Memory: 将工具执行结果加入历史
            self.messages.append(tool_result_message)

        # Planning: 带着工具结果再次请求模型，进行动态调整
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema() if self.tools else None,
            stream=False,
        )
        message = response.choices[0].message

    # Memory: 模型不再调用工具，返回最终回复，存入历史并返回
    self.messages.append({"role": "assistant", "content": message.content})
    return message.content or ""
```

**ReAct 循环流程**（体现 Planning 的三大能力）：
1. **任务拆解**：模型分析用户问题，决定调用哪个工具
2. **制定策略**：Agent 执行工具函数
3. **动态调整**：将工具结果反馈给模型，模型根据结果生成最终回答或继续调用其他工具

### 第七步：创建主程序（main.py）

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from agent import Agent
from tools.time_tool import get_current_datetime
from tools.math_tools import add, compare, count_letter_in_string

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")

if __name__ == "__main__":
    if not API_KEY:
        print("错误: 请在 .env 文件中配置 DASHSCOPE_API_KEY")
        exit(1)

    # 初始化 OpenAI 客户端（使用阿里云百炼的兼容接口）
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 创建 Agent 实例，注入工具
    agent = Agent(
        client=client,
        model="qwen-plus",
        tools=[get_current_datetime, add, compare, count_letter_in_string],
        verbose=True,
    )

    print("JUN 人工智能助手已启动！(输入 'exit' 退出)")

    # 交互式对话循环
    while True:
        try:
            prompt = input("\033[94mUser: \033[0m")
            
            if prompt.lower() in ["exit", "quit", "退出"]:
                print("再见！")
                break

            if not prompt.strip():
                continue

            response = agent.get_completion(prompt)
            print("\033[92mAssistant: \033[0m", response, "\n")

        except KeyboardInterrupt:
            print("\n程序终止")
            break
        except Exception as e:
            print(f"\033[91m发生错误: {e}\033[0m")
```

---

## 🎬 运行项目

1. **确保已配置 `.env` 文件**
2. **运行主程序**：
   ```bash
   python3 main.py
   ```
3. **测试对话**：
   - "现在几点了？" → 会调用 `get_current_datetime`
   - "帮我算一下 123 + 456" → 会调用 `add`
   - "比较一下 3.14 和 3.15 哪个大" → 会调用 `compare`

---

## 🔍 核心原理深度解析

### 1. 为什么需要工具调用？

大模型虽然强大，但它有局限性：
- **无法获取实时信息**（如当前时间、天气）
- **无法执行计算密集型任务**（如复杂数学运算）
- **无法访问外部系统**（如数据库、API）

通过工具调用，我们让模型具备了"动手能力"，这正是 Agent 架构中 **Tools（工具）** 组件的核心价值。

### 2. JSON Schema 的作用

模型需要知道：
- 工具有什么功能？（通过 `description`）
- 需要什么参数？（通过 `parameters`）
- 参数是什么类型？（通过 `type`）

这就是为什么我们需要将 Python 函数转换为 JSON Schema。这相当于给模型提供了一份"工具使用说明书"。

### 3. ReAct 循环的优势（Planning 的体现）

- **可解释性**：每一步推理和行动都清晰可见
- **可扩展性**：可以轻松添加新工具
- **鲁棒性**：工具执行失败时，模型可以重新规划（动态调整）

### 4. Memory 在本项目中的实现

本项目实现了 **短期记忆**：
- 通过 `messages` 列表维护对话上下文
- 每次对话都会保留完整的交互历史
- 支持多轮对话的连贯性

**扩展方向**（长期记忆）：
- 使用向量数据库存储历史对话
- 实现知识库检索功能
- 持久化用户偏好和习惯

---

## 🚀 扩展建议

### 1. 添加更多工具（Tools 扩展）

在 `tools/` 目录下创建新文件，例如：
- `web_tool.py`：网页搜索、内容抓取
- `file_tool.py`：文件读写操作
- `api_tool.py`：调用第三方 API
- `database_tool.py`：数据库查询操作

### 2. 持久化对话历史（Memory 扩展）

使用数据库（如 SQLite）保存对话记录，实现多轮对话的持久化：
- 实现长期记忆存储
- 支持对话历史检索
- 用户偏好学习

### 3. 添加流式输出

修改 `stream=False` 为 `stream=True`，实现打字机效果，提升用户体验。

### 4. 错误处理增强

- 工具执行超时处理
- 参数验证
- 重试机制
- 优雅降级策略

### 5. 多 Agent 协作

实现多个 Agent 分工合作，处理复杂任务：
- 不同 Agent 负责不同领域
- Agent 之间的通信机制
- 任务分配和协调

### 6. 添加 Perception 能力

虽然本项目主要关注 Tools，但你可以扩展感知能力：
- 图像理解（集成视觉模型）
- 语音识别（集成 ASR）
- 多模态输入处理

---

## 💡 应用场景

基于本项目构建的 Agent 可以应用于：

1. **智能客服**：自动回答用户问题，调用订单查询、退款等工具
2. **数据分析助手**：理解自然语言查询，自动执行数据分析任务
3. **代码助手**：理解需求，调用代码生成、测试、部署等工具
4. **智能家居控制**：通过语音或文本控制智能设备
5. **自动化办公**：处理邮件、生成报告、操作办公软件

---

## 📚 学习资源

- [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling)
- [阿里云百炼文档](https://help.aliyun.com/zh/model-studio/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [LangChain Agents 文档](https://python.langchain.com/docs/modules/agents/)

---

## 🐛 常见问题

### Q: 为什么模型不调用工具？
A: 检查：
1. 工具的 docstring 是否清晰
2. 系统提示词是否明确要求使用工具
3. 用户问题是否足够明确

### Q: 工具执行失败怎么办？
A: 代码中已有异常捕获，会将错误信息返回给模型，模型可以重新规划（体现 Planning 的动态调整能力）。

### Q: 如何切换其他模型？
A: 修改 `main.py` 中的 `model` 参数，例如 `"qwen-max"` 或 `"qwen-turbo"`。

### Q: 如何实现长期记忆？
A: 可以使用向量数据库（如 Chroma、Pinecone）存储历史对话，并在每次对话时进行相似度检索。

---

## 📝 总结

通过这个项目，我们深入理解了 Agent 的架构组成：

1. ✅ **LLM**：作为 Agent 的"大脑"，提供语言理解和生成能力
2. ✅ **Planning**：通过 ReAct 循环实现任务拆解、策略制定和动态调整
3. ✅ **Memory**：通过对话历史维护短期记忆，支持多轮对话
4. ✅ **Tools**：通过 Python 函数实现工具调用，让 Agent 具备执行能力

**核心收获**：
- 理解了 Agent = LLM + Planning + Memory + Tools 的架构公式
- 掌握了如何将大模型与外部工具结合
- 实现了 ReAct 循环的核心逻辑
- 构建了一个可扩展的 Agent 框架

**下一步**：
- 尝试添加你自己的工具，让 Agent 变得更强大
- 探索长期记忆的实现方式
- 研究多 Agent 协作的可能性
- 思考如何添加 Perception 能力

---

**作者**：JUN  
**项目地址**：[GitHub 链接]  
**许可证**：MIT
