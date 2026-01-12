"""
LangChain 教学 Demo
展示 LangChain 框架的核心功能：
1. Models - 大语言模型封装
2. Tools - 工具扩展
3. Agents - 智能代理
4. Memory - 对话记忆
5. Document Processing - 文档处理（详见 document_vector_demo.py）
6. Vector Stores - 向量存储（详见 document_vector_demo.py）
7. Retrieval - 检索增强生成（详见 document_vector_demo.py）
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from tools.math_tools import add, subtract, multiply, divide
from tools.time_tool import get_current_datetime, get_current_date, get_current_time

# 加载环境变量
load_dotenv()


def create_llm():
    """
    创建大语言模型实例（Models 模块）
    
    LangChain 的 Models 模块作用：
    - 统一接口：解决不同模型提供商 API 差异问题
    - 参数管理：集中管理 API 密钥、temperature 等参数
    - 集成方便：天然适配于 LangChain 的其他组件
    - 错误处理：内置重试、错误处理机制
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 文件中配置 DASHSCOPE_API_KEY")
    
    llm = ChatOpenAI(
        model="qwen-plus",  # 通义千问模型
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,  # 控制输出的随机性
    )
    return llm


def create_tools():
    """
    创建工具列表（Tools 模块）
    
    LangChain 的 Tools 模块作用：
    - 功能扩展：扩展 AI 的外部能力，如调用地图 API、时间 API 等
    - 系统交互：与外部的系统交互，访问数据库等
    - 任务执行：执行计算任务等
    
    @tool 装饰器的作用：
    - 自动生成工具描述：从函数文档字符串提取，供 Agent 理解工具用途
    - 自动生成工具 Schema：将函数签名转换为 JSON Schema，供 Agent 调用
    - 工具注册：将工具注册到 Agent 的工具列表中
    """
    tools = [
        add,
        subtract,
        multiply,
        divide,
        get_current_datetime,
        get_current_date,
        get_current_time,
    ]
    return tools


def create_memory():
    """
    创建记忆模块（Memory 模块）
    
    LangChain 的 Memory 模块作用：
    - 上下文连贯：Agent 能理解多轮对话的上下文
    - 信息持久化：在单次会话中记住用户提供的信息
    - 智能对话：支持需要多轮交互的复杂任务
    - 自动管理：无需手动维护对话历史列表
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 在 prompt 中使用的变量名
        return_messages=False,  # 返回字符串格式，适合字符串模板
    )
    return memory


def create_agent_executor(llm, tools, memory):
    """
    创建 Agent 执行器（Agents 模块）
    
    LangChain 的 Agents 模块作用：
    - 初始化 Agent：构建一个采用 ReAct 模式的 Agent，由 LLM 驱动，配备工具列表和提示词
    - 循环管理：执行 Agent 循环逻辑，自动管理 ReAct 循环
    - 工具调用：根据 Agent 决策调用工具并获取结果
    - 短期记忆：封装 memory 模块，用于记忆多轮对话
    - 错误处理：处理解析错误和工具执行异常
    - 迭代控制：限制最大迭代次数，防止无限循环
    - 日志输出：verbose=True 时，显示详细执行过程
    """
    # ReAct 提示词模板
    prompt_template = """你是一个友好的 AI 助手。你可以使用工具来回答问题。

你可以使用的工具：
{tools}

使用以下格式：
Question: 需要回答的问题
Thought: 你应该思考要做什么
Action: 要采取的行动，应该是 [{tool_names}] 中的一个
Action Input: 行动的输入
Observation: 行动的结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答案

{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(prompt_template)

    # 创建 ReAct Agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # 创建 Agent 执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,  # 添加 memory 支持，实现多轮对话记忆
        verbose=True,  # 显示详细的执行过程
        handle_parsing_errors=True,  # 处理解析错误
        max_iterations=5,  # 最大迭代次数
    )

    return agent_executor


def main():
    """
    主函数：演示 LangChain 的核心功能
    """
    print("=" * 80)
    print("LangChain 教学 Demo")
    print("=" * 80)
    print()
    print("本 Demo 展示了 LangChain 框架的 4 个核心模块：")
    print("1. Models - 大语言模型封装（使用通义千问）")
    print("2. Tools - 工具扩展（数学运算、时间查询）")
    print("3. Agents - 智能代理（ReAct 模式）")
    print("4. Memory - 对话记忆（多轮对话支持）")
    print()
    print("💡 提示：")
    print("  - Document Processing（文档处理）和 Vector Stores（向量存储）模块")
    print("    请运行: python document_vector_demo.py")
    print("  - 这两个模块通常与 Retrieval（检索）模块一起使用，实现 RAG 功能")
    print()
    print("=" * 80)
    print()

    try:
        # 1. 创建 LLM（Models 模块）
        print("【步骤 1】初始化大语言模型（Models 模块）...")
        llm = create_llm()
        print("✓ 模型初始化成功：qwen-plus")
        print()

        # 2. 创建工具（Tools 模块）
        print("【步骤 2】加载工具（Tools 模块）...")
        tools = create_tools()
        print(f"✓ 已加载 {len(tools)} 个工具：")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")
        print()

        # 3. 创建记忆（Memory 模块）
        print("【步骤 3】初始化记忆模块（Memory 模块）...")
        memory = create_memory()
        print("✓ 记忆模块初始化成功（支持多轮对话）")
        print()

        # 4. 创建 Agent（Agents 模块）
        print("【步骤 4】创建 Agent 执行器（Agents 模块）...")
        agent_executor = create_agent_executor(llm, tools, memory)
        print("✓ Agent 执行器创建成功（ReAct 模式）")
        print()

        print("=" * 80)
        print("AI 助手已启动！(输入 'exit' 或 '退出' 退出)")
        print("=" * 80)
        print()
        print("💡 提示：你可以尝试以下问题：")
        print("  - 计算 123 + 456")
        print("  - 现在几点了？")
        print("  - 今天是几号？")
        print("  - 先计算 10 * 20，然后告诉我结果")
        print()

        # 交互式对话循环
        while True:
            try:
                # 使用彩色输出区分用户输入和 AI 回答
                prompt = input("\033[94m用户: \033[0m")  # 蓝色显示用户输入提示

                if prompt.lower() in ["exit", "quit", "退出"]:
                    print("\n再见！")
                    break

                if not prompt.strip():
                    continue

                # 调用 Agent 执行器
                print("\033[92m助手: \033[0m", end="")  # 绿色显示 AI 助手回答
                response = agent_executor.invoke({"input": prompt})
                print(response["output"])
                print()

            except KeyboardInterrupt:
                print("\n\n程序已中断")
                break
            except Exception as e:
                print(f"\033[91m发生错误: {e}\033[0m")
                print()

    except ValueError as e:
        print(f"\033[91m配置错误: {e}\033[0m")
        print("提示：请确保已创建 .env 文件并配置 DASHSCOPE_API_KEY")
    except Exception as e:
        print(f"\033[91m发生错误: {e}\033[0m")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

