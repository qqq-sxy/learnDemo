import os  # 导入os模块，用于获取环境变量
from dotenv import load_dotenv  # 导入dotenv模块，用于加载环境变量
from openai import OpenAI  # 导入OpenAI的官方SDK
from agent import Agent  # 导入agent.py模块
from tools.time_tool import get_current_datetime  # 导入time_tool.py模块
from tools.math_tools import (
    add,  # 导入add函数
)  # 导入math_tools.py模块

load_dotenv()

# 获取 API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY")

if __name__ == "__main__":
    if not API_KEY:
        print("错误: 请在 .env 文件中配置 DASHSCOPE_API_KEY")
        exit(1)

    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 创建 Agent 实例
    # 传入 client、模型名称和工具函数列表
    agent = Agent(
        client=client,
        model="qwen-plus",  # 阿里云百炼常用的模型
        tools=[get_current_datetime, add],
        verbose=True,  # 设置为 True 可以看到工具调用过程
    )

    print("JUN 人工智能助手已启动！(输入 'exit' 退出)")

    # 开始交互式对话循环
    while True:
        try:
            # 使用彩色输出区分用户输入和 AI 回答
            prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示

            if prompt.lower() in ["exit", "quit", "退出"]:
                print("再见！")
                break

            if not prompt.strip():  # 判断当前是否有有效内容
                continue

            # 调用agent.py模块的get_completion方法，获取AI助手回答
            response = agent.get_completion(prompt)
            print("\033[92mAssistant: \033[0m", response, "\n")  # 绿色显示 AI 助手回答

        except KeyboardInterrupt:
            print("\n程序终止")
            break
        except Exception as e:
            print(f"\033[91m发生错误: {e}\033[0m")
