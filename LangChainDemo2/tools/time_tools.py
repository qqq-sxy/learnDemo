"""
时间工具模块
用于获取当前日期和时间
"""

from datetime import datetime
from langchain.tools import tool


@tool
def get_current_time() -> str:
    """
    获取当前时间。
    Agent 会在需要获取当前时间时自动调用此工具。

    Returns:
        当前时间（格式为 HH:MM:SS）
    """
    return datetime.now().strftime("%H:%M:%S")
