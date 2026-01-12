"""
时间工具模块
用于获取当前日期和时间
"""
from datetime import datetime
from langchain.tools import tool


@tool
def get_current_datetime() -> str:
    """
    获取当前日期和时间的工具。
    Agent 可以调用此工具来了解当前的实时时间。
    
    Returns:
        格式化后的日期时间字符串，例如 "2024-01-05 12:00:00"。
    """
    try:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        return formatted_datetime
    except Exception as e:
        return f"获取时间时发生错误: {str(e)}"


@tool
def get_current_date() -> str:
    """
    获取当前日期的工具。
    Agent 可以调用此工具来了解当前的日期。
    
    Returns:
        格式化后的日期字符串，例如 "2024-01-05"。
    """
    try:
        current_date = datetime.now()
        formatted_date = current_date.strftime("%Y-%m-%d")
        return formatted_date
    except Exception as e:
        return f"获取日期时发生错误: {str(e)}"


@tool
def get_current_time() -> str:
    """
    获取当前时间的工具。
    Agent 可以调用此工具来了解当前的时间。
    
    Returns:
        格式化后的时间字符串，例如 "12:00:00"。
    """
    try:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%H:%M:%S")
        return formatted_time
    except Exception as e:
        return f"获取时间时发生错误: {str(e)}"

