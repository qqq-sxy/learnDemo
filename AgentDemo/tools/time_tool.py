from datetime import datetime


def get_current_datetime() -> str:
    """
    获取当前日期和时间的工具。
    Agent 可以调用此工具来了解当前的实时时间。
    :return: 格式化后的日期时间字符串，例如 "2024-01-05 12:00:00"。
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime
