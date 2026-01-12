"""
数学工具模块
用于执行数学运算
"""
from langchain.tools import tool


@tool
def add(a: int, b: int) -> int:
    """
    计算两个整数的和。
    Agent 会在需要进行加法运算时自动调用此工具。
    
    Args:
        a: 第一个整数
        b: 第二个整数
        
    Returns:
        两个整数的和
    """
    try:
        # 处理字符串类型的输入（Agent 可能传递字符串格式的数字）
        a = int(a) if isinstance(a, str) else a
        b = int(b) if isinstance(b, str) else b
        return a + b
    except (ValueError, TypeError) as e:
        return f"计算时发生错误: 无法将输入转换为整数 - {str(e)}"
    except Exception as e:
        return f"计算时发生错误: {str(e)}"


@tool
def subtract(a: int, b: int) -> int:
    """
    计算两个整数的差。
    Agent 会在需要进行减法运算时自动调用此工具。
    
    Args:
        a: 被减数
        b: 减数
        
    Returns:
        两个整数的差
    """
    try:
        # 处理字符串类型的输入（Agent 可能传递字符串格式的数字）
        a = int(a) if isinstance(a, str) else a
        b = int(b) if isinstance(b, str) else b
        return a - b
    except (ValueError, TypeError) as e:
        return f"计算时发生错误: 无法将输入转换为整数 - {str(e)}"
    except Exception as e:
        return f"计算时发生错误: {str(e)}"


@tool
def multiply(a: int, b: int) -> int:
    """
    计算两个整数的乘积。
    Agent 会在需要进行乘法运算时自动调用此工具。
    
    Args:
        a: 第一个整数
        b: 第二个整数
        
    Returns:
        两个整数的乘积
    """
    try:
        # 处理字符串类型的输入（Agent 可能传递字符串格式的数字）
        a = int(a) if isinstance(a, str) else a
        b = int(b) if isinstance(b, str) else b
        return a * b
    except (ValueError, TypeError) as e:
        return f"计算时发生错误: 无法将输入转换为整数 - {str(e)}"
    except Exception as e:
        return f"计算时发生错误: {str(e)}"


@tool
def divide(a: float, b: float) -> float:
    """
    计算两个数的商。
    Agent 会在需要进行除法运算时自动调用此工具。
    
    Args:
        a: 被除数
        b: 除数
        
    Returns:
        两个数的商
    """
    try:
        # 处理字符串类型的输入（Agent 可能传递字符串格式的数字）
        a = float(a) if isinstance(a, str) else a
        b = float(b) if isinstance(b, str) else b
        
        if b == 0:
            return "错误: 除数不能为0"
        return a / b
    except (ValueError, TypeError) as e:
        return f"计算时发生错误: 无法将输入转换为数字 - {str(e)}"
    except Exception as e:
        return f"计算时发生错误: {str(e)}"

