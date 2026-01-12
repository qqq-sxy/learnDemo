import inspect

def function_to_json(func) -> dict:
    """
    通过 Python 反射机制，自动将一个 Python 函数转换为 OpenAI/DashScope 兼容的工具定义 (JSON Schema)。
    这使得我们只需要编写普通的 Python 函数，就能直接被 Agent 识别和调用。
    """
    # 获取函数的签名信息（包含参数名、参数类型注解、默认值等）
    sig = inspect.signature(func)
    parameters = {}
    required = []

    # 遍历函数的所有参数
    for name, param in sig.parameters.items():
        # 根据 Python 的类型注解推断 JSON Schema 的类型
        if param.annotation == float:
            param_type = "number"
        elif param.annotation == int:
            param_type = "integer"
        elif param.annotation == str:
            param_type = "string"
        elif param.annotation == bool:
            param_type = "boolean"
        else:
            # 默认设为 string
            param_type = "string"

        # 构造单个参数的 schema 定义
        parameters[name] = {"type": param_type}

        # 如果参数没有设定默认值，则在 Schema 中标记为必填项 (required)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    # 返回符合 OpenAI Tool 定义规范的字典结构
    return {
        "type": "function",
        "function": {
            "name": func.__name__,                      # 函数名
            "description": inspect.getdoc(func) or "",  # 使用函数的 docstring 作为工具描述
            "parameters": {
                "type": "object",
                "properties": parameters,                # 参数列表
                "required": required,                    # 必填参数列表
            },
        },
    }
