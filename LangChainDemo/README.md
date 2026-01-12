# LangChain 教学 Demo

这是一个基于 LangChain 框架和通义千问大模型的教学演示项目，展示了 LangChain 的核心功能模块。

## 📚 项目简介

本项目通过完整的 Demo，演示了 LangChain 框架的 7 个核心模块：

1. **Models（模型）** - 封装和调用大语言模型，提供统一的接口
2. **Tools（工具）** - 扩展 AI 能力的外部功能
3. **Agents（代理）** - 智能代理，采用 ReAct 模式
4. **Memory（记忆）** - 多轮对话记忆支持
5. **Document Processing（文档处理）** - 文档加载和分割
6. **Vector Stores（向量存储）** - 向量存储和检索
7. **Retrieval（检索）** - 检索增强生成（RAG）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件，并配置你的 API Key：

```bash
# 复制 env_example.txt 为 .env
cp env_example.txt .env
```

编辑 `.env` 文件，填入你的 `DASHSCOPE_API_KEY`：

```
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 运行 Demo

**基础功能演示（Models、Tools、Agents、Memory）：**
```bash
python main.py
```

**文档处理和向量存储演示（Document Processing、Vector Stores、Retrieval）：**
```bash
python document_vector_demo.py
```

## 📖 项目结构

```
LangChainDemo/
├── main.py                    # 主程序，展示 Models、Tools、Agents、Memory
├── document_vector_demo.py     # 文档处理和向量存储演示
├── requirements.txt           # 项目依赖
├── README.md                  # 项目说明文档
├── env_example.txt            # 环境变量示例文件
├── tools/                     # 工具模块目录
│   ├── __init__.py           # 包初始化文件
│   ├── math_tools.py         # 数学运算工具
│   └── time_tool.py          # 时间查询工具
└── 1.md                       # LangChain 教学文档
```

## 🎯 核心功能演示

### 1. Models 模块

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
)
```

**作用：**
- 统一接口：解决不同模型提供商 API 差异问题
- 参数管理：集中管理 API 密钥、temperature 等参数
- 集成方便：天然适配于 LangChain 的其他组件
- 错误处理：内置重试、错误处理机制

### 2. Tools 模块

```python
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b
```

**作用：**
- 功能扩展：扩展 AI 的外部能力
- 系统交互：与外部系统交互
- 任务执行：执行计算任务等

**@tool 装饰器的作用：**
- 自动生成工具描述：从函数文档字符串提取
- 自动生成工具 Schema：将函数签名转换为 JSON Schema
- 工具注册：将工具注册到 Agent 的工具列表中

### 3. Agents 模块

```python
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=5,
)
```

**作用：**
- 初始化 Agent：构建 ReAct 模式的 Agent
- 循环管理：自动管理 ReAct 循环
- 工具调用：根据 Agent 决策调用工具
- 错误处理：处理解析错误和工具执行异常
- 迭代控制：限制最大迭代次数

### 4. Memory 模块

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=False,
)
```

**作用：**
- 上下文连贯：Agent 能理解多轮对话的上下文
- 信息持久化：在单次会话中记住用户提供的信息
- 智能对话：支持需要多轮交互的复杂任务
- 自动管理：无需手动维护对话历史列表

### 5. Document Processing 模块

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 文档加载
loader = TextLoader("documents/langchain_intro.txt", encoding="utf-8")
documents = loader.load()

# 文档分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)
splits = text_splitter.split_documents(documents)
```

**作用：**
- 文档加载：从各种数据源加载文档（文件、数据库、API等）
- 文档分割：将长文档分割成较小的片段，便于向量化和检索
- 格式转换：支持多种文档格式（TXT、PDF、CSV、HTML等）
- 元数据提取：提取文档的元数据信息

### 6. Vector Stores 模块

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建嵌入模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-v2",
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 创建向量数据库
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 相似度检索
results = vectorstore.similarity_search("LangChain 有哪些核心概念？", k=2)
```

**作用：**
- 向量存储：将文档向量存储到向量数据库中
- 相似度检索：根据查询向量检索最相似的文档片段
- 持久化存储：支持将向量数据库持久化到磁盘
- 多种后端：支持 Chroma、Pinecone、Weaviate 等多种向量数据库

### 7. Retrieval 模块（RAG）

```python
from langchain.chains import RetrievalQA

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 创建 RAG 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# 使用 RAG 回答问题
result = qa_chain.invoke({"query": "LangChain 有哪些核心概念？"})
```

**作用：**
- 检索增强生成：将检索到的文档作为上下文，与问题一起发送给 LLM
- 基于知识的回答：让 LLM 基于实际文档内容回答问题
- 减少幻觉：提供可追溯的答案来源
- 实时更新：支持实时更新知识库

## 💡 使用示例

启动程序后，你可以尝试以下问题：

```
用户: 计算 123 + 456
助手: 123 + 456 = 579

用户: 现在几点了？
助手: 当前时间是 2024-01-05 14:30:00

用户: 今天是几号？
助手: 今天是 2024-01-05

用户: 先计算 10 * 20，然后告诉我结果
助手: 10 * 20 = 200
```

**文档处理和向量存储示例（运行 `document_vector_demo.py`）：**

```
【模块 5】Document Processing（文档处理）演示
  ✓ 加载文档: langchain_intro.txt
  ✓ 文档分割完成

【模块 6】Vector Stores（向量存储）演示
  ✓ 向量数据库创建成功
  ✓ 检索到相关文档片段

RAG（检索增强生成）演示
  ✓ 答案生成成功：LangChain 的核心概念包括...
```

## 🔧 工具列表

### 数学工具（math_tools.py）
- `add(a, b)` - 加法运算
- `subtract(a, b)` - 减法运算
- `multiply(a, b)` - 乘法运算
- `divide(a, b)` - 除法运算

### 时间工具（time_tool.py）
- `get_current_datetime()` - 获取当前日期和时间
- `get_current_date()` - 获取当前日期
- `get_current_time()` - 获取当前时间

## 📝 注意事项

1. 确保已配置 `DASHSCOPE_API_KEY` 环境变量
2. 需要 Python 3.8+ 版本
3. 首次运行会自动安装依赖包

## 🎓 学习资源

- [LangChain 官方文档](https://python.langchain.com/)
- [通义千问 API 文档](https://help.aliyun.com/zh/dashscope/)

## 📄 许可证

本项目仅用于教学演示目的。

