import os
from dotenv import load_dotenv  # 加载环境变量
from langchain_community.document_loaders import TextLoader  # 文本加载器
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 文本分割器
from langchain_community.vectorstores import Chroma  # 向量存储
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 聊天模型和嵌入模型
from langchain.prompts import PromptTemplate  # 提示模板
from langchain_core.embeddings import Embeddings  # 嵌入模型
from typing import List  # 类型注解
from langchain_community.embeddings import DashScopeEmbeddings

# 加载环境变量
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")  # 文档目录
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db_langchain")  # 向量数据库目录


def create_embeddings():  # 创建嵌入模型
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 文件中配置 DASHSCOPE_API_KEY")
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=api_key, model="text-embedding-v2"
    )
    test_embedding = embeddings.embed_query("test")
    return embeddings


def demo_document_processing():  # 演示文档处理模块
    # 创建示例文档
    sample_docs = [
        {
            "filename": os.path.join(DOCUMENTS_DIR, "langchain_intro.txt"),
            "content": """LangChain 简介

            LangChain 是一个用于构建 LLM 应用的框架。

            核心概念：
            1. Chains（链）：将多个组件连接起来
            2. Agents（代理）：能够自主决策和执行任务
            3. Memory（记忆）：维护对话历史
            4. Tools（工具）：扩展 LLM 的能力
            5. Document Processing（文档处理）：加载和预处理文档
            6. Vector Stores（向量存储）：存储和检索文档向量

            主要功能：
            - 文档加载和预处理
            - 向量数据库集成
            - 检索增强生成（RAG）
            - 工具调用和 Agent 构建

            使用场景：
            - 构建智能问答系统
            - 文档检索和分析
            - 自动化任务处理
            - 知识库构建
            """,
        },
        {
            "filename": os.path.join(DOCUMENTS_DIR, "python_basics.txt"),
            "content": """Python 基础语法

            Python 是一种高级编程语言，具有简洁明了的语法。

            变量和数据类型：
            - 整数：x = 10
            - 浮点数：y = 3.14
            - 字符串：name = "Python"
            - 列表：numbers = [1, 2, 3]

            控制流：
            - if/else 语句用于条件判断
            - for 循环用于遍历序列
            - while 循环用于重复执行

            函数定义：
            def greet(name):
                return f"Hello, {name}!"

            Python 的特点：
            - 语法简洁，易于学习
            - 丰富的标准库和第三方库
            - 跨平台支持
            - 广泛用于 Web 开发、数据科学、人工智能等领域""",
        },
    ]
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)  # 确保文档目录存在

    documents = []
    for doc_info in sample_docs:
        filepath = doc_info["filename"]
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:  # 写入文档内容
            f.write(doc_info["content"])
        loader = TextLoader(filepath, encoding="utf-8")  # 使用 TextLoader 加载文档
        loaded_docs = loader.load()  # 加载文档
        documents.extend(loaded_docs)  # 将加载的文档添加到 documents 列表中

    # 在循环外部创建文本分割器并分割文档
    text_splitter = (
        RecursiveCharacterTextSplitter(  # 使用 RecursiveCharacterTextSplitter 分割文档
            chunk_size=200,  # 每个片段的最大字符数
            chunk_overlap=50,  # 片段之间的重叠字符数
            length_function=len,
        )
    )
    splits = text_splitter.split_documents(documents)  # 分割文档

    return splits  # 返回分割后的文档片段


def demo_vector_stores(splits, embeddings):  # 演示向量存储模块
    vectorstore = Chroma.from_documents(  # 创建向量数据库
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,  # 持久化目录
    )

    test_queries = [
        "LangChain 有哪些核心概念？",
        "Python 的基本数据类型有哪些？",
        "文档处理的作用是什么？",
    ]
    for query in test_queries:
        print(f"  查询: {query}")
        results = vectorstore.similarity_search(query, k=2)  # 返回最相似的2个片段
        print(f"  ✓ 检索到 {len(results)} 个相关片段：")
        for i, doc in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            content_preview = doc.page_content[:80].replace("\n", " ")
            print(f"    [{i}] 来源: {source}")
            print(f"        内容: {content_preview}...")
        print()

    query = "LangChain 的核心功能"
    results_with_scores = vectorstore.similarity_search_with_score(
        query, k=2
    )  # 返回最相似的2个片段及其相似度分数
    for i, (doc, score) in enumerate(results_with_scores, 1):  # 遍历结果
        source = os.path.basename(doc.metadata.get("source", "未知"))
        content_preview = doc.page_content[:80].replace("\n", " ")
        print(f"    [{i}] 相似度分数: {score:.4f}")
        print(f"        来源: {source}")
        print(f"        内容: {content_preview}...")
    print()

    return vectorstore


def demo_rag_with_vectorstore(vectorstore):  # 演示结合向量存储的 RAG 模块
    api_key = os.getenv("DASHSCOPE_API_KEY")
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # 创建检索器
    # 自定义提示词模板
    prompt_template = """基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。
        上下文：
        {context}
        问题：{question}
        请用中文回答："""
    PROMPT = PromptTemplate(  # 创建提示词模板
        template=prompt_template, input_variables=["context", "question"]
    )

    test_question = "LangChain 有哪些核心概念？"
    docs = retriever.get_relevant_documents(test_question)  # 检索相关文档
    context = "\n\n".join([doc.page_content for doc in docs])  # 构建上下文
    prompt = PROMPT.format(context=context, question=test_question)
    response = llm.invoke(prompt)
    print(f"\n  ✓ 答案生成成功：")
    print(f"  {response.content if hasattr(response, 'content') else str(response)}")
    print(f"\n  参考文档来源：")
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "未知"))
        print(f"    [{i}] {source}")
    print()


def main():
    embeddings = create_embeddings()
    # 2. 演示 Document Processing
    splits = demo_document_processing()

    # 3. 演示 Vector Stores
    vectorstore = demo_vector_stores(splits, embeddings)

    # 4. 演示 RAG
    demo_rag_with_vectorstore(vectorstore)


if __name__ == "__main__":
    main()
