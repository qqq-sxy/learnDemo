"""
LangChain Document Processing 和 Vector Stores 模块演示
展示 LangChain 框架的文档处理和向量存储功能：
5. Document Processing - 文档处理（加载、分割）
6. Vector Stores - 向量存储（存储、检索）
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from typing import List

# 加载环境变量
load_dotenv()

# 获取脚本所在目录（LangChainDemo 目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db_langchain")


class DashScopeEmbeddingsCustom(Embeddings):
    """
    自定义 DashScope 嵌入类
    用于适配通义千问的嵌入 API
    """
    def __init__(self, dashscope_api_key: str, model: str = "text-embedding-v2"):
        try:
            import dashscope
            dashscope.api_key = dashscope_api_key
            self.dashscope = dashscope
            self.model = model
        except ImportError:
            raise ImportError("请安装 dashscope: pip install dashscope")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        try:
            response = self.dashscope.embeddings.call(
                model=self.model,
                input=texts
            )
            if response.status_code == 200:
                return [item['embedding'] for item in response.output['embeddings']]
            else:
                raise Exception(f"API 调用失败: {response.message}")
        except Exception as e:
            raise Exception(f"嵌入文档时发生错误: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本"""
        return self.embed_documents([text])[0]


def create_embeddings():
    """
    创建嵌入模型（用于向量化）
    
    优先尝试以下方式：
    1. langchain_community 中的 DashScopeEmbeddings（如果存在）
    2. 自定义 DashScopeEmbeddingsCustom（使用 dashscope SDK）
    3. OpenAI 兼容模式（可能不兼容）
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 文件中配置 DASHSCOPE_API_KEY")
    
    # 方法1: 尝试使用 langchain_community 中的 DashScopeEmbeddings
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        embeddings = DashScopeEmbeddings(
            dashscope_api_key=api_key,
            model="text-embedding-v2"
        )
        test_embedding = embeddings.embed_query("test")
        print(f"✓ 嵌入模型初始化成功（使用 langchain_community.DashScopeEmbeddings，向量维度: {len(test_embedding)}）")
        return embeddings
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠ langchain_community.DashScopeEmbeddings 初始化失败: {e}")
    
    # 方法2: 使用自定义 DashScopeEmbeddingsCustom（使用 dashscope SDK）
    try:
        embeddings = DashScopeEmbeddingsCustom(
            dashscope_api_key=api_key,
            model="text-embedding-v2"
        )
        test_embedding = embeddings.embed_query("test")
        print(f"✓ 嵌入模型初始化成功（使用自定义 DashScopeEmbeddingsCustom，向量维度: {len(test_embedding)}）")
        return embeddings
    except ImportError:
        print("⚠ dashscope SDK 未安装，尝试使用 OpenAI 兼容模式...")
    except Exception as e:
        print(f"⚠ 自定义 DashScopeEmbeddingsCustom 初始化失败: {e}")
    
    # 方法3: 尝试使用 OpenAI 兼容模式（可能不兼容）
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-v2",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        test_embedding = embeddings.embed_query("test")
        print(f"✓ 嵌入模型初始化成功（使用 OpenAI 兼容模式，向量维度: {len(test_embedding)}）")
        return embeddings
    except Exception as e:
        error_msg = str(e)
        print(f"⚠ OpenAI 兼容模式初始化失败: {error_msg}")
        raise Exception(
            f"所有嵌入模型初始化方法都失败了。\n"
            f"请尝试以下解决方案：\n"
            f"1. 安装 dashscope SDK: pip install dashscope\n"
            f"2. 或者安装 langchain-community: pip install langchain-community\n"
            f"3. 检查 API Key 是否正确配置\n"
            f"4. 错误详情: {error_msg}"
        )


def demo_document_processing():
    """
    演示 Document Processing 模块
    
    Document Processing 模块作用：
    - 文档加载：从各种数据源加载文档（文件、数据库、API等）
    - 文档分割：将长文档分割成较小的片段，便于向量化和检索
    - 格式转换：支持多种文档格式（TXT、PDF、CSV、HTML等）
    - 元数据提取：提取文档的元数据信息
    """
    print("=" * 80)
    print("【模块 5】Document Processing（文档处理）演示")
    print("=" * 80)
    print()
    
    # ==================== 步骤1: 文档加载 ====================
    print("【步骤 1】文档加载（Document Loading）")
    print("-" * 80)
    print("作用：从各种数据源加载文档（文件、数据库、API等）")
    print("      支持多种格式：TXT、PDF、CSV、HTML、Markdown等")
    print()
    
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
"""
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
- 广泛用于 Web 开发、数据科学、人工智能等领域
"""
        }
    ]
    
    # 确保文档目录存在（在 LangChainDemo 目录下）
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    # 创建文档文件并加载
    documents = []
    for doc_info in sample_docs:
        filepath = doc_info["filename"]
        # 确保文件所在目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 写入文档内容
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc_info["content"])
        
        # 使用 TextLoader 加载文档
        loader = TextLoader(filepath, encoding="utf-8")
        loaded_docs = loader.load()
        documents.extend(loaded_docs)
        print(f"  ✓ 加载文档: {os.path.basename(filepath)}")
        print(f"    内容长度: {len(loaded_docs[0].page_content)} 字符")
        print(f"    元数据: {loaded_docs[0].metadata}")
    
    print(f"\n  共加载 {len(documents)} 个文档")
    print()
    
    # ==================== 步骤2: 文档分割 ====================
    print("【步骤 2】文档分割（Text Splitting）")
    print("-" * 80)
    print("作用：将长文档分割成较小的片段，便于向量化和检索")
    print("      每个片段包含一定数量的字符，片段之间有重叠")
    print("      重叠可以保证上下文的连贯性")
    print()
    
    # 使用 RecursiveCharacterTextSplitter 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # 每个片段的最大字符数
        chunk_overlap=50,    # 片段之间的重叠字符数
        length_function=len,
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"  ✓ 文档分割完成")
    print(f"  - 原始文档数: {len(documents)}")
    print(f"  - 分割后片段数: {len(splits)}")
    print(f"  - 平均每个片段长度: {sum(len(s.page_content) for s in splits) // len(splits)} 字符")
    print()
    
    # 显示一个分割示例
    if splits:
        print("  示例片段（第一个）：")
        print(f"  {splits[0].page_content[:100]}...")
        print()
    
    return splits


def demo_vector_stores(splits, embeddings):
    """
    演示 Vector Stores 模块
    
    Vector Stores 模块作用：
    - 向量存储：将文档向量存储到向量数据库中
    - 相似度检索：根据查询向量检索最相似的文档片段
    - 持久化存储：支持将向量数据库持久化到磁盘
    - 多种后端：支持 Chroma、Pinecone、Weaviate 等多种向量数据库
    """
    print("=" * 80)
    print("【模块 6】Vector Stores（向量存储）演示")
    print("=" * 80)
    print()
    
    # ==================== 步骤1: 向量存储 ====================
    print("【步骤 1】向量存储（Vector Storage）")
    print("-" * 80)
    print("作用：将文档向量存储到向量数据库中，支持快速相似度搜索")
    print("      使用嵌入模型将文本转换为向量，然后存储到向量数据库")
    print()
    
    # 创建向量数据库（使用 Chroma）
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR  # 持久化目录（在 LangChainDemo 目录下）
    )
    
    print(f"  ✓ 向量数据库创建成功")
    print(f"  - 存储位置: {CHROMA_DB_DIR}")
    print(f"  - 存储的文档片段数: {len(splits)}")
    print(f"  - 向量维度: {len(embeddings.embed_query('test'))}")
    print()
    
    # ==================== 步骤2: 相似度检索 ====================
    print("【步骤 2】相似度检索（Similarity Search）")
    print("-" * 80)
    print("作用：根据用户查询，从向量数据库中检索最相关的文档片段")
    print("      使用向量相似度计算，找到语义最接近的文档")
    print()
    
    # 测试查询
    test_queries = [
        "LangChain 有哪些核心概念？",
        "Python 的基本数据类型有哪些？",
        "文档处理的作用是什么？"
    ]
    
    for query in test_queries:
        print(f"  查询: {query}")
        results = vectorstore.similarity_search(query, k=2)  # 返回最相似的2个片段
        print(f"  ✓ 检索到 {len(results)} 个相关片段：")
        for i, doc in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get('source', '未知'))
            content_preview = doc.page_content[:80].replace('\n', ' ')
            print(f"    [{i}] 来源: {source}")
            print(f"        内容: {content_preview}...")
        print()
    
    # ==================== 步骤3: 带分数的检索 ====================
    print("【步骤 3】带相似度分数的检索（Similarity Search with Score）")
    print("-" * 80)
    print("作用：检索相关文档并返回相似度分数，分数越高表示越相似")
    print()
    
    query = "LangChain 的核心功能"
    results_with_scores = vectorstore.similarity_search_with_score(query, k=2)
    
    print(f"  查询: {query}")
    print(f"  ✓ 检索结果（带分数）：")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        source = os.path.basename(doc.metadata.get('source', '未知'))
        content_preview = doc.page_content[:80].replace('\n', ' ')
        print(f"    [{i}] 相似度分数: {score:.4f}")
        print(f"        来源: {source}")
        print(f"        内容: {content_preview}...")
    print()
    
    return vectorstore


def demo_rag_with_vectorstore(vectorstore):
    """
    演示结合 Vector Stores 的 RAG（检索增强生成）
    """
    print("=" * 80)
    print("RAG（检索增强生成）演示")
    print("=" * 80)
    print()
    print("作用：将检索到的文档作为上下文，与问题一起发送给 LLM")
    print("      让 LLM 基于检索到的知识生成答案，而不是仅依赖训练数据")
    print()
    
    # 初始化 LLM
    api_key = os.getenv("DASHSCOPE_API_KEY")
    llm = ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
    )
    
    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 自定义提示词模板
    prompt_template = """基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的文档中找到答案。

上下文：
{context}

问题：{question}

请用中文回答："""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # 测试 RAG
    test_question = "LangChain 有哪些核心概念？"
    print(f"  问题: {test_question}")
    print("  正在检索相关文档...")
    
    # 检索相关文档
    docs = retriever.get_relevant_documents(test_question)
    print(f"  ✓ 检索到 {len(docs)} 个相关文档片段")
    
    # 构建上下文
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 使用 LLM 生成答案
    print("  正在生成答案...")
    prompt = PROMPT.format(context=context, question=test_question)
    response = llm.invoke(prompt)
    
    print(f"\n  ✓ 答案生成成功：")
    print(f"  {response.content if hasattr(response, 'content') else str(response)}")
    print(f"\n  参考文档来源：")
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get('source', '未知'))
        print(f"    [{i}] {source}")
    print()


def main():
    """
    主函数：演示 Document Processing 和 Vector Stores 模块
    """
    try:
        # 1. 创建嵌入模型
        print("初始化嵌入模型...")
        embeddings = create_embeddings()
        print()
        
        # 2. 演示 Document Processing
        splits = demo_document_processing()
        
        # 3. 演示 Vector Stores
        vectorstore = demo_vector_stores(splits, embeddings)
        
        # 4. 演示 RAG
        demo_rag_with_vectorstore(vectorstore)
        
        # ==================== 总结 ====================
        print("=" * 80)
        print("Document Processing 和 Vector Stores 模块总结")
        print("=" * 80)
        print("""
【Document Processing（文档处理）模块】
1. 文档加载：从各种数据源加载文档，支持多种格式
2. 文档分割：将长文档分割成较小的片段，提高检索精度
3. 格式转换：支持 TXT、PDF、CSV、HTML、Markdown 等格式
4. 元数据提取：提取文档的元数据信息

【Vector Stores（向量存储）模块】
1. 向量存储：将文档向量存储到向量数据库中
2. 相似度检索：根据查询向量检索最相似的文档片段
3. 持久化存储：支持将向量数据库持久化到磁盘
4. 多种后端：支持 Chroma、Pinecone、Weaviate 等

【两个模块的协作】
- Document Processing 负责文档的预处理（加载、分割）
- Vector Stores 负责文档向量的存储和检索
- 两者结合实现 RAG（检索增强生成）功能
- 让 LLM 能够基于实际文档内容回答问题
""")
        print("=" * 80)
        
    except ValueError as e:
        print(f"\033[91m配置错误: {e}\033[0m")
        print("提示：请确保已创建 .env 文件并配置 DASHSCOPE_API_KEY")
    except Exception as e:
        print(f"\033[91m发生错误: {e}\033[0m")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

