import logging

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List

from backend.knowledge.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreRepository:

    def __init__(self):
        """
        创建向量数据库实例
        创建嵌入模型实例
        向量数据库能力：1. 存储向量数据 2.搜索能力（向量数据库检索器）
        """
        self.embedding = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_base=settings.BASE_URL,
            openai_api_key=settings.API_KEY
        )

        self.vector_database = Chroma(
            persist_directory=settings.VECTOR_STORE_PATH,
            collection_name="its-knowledge",
            embedding_function=self.embedding
        )

    def add_documents(self, documents: list, batch_size: int = 16) -> int:
        """
        将切分之后的文档快保存到向量数据库中
        Args:
            self:
            documents: 切分之后的文档快
            batch_size: 分批保存批次大小（最大允许16）

        Returns:
            int： 成功添加到向量数据库中文档块的数量
        """
        # 1. 文档块的总数量
        total_documents_chunks = len(documents)

        # 2. 分批次保存
        # 场景： documents:[1,2,3,4,5] batch：2 遍历3次 [1,2] [3,4] [5]
        documents_chunks_added = 0
        try:
            for i in range(0, total_documents_chunks, batch_size):
                bath = documents[i:i + batch_size]

                self.vector_database.add_documents(bath)
                documents_chunks_added += len(bath)
                logger.info(f"成功将文档块：{documents_chunks_added}/{total_documents_chunks}保存到向量数据库...")
            return documents_chunks_added
        except Exception as e:
            logger.error(f"文档块列表:{documents}保存到向量数据库失败：{str(e)}")
            raise e

    def embed_query(self, text: str) -> list[float]:
        """
            对 query 进行向量化
        Args:
            text: 输入文本

        Returns:
            list[float]： 嵌入后的浮点数列表
        """
        return self.embedding.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
            对 字符串列表 进行向量化
        Args:
            texts: 输入文本字符串列表

        Returns:
            list[list[float]]： 嵌入后的浮点数列表
        """
        return self.embedding.embed_documents(texts)

    def search_similarity_with_scores(self, user_question: str, top_k: int = 5) -> List[tuple[Document, float]]:
        """
        相似性检索带文档分数
        分数（chroma向量数据库）：返回 L2 距离得分（分数越小越相似）
            不是余弦相似度（越高越相似） 距离得分：1——余弦相似度得分
        Args:
            user_question: 用户输入的问题
            top_k: 返回得分最高的相似文档个数

        Returns:
            List[tuple[Document, float]]: 返回带有分数的相似文档元组 列表
        """
        return self.vector_database.similarity_search_with_score(user_question, top_k)
