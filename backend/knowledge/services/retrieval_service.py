import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List
from langchain_core.documents import Document

from backend.knowledge.repositories.vector_store_reposity import VectorStoreRepository


class RetrievalService:
    """
        负责检索的类（检索器）
    """

    def __init__(self):
        self.vector_store_repository = VectorStoreRepository()

    def retrieve(self, user_question: str) -> List[Document]:
        """
            核心检索方法：检索器的入口
        Args:
            user_question: 用户输入的问题

        Returns:
            List[Document]：返回指定 TOP-N 个相似的文档列表
        """

        # 1. 第一路检索（基于嵌入模型的向量检索）
        based_vector_candidates = self._search_based_vector(user_question)

        # 2. 第二路检索(基于 jieba 分词匹配)
        based_title_candidates = self._search_based_title(user_question)

        # 3. 合并两路检索的文档列表
        total_candidates = based_vector_candidates + based_title_candidates

        # 4. 对合并后的文档列表去重
        unique_candidates = self._deduplicate(total_candidates)

        # 5. 重新打分排序
        top_documents = self._reranking(unique_candidates)

        # 6. 返回指定 TOP-N 个相似的文档列表
        return top_documents

    def _search_based_vector(self, user_question: str) -> List[Document]:
        """
        第一路检索
        基于语义相似度检查
        Args:
            user_question: 用户输入的问题

        Returns:
            List[Document]： 相似的 TOP-N 个文档列表
        """
        # 1. 返回带分数的文档列表
        documents_with_score = self.vector_store_repository.search_similarity_with_scores(user_question)

        # 2. TODO 不用距离得分
        base_vector_candidates = []
        for document, score in documents_with_score:
            base_vector_candidates.append(document)

        return base_vector_candidates

    def _search_based_title(self, user_question: str) -> List[Document]:
        """
        第二路检索
        基于 标题的关键词 匹配搜索
        Args:
            user_question: 用户输入的问题

        Returns:
            List[Document]： 相似的 TOP-N 个文档列表
        """
        pass

    def _deduplicate(self, total_candidates: List[Document]) -> List[Document]:
        """
        对合并后的文档列表去重
        用 set() 集合去重，（（title, 内容前「100」个字符）） ----> key
        Args:
            total_candidates: 合并的文档列表

        Returns:
            List[Document]： 唯一的文档列表
        """
        pass

    def _reranking(self, unique_candidates: List[Document], user_question: str) -> List[Document]:
        """
        重新计算打分 && 排序
        Args:
            unique_candidates: 唯一的候选文档列表
            user_question: 用户输入的问题
        Returns:
            List[Document]：最终指定的 TOP-N 个文档列表
        """
        pass
