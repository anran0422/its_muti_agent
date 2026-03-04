import logging

import jieba

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Dict, Any
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from backend.knowledge.utils.markdown_utils import MarkDownUtils
from backend.knowledge.repositories.vector_store_reposity import VectorStoreRepository
from backend.knowledge.config.settings import settings


class RetrievalService:
    """
        负责检索的类（检索器）
    """

    def __init__(self):
        self.chroma_vector = VectorStoreRepository()

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
        documents_with_score = self.chroma_vector.search_similarity_with_scores(user_question)

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
        # 1. 获取指定目录下的文件的标题
        mds_metadata = MarkDownUtils.collect_md_metadata(settings.CRAWL_OUTPUT_DIR)

        # 2. 进行标题匹配
        # 2.1 关键词匹配（jieba）---> 比较对象：用户输入的问题 vs crawl 目录下的文件标题
        # 2.2 标题的语义匹配 ---> 比较对象：用户输入的问题 vs crawl md 目录下的
        rough_mds_metadata = self.rough_ranking(user_question, mds_metadata)
        fine_mds_metadata = self.fine_ranking(user_question, rough_mds_metadata)

        # 3. 返回指定文档列表

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

    def rough_ranking(self, user_question, mds_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对标题进行粗排
        基于 jieba 进行标题的分词匹配
        Args:
            user_question: 用户的问题
            mds_metadata: 所有 md 的元数据（标题【title】， 路径 【path】）

        Returns:
            List[Dict[str, Any]]：所有 md 的元数据（标题【title】， 路径 【path】， 标题的粗排得分【rough_score】）
        """
        # 1. 用户输入问题是否存在
        if not user_question:
            return []
        ROUGH_WORD_WEIGHT = 0.7
        # 2. 遍历 mds_metadata（所有md的元数据）
        for md_metadata in mds_metadata:
            # 2.1 获取 md 标题
            md_metadata_title = md_metadata["title"]

            # 2.2 判断标题是否存在
            if not md_metadata_title and not md_metadata_title.strip():
                continue

            # 2.3 进行分词 && 算得分
            # 2.3.1 优先字符切 set 交并差 jarcard 算法
            user_question_char = set(user_question)
            md_metadata_title_char = set(md_metadata_title)
            unique_char = user_question_char | md_metadata_title_char
            char_score = len(user_question_char & md_metadata_title_char) / len(unique_char) if len(
                unique_char) > 0 else 0

            # 2.3.2 再用 jieba 词项切，影响因素大一些
            user_question_word = set(jieba.lcut(user_question))
            md_metadata_title_word = set(jieba.lcut(md_metadata_title))
            unique_word = user_question_word | md_metadata_title_word
            word_score = len(user_question_word & md_metadata_title_word) / len(unique_word) if len(
                unique_word) > 0 else 0

            # 2.3.3 计算粗排分数：字符集 + 词性项级
            rough_score = word_score * ROUGH_WORD_WEIGHT + char_score * (1 - ROUGH_WORD_WEIGHT)

            md_metadata['rough_score'] = rough_score

        # 3. 根据标题的元数据（rough_score)排序，并且留下前 50 个
        return sorted(mds_metadata, key=lambda x: x['rough_score'], reverse=True)[:50]

    def fine_ranking(self, user_question: str, rough_mds_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对标题进行精排
        基于嵌入模型相似性以及 cosine_similarity()
        Args:
            rough_mds_metadata: 粗排后的 md 元数据
            user_question: 用户输入的问题

        Returns:
            List[Dict[str, Any]]: 带精排分数的元数据
        """
        # 1. 判断粗排元数据
        if not rough_mds_metadata:
            return []

        # 两个二维矩阵（X[样本数] Y[样本质量]）
        # 2. 对问题向量化
        question_embedding = self.chroma_vector.embed_query(user_question)

        # 3. 获取粗排后的标题
        rough_title = [md_metadata['title'] for md_metadata in rough_mds_metadata]

        # 4. 标题的向量值
        rough_title_embeddings = self.chroma_vector.embed_documents(rough_title)

        # 5. 计算问题和粗排标题的相似度（余弦相似度）
        # X:1 Y[1,2,3,4,5] similarity=[0.1, 0.4, 0.01, 0.6, 0.3][-1,0]
        similarity = cosine_similarity([question_embedding], rough_title_embeddings).flatten()

        # 6. 遍历粗排元数据
        ROUGH_WEIGHT = 0.3
        SIM_WEIGHT = 0.7
        for index, md_metadata in enumerate(rough_mds_metadata):
            # a. 获取精排分数（归一化处理）
            simi = similarity[index]
            if simi < 0:
                simi = 0
            # b. 获取粗排
            rough_score = md_metadata['rough_score']

            # c. 加权求取最终分数
            final_score = rough_score * ROUGH_WEIGHT + simi * SIM_WEIGHT

            # d. 存放到 md_metadata 中
            md_metadata['simi_score'] = simi
            md_metadata['final_score'] = final_score

        simi_mds_metadata = sorted(rough_mds_metadata, key=lambda x: x['final_score'], reverse=True)[:5]
        return simi_mds_metadata


if __name__ == '__main__':
    retrieval_service = RetrievalService()

    rough_ranking_res = retrieval_service.rough_ranking("电脑如何开机",
                                                        MarkDownUtils.collect_md_metadata(settings.CRAWL_OUTPUT_DIR))
    for rough in rough_ranking_res[:10]:
        print(f"粗排---{rough}")

    simi_ranking_res = retrieval_service.fine_ranking("电脑如何开机", rough_ranking_res[:10])

    for simi in simi_ranking_res:
        print(f"精排---{simi}")
