import logging

import jieba

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Dict, Any
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from backend.knowledge.utils.markdown_utils import MarkDownUtils
from backend.knowledge.repositories.vector_store_reposity import VectorStoreRepository
from backend.knowledge.services.ingestion.ingestion_prosessor import IngestionProcessor
from backend.knowledge.config.settings import settings


class RetrievalService:
    """
        负责检索的类（检索器）
    """

    def __init__(self):
        self.chroma_vector = VectorStoreRepository()
        self.ingestion_processor = IngestionProcessor()  # 切分器

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

        # 3. 处理文档（根据标题读取标题对应的文档内容----Document（page_content, metadata={}）
        based_title_candidates = []
        for fine_md_metadata in fine_mds_metadata:
            try:
                # 3.1 打开文件
                with open(fine_md_metadata['path'], "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # 3.2 判断 content 长度
                if len(content) < 3000:
                    # 不切分
                    doc = Document(page_content=content, metadata={
                        "path": fine_md_metadata['path'],
                        "title": fine_md_metadata['title']
                    })
                    based_title_candidates.append(doc)
                else:
                    doc_chunks = self._deal_long_title_content(content, fine_md_metadata, user_question)
                    based_title_candidates.extend(doc_chunks)
            except Exception as e:
                logger.error(f"打开文件失败：{e}")

        # 4. 返回指定文档列表
        return based_title_candidates

    def _deduplicate(self, total_candidates: List[Document]) -> List[Document]:
        """
        对合并后的文档列表去重
        用 set() 集合去重，（（title, 内容前「100」个字符）） ----> key
        Args:
            total_candidates: 合并的文档列表

        Returns:
            List[Document]： 唯一的文档列表
        """
        # 1. 候选集合为空
        if not total_candidates:
            return []

        # 2. 定义set集合
        seen = set()
        unique_candidates = []

        # 3. 遍历合并后的每一个文档列表
        for document in total_candidates:
            key = (document.metadata['title'], document.page_content[:100])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(document)

        # 4. 返回唯一的
        return unique_candidates

    def _reranking(self, unique_candidates: List[Document], user_question: str) -> List[Document]:
        """
        重新计算打分 && 排序
        长文档已经进行了 cosine_similarity() 的计算，无需再次打分
        对第一路的文档和第二路的短文档重新计算
        Args:
            unique_candidates: 唯一的候选文档列表
            user_question: 用户输入的问题
        Returns:
            List[Document]：最终指定的 TOP-N 个文档列表
        """
        # 1. 检查文档是否存在
        if not unique_candidates:
            return []

        # 2. 遍历去重并且合并后的文档列表
        need_embedding_docs = []
        need_embedding_candidates_indices = []
        score_doc = []
        for candidate_index, unique_candidate in enumerate(unique_candidates):
            # 2.1 如何去判断 第二路长文档、第二路短文档 和 第一路文档 ？
            if "chunk_index" in unique_candidate.metadata and "similarity" in unique_candidate.metadata:
                score_doc.append((unique_candidate, unique_candidate.metadata["similarity"]))

            # 2.2 第二路短文档和第一路文档
            else:
                need_embedding_docs.append(unique_candidate)
                need_embedding_candidates_indices.append(candidate_index)

        # 3. 处理需要重新计算分数的文档
        if need_embedding_docs:
            # 3.1 计算用户问题的向量
            question_embedding = self.chroma_vector.embed_query(user_question)
            # 3.2 获取需要向量的文档内容
            embedding_docs_content = ["文档来源:" + doc.metadata['title'] + doc.page_content for doc in
                                      need_embedding_docs]

            # 3.3 计算文档的向量
            doc_embeddings = self.chroma_vector.embed_documents(embedding_docs_content)

            # 3.4 计算相似性得分
            similarity = cosine_similarity([question_embedding, doc_embeddings]).flatten()

            # 3.5 封装到带得分的文档列表
            for idx, candidate_index in enumerate(need_embedding_candidates_indices):
                score_doc.append((need_embedding_docs[candidate_index], similarity[idx]))
        # 4. 排序
        sorted_docs = sorted(unique_candidates, key=lambda x: x[1], reverse=True)

        # 5. 返回 Top-N
        return [doc for doc,_ in sorted_docs]

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

    def _deal_long_title_content(self, content: str, fine_md_metadata: Dict[str, Any], user_question: str) -> List[
        Document]:
        """
        处理标题对应的长文本
        切分 --> 文档块 --> 算文档块和问题的相似度
        Args:
            content: 长文本
            fine_md_metadata: 长文本对应的元数据
            user_question: 用户输入的问题

        Returns:
            List[Document]: 和问题相似的文档快（chunk）
        """
        # 1. 对长文本切分（可以换合适切分器）
        chunks = self.ingestion_processor.document_splitter.split_text(content)

        # 2. 获取对应的标题
        doc_chunks_title = fine_md_metadata['title']

        # 3. 标题注入到文档块中
        doc_chunks_inject_title = [f"文档来源:{doc_chunks_title}" + doc_chunk for doc_chunk in chunks]

        # 4. 对问题向量化
        question_embedding = self.chroma_vector.embed_query(user_question)

        # 5. 对切分后的文档块向量化
        doc_chunk_embeddings = self.chroma_vector.embed_documents(doc_chunks_inject_title)

        # 6. 计算相似性 doc_chunks_similarity [0.8, 0.6, 0.7, 0.1, 0.9]
        doc_chunks_similarity = cosine_similarity([question_embedding], doc_chunk_embeddings).flatten()

        # 7. 获取3个相似性分数最高的三个索引 argsort -> [3,1,2,0,4] -> [4,0,2]
        top_doc_chunks_indices = doc_chunks_similarity.argsort()[-3:][::-1]

        # 8. 构建最终的文档对象列表
        docs = []
        for i, chunk_idx in enumerate(top_doc_chunks_indices):
            doc = Document(
                page_id=chunks[chunk_idx],
                metadata={
                    "path": fine_md_metadata['path'],
                    "title": fine_md_metadata['title'],
                    "chunk_index": int(chunk_idx),
                    "similarity": float(doc_chunks_similarity[chunk_idx])
                }
            )
            docs.append(doc)

        return docs


if __name__ == '__main__':
    retrieval_service = RetrievalService()

    rough_ranking_res = retrieval_service.rough_ranking("电脑如何开机",
                                                        MarkDownUtils.collect_md_metadata(settings.CRAWL_OUTPUT_DIR))
    for rough in rough_ranking_res[:10]:
        print(f"粗排---{rough}")

    simi_ranking_res = retrieval_service.fine_ranking("电脑如何开机", rough_ranking_res[:10])

    for simi in simi_ranking_res:
        print(f"精排---{simi}")
