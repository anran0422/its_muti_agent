from backend.knowledge.repositories.vector_store_reposity import VectorStoreRepository
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionProcessor:
    """
    文档摄入类：摄入：加载、切分存储
    """

    def __init__(self):
        self.vector_store = VectorStoreRepository()
        self.document_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # 长文档分块的阈值
            chunk_overlap=200,
            separators=[  # 需要自己定义特殊的切分方式
                "\n##",  # 二级标题切分
                "\n**",  # 加粗切分
                "\n\n",  # 这几个都是通用
                "\n",
                " ",
                ""
            ]
        )

    def ingest_file(self, md_path: str) -> int:
        """
        文档完整操作
        包含阶段：文档的加载 -> 文档切割 -> 文档存储
        Args:
            md_path: 文件存储路径

        Returns:
            int：保存成功的文档数
        """

        # 1. 根据文件路径加载得到文档列表
        # a.定义文档加载器（1. 非结构化的文档加载器 MarkDownLoader 2. 文本加载器 TextLoader
        try:
            # 尝试使用 UTF-8 编码加载文件
            text_loader = TextLoader(file_path=md_path, encoding='utf-8')
            # b. 加载文件返回文档列表（TextLoader 返回的文档列表中有且只有一个文档对象）
            documents = text_loader.load()
        except Exception as e:
            logger.error(f"文件：{md_path}没有加载到，原因：{str(e)}")
            raise Exception(f"文件：{md_path}没有加载到，原因：{str(e)}")

        # 2. 切分文档得到文档块列表
        # 2.1 动态机制切分
        # a. 如果文档内容不大，直接将这内容作为一个 chunk（不用切分）
        # b. 如果内容比较大，分析大内容的数据结构，然后定制切分策略。采用Header rejection：标题注入（保留每一块的业务背景、上下文）
        final_documents_chunks = []
        for doc in documents:
            if len(doc.page_content) < 3000:
                final_documents_chunks.append(doc)
            else:
                documents_chunks_list = self.document_splitter.split_documents(documents)
                # 插入每个文档 page_content 拼接上 md 标题
                for document_chunk in documents_chunks_list:
                    # 获取每一个文档块的标题
                    md_path = document_chunk.metadata["source"]
                    title = os.path.basename(md_path)

                    # 拼接到每一个文档块的 page_content 上
                    document_chunk.page_content = f"上下文来源：{title}\n{document_chunk.page_content}"

                final_documents_chunks.extend(documents_chunks_list)

        # 3. 切分后文档快的元数据校验（过滤不被向量数据库支持的元数据清除掉）
        clean_documents_chunks = filter_complex_metadata(final_documents_chunks)

        # 4. 无效性检查（校验page_content是否合法 不能为空）
        valid_documents_chunks = [document for document in clean_documents_chunks if document.page_content.strip()]
        if not valid_documents_chunks:
            logger.error("切分后的文档块中没有任何的内容")
            return 0

        # 5. 存储文档块到向量数据库
        total_documents_chunks = self.vector_store.add_documents(valid_documents_chunks)

        # 6. 返回保存成功的文档数
        return total_documents_chunks
