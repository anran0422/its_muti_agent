import os.path

from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool
import aiofiles
from backend.knowledge.schemas.schema import UploadResponse
from backend.knowledge.services.ingestion.ingestion_prosessor import IngestionProcessor
from backend.knowledge.config.settings import settings
import logging
import shutil

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 1.创建APIRouter
router = APIRouter()

# 2.创建上传知识库切分实例
ingestion_processor = IngestionProcessor()


# IO 文件读写 执行SQL 网络请求，典型的耗时任务
@router.post("/upload", response_model=UploadResponse, summary="处理知识库上传")
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    # xxx.md
    try:
        # 0. 临时目录
        temp_md_dir = settings.TMP_MD_FOLDER_PATH
        # 使用原始文件名作为临时文件名
        tmp_md_path = os.path.join(temp_md_dir, file.filename)
        if not os.path.exists(temp_md_dir):
            os.makedirs(temp_md_dir, exist_ok=True)

        # 1. 处理临时文件
        file_suffix = os.path.splitext(file.filename)[1]
        async with aiofiles.tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as temp_file:
            # a. 读取上传文件的内容（异步协程） 对象（异步协程）缓冲区（1M）
            while file_content := await file.read(1024 * 1024):
                # b. 将文件内容写入到临时文件
                await temp_file.write(file_content)

            # c. 获取临时文件路径
            temp_file_path = temp_file.name

        # 移动文件展示临时目录
        shutil.move(temp_file_path, tmp_md_path)

        # 2. 磁盘写入完成，开始入库 # TODO 去重
        # chunks_added = ingestion_processor.ingest_file(temp_file_path)
        chunks_added = await run_in_threadpool(ingestion_processor.ingest_file, tmp_md_path)

        # 3. 构建文件上传响应对象
        return UploadResponse(
            status="success",
            message="文档上传知识库成功",
            file_name=file.filename,
            chunks_added=chunks_added
        )

    except Exception as e:
        logger.error(f"文件上传知识库失败：{e}")
        raise e

    finally:
        # 删除临时文件（如果还存在的话，移动后文件已不存在此处）
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"临时文件:{temp_file_path}已删除...")
