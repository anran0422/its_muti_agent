from pydantic import BaseModel


class UploadResponse(BaseModel):
    """
        文件上传的响应数据类型
    """
    status: str # 响应状态
    message: str # 响应消息内容
    file_name: str # 上传的文档名称
    chunks_added: int # 上传文档切分后的块数