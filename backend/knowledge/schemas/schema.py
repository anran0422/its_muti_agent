from pydantic import BaseModel


class UploadResponse(BaseModel):
    """
        文件上传的响应数据类型
    """
    status: str  # 响应状态
    message: str  # 响应消息内容
    file_name: str  # 上传的文档名称
    chunks_added: int  # 上传文档切分后的块数


class QueryResponse(BaseModel):
    """
        查询响应的数据模型
    """
    question: str  # 用户问题
    answer: str  # 模型的回答


class QueryRequest(BaseModel):
    """
        查询请求的数据模型
    """
    question: str  # 用户问题
