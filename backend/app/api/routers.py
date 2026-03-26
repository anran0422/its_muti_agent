from fastapi.routing import APIRouter
from schemas.request import ChatMessageRequest
from starlette.responses import StreamingResponse
from services.agent_service import MultiAgentService

from infrastructure.logging.logger import logger

# 1. 定义请求路由器
router = APIRouter()


# 2. 定义对话请求
@router.post("/api/query", summary="智能体对话接口")
async def query(request_context: ChatMessageRequest) -> StreamingResponse:
    """
    SSE 返回数据（流式响应）
    响应头中：text/event-stream
    Args:
        request_context: 请求上下文

    Returns:
        StreamingResponse
    """
    # 1. 获取请求上下文的属性
    user_id = request_context.context.user_id
    user_query = request_context.query
    logger.info(f"用户: {user_id} 发送的待处理任务: {user_query}")

    # 2. 调用 AgentService（智能体的业务服务类）
    async_generator_result = MultiAgentService.process_task(request_context, flag=True)

    # 3. 封装结果到 StreamResponse 中
    return StreamingResponse(
        content=async_generator_result,
        status_code=200,
        media_type="text/event-stream"
    )
