import asyncio
import httpx
from typing import Dict
from agents import function_tool
from infrastructure.logging.logger import logger
from config.settings import settings


@function_tool
async def query_knowledge(question: str) -> Dict:
    """
       查询电脑问题知识库服务,用于检索与用户问题相关的技术文档或解决方案。

       Args:
           question (Optional[str]): 需要查询的具体问题文本。

       Returns:
           dict: 包含查询结果的字典。包含 'question':用户输出问题 'answer':答案
    """
    # 调试：打印实际请求的 URL
    request_url = f"{settings.KNOWLEDGE_BASE_URL}/query"
    logger.info(f"[知识库查询] ========== 开始 ==========")
    logger.info(f"[知识库查询] URL: {request_url}")
    logger.info(f"[知识库查询] 问题: {question}")
    logger.info(f"[知识库查询] 正在发送 POST 请求...")
    
    async with httpx.AsyncClient() as client:
        try:
            # 1. 发送请求（异步上下文管理器对象）
            response = await client.post(
                url=request_url,
                json={"question": question},
                timeout=120
            )
            logger.info(f"[知识库查询] 收到响应! 状态码: {response.status_code}")

            # 2. 处理异常情况（4xx-600x）直接抛出异常
            response.raise_for_status()

            # 3. 处理正常情况
            result = response.json()
            logger.info(f"[知识库查询] 请求成功完成!")
            logger.info(f"[知识库查询] ========== 结束 ==========")
            return result

        except httpx.HTTPError as e:
            logger.error(f"[知识库查询] HTTP错误: type={type(e).__name__}, msg={str(e)}, url={request_url}")
            return {"status": "error", "err_msg": f"发送请求获取知识库服务下的知识库数据失败：{str(e)}"}

        except Exception as e:
            logger.error(f"[知识库查询] 未知错误: {str(e)}")
            return {"status": "error", "err_msg": f"未知错误：{str(e)}"}


async def main():
    result = await query_knowledge(question="电脑开机不了怎么解决？")
    print(result)

if __name__ == '__main__':
    asyncio.run(main())
