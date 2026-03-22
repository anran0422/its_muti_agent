"""
    创建 FastAPI 实例 并且管理所有的路由
"""
import uvicorn
from fastapi import FastAPI
from api.routers import router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    # 1. 创建 FastAPI 实例
    app = FastAPI(title="Knowledge API", description="知识库接口")

    # 2. 注册各种路由
    app.include_router(router=router)

    return app


if __name__ == '__main__':
    print("1.启动 web 服务器")
    try:
        uvicorn.run(app=create_app(), host="127.0.0.1", port=8001)
        logger.info("2.启动 web 服务器成功")
    except KeyboardInterrupt as e:
        logger.info(f"2.启动 web 服务器失败：{str(e)}")
