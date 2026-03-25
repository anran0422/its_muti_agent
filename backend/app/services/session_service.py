from typing import List, Dict, Any

from repositories.session_repository import session_repository
from infrastructure.logging.logger import logger


class SessionService:
    """
    会话业务管理服务类
    主要负责对用户历史会话的管理，包含
    1. 准备加载历史会话
    2. 读取历史会话
    3. 存储历史会话
    4. 查询会话列表
    """

    DEFAULT_SESSION_ID = "default_session_id"

    def __init__(self):
        """
        初始化会话操作的工具
        """
        self._repo = session_repository

    def prepare_history(self, user_id: str, session_id: str, user_input: str, max_turn: int = 3) -> list[dict[str, Any]]:
        """
        准备历史会话：加载历史会话 --- 裁剪历史会话（保留指定轮次） --- 返回历史会话
        调用时机：发送请求给 LLM 之前
        Args:
            user_id: 用户id
            session_id: 会话id
            max_turn: 保留的最大轮数

        Returns:
            会话列表
        """
        # 1. 加载历史会话
        chat_history = self.load_history(user_id, session_id)

        # 2. 拼接用户角色的消息
        chat_history.append({"role": "user", "content": user_input})

        # 3. 裁剪历史会话
        truncate_history = self._truncate_history(chat_history, max_turn)

        # 4. 返回历史会话
        return truncate_history

    def load_history(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """
        主要负责：加载历史会话（从文件中读取）
        Args:
            user_id: 用户id
            session_id: 会话id

        Returns:
            List[Dict[str, Any]]
        """

        # 1. 判断 session_id 是否存在
        target_session_id = session_id if session_id else self.DEFAULT_SESSION_ID

        # 2. 加载
        try:
            session_history = self._repo.load_session(user_id, target_session_id)

            # 历史不存在，初始化一个会话结构
            if session_history is None:
                # 构建一个新的结构
                return self._init_system_msg_instruct(session_id)
            return session_history
        except Exception as e:
            logger.error(f"加载会话历史 {session_id} 失败 {str(e)}")

    def _init_system_msg_instruct(self, session_id: str) -> List[Dict[str, Any]]:
        """
        初始化一个带系统级别的消息结构
        Args:
            session_id: 会话id

        Returns:
            List[Dict[str, Any]]
        """
        return [{
            "role": "system",
            "content": f"你是一个有记忆的智能体助手，请基于上下文历史会话回答用户问题（会话ID{session_id}）"
        }]

    def _truncate_history(self, chat_history: List[Dict[str, Any]], max_turn: int = 3) -> List[Dict[str, Any]]:
        """
        裁剪指定轮次的消息
        Args:
            chat_history: 加载到的历史会话消息
            max_turn: 指定最大轮数的历史消息

        Returns:
            List[Dict[str, Any]]
        """
        # 1. 获取系统角色的消息[无论如何都要留，通常来说就一条]
        system_msg = [msg for msg in chat_history if msg.get("role") == "system"]

        # 2. 获取非系统角色的消息
        no_system_msg = [msg for msg in chat_history if msg.get("role") != "system"]

        msg_limit = max_turn * 2

        # 3. 裁剪非系统角色的消息列表
        truncate_msg = no_system_msg[-msg_limit:]

        # 4. 拼接上系统角色的消息
        final_msg = system_msg + truncate_msg

        # 5. 返回指定轮数的消息
        return final_msg

    def save_history(self, user_id: str, session_id: str, chat_history: List[Dict[str, Any]]):
        """
        保存历史会话
        调用时机：调用完 LLM（Agent）之后
        Args:
            user_id: 用户id
            session_id: 会话id
            chat_history: 要保存的历史消息【角色：system、user、assistant】

        Returns:

        """
        # 1. 历史会话是否存在
        if chat_history is None:
            return

        # 2. 保存
        try:
            self._repo.save_session(user_id, session_id, chat_history)
        except Exception as e:
            logger.error(f"保存角色 {user_id} 会话 {session_id} 文件失败：{str(e)}")
            return


# 全局单例
session_service = SessionService()

if __name__ == '__main__':
    result = session_service.prepare_history("lwt", "666", "请问北京怎么走？")
    result.append({"role": "user", "content": "你好！"})
    result.append({"role": "assistant", "content": "你好！请问有什么可以帮助您的？"})  # Agent的输出（final_output）

    session_service.save_history("lwt", "666", result)
