import os
import json
import aiofiles
from typing import List, Tuple, Dict, Any
from langchain_core.messages import BaseMessage, messages_to_dict, messages_from_dict, HumanMessage, AIMessage

# 定义存储路径
HISTORY_DIR = "./data/history"

# 确保目录存在
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)


class LocalHistoryManager:
    """
    负责本地文件系统的会话历史存储。
    根据 session_id 进行物理文件隔离。
    """

    @staticmethod
    def _get_file_path(session_id: str) -> str:
        # 为了安全，移除 session_id 中可能包含的非法字符
        safe_id = "".join([c for c in session_id if c.isalnum() or c in ('_', '-')])
        return os.path.join(HISTORY_DIR, f"{safe_id}.json")

    @classmethod
    async def save_state(cls, messages: List[BaseMessage], summary: str, session_id: str):
        """
        异步保存当前对话状态到文件。
        :param messages: LangChain 消息列表
        :param summary: 当前的对话总结
        :param session_id: 会话唯一标识 (private_xxx 或 group_xxx)
        """
        if not session_id:
            print("⚠️ [History] Cannot save: session_id is missing.")
            return

        file_path = cls._get_file_path(session_id)

        # 将消息对象序列化为字典
        serialized_msgs = messages_to_dict(messages)

        data = {
            "summary": summary,
            "messages": serialized_msgs,
            "updated_at": str(os.path.getmtime(file_path)) if os.path.exists(file_path) else "new"
        }

        try:
            async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"❌ [History] Save failed for {session_id}: {e}")

    @classmethod
    async def load_state(cls, session_id: str) -> Tuple[List[BaseMessage], str]:
        """
        异步读取会话状态。
        :param session_id: 会话唯一标识
        :return: (messages, summary)
        """
        if not session_id:
            return [], ""

        file_path = cls._get_file_path(session_id)

        if not os.path.exists(file_path):
            return [], ""

        try:
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
                if not content:
                    return [], ""

                data = json.loads(content)
                summary = data.get("summary", "")
                msgs_dict = data.get("messages", [])

                # 反序列化回 LangChain 消息对象
                messages = messages_from_dict(msgs_dict)

                return messages, summary

        except Exception as e:
            print(f"❌ [History] Load failed for {session_id}: {e}")
            return [], ""

    @classmethod
    def get_existing_summary_sync(cls, session_id: str) -> str:
        """
        同步辅助方法：仅获取 Summary (用于初始化时快速读取)
        """
        if not session_id: return ""
        file_path = cls._get_file_path(session_id)
        if not os.path.exists(file_path): return ""

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("summary", "")
        except:
            return ""

