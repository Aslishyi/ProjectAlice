import os
import json
import aiofiles
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any
from langchain_core.messages import BaseMessage, messages_to_dict, messages_from_dict, HumanMessage, AIMessage
from pathlib import Path

# 配置日志
logger = logging.getLogger("LocalHistory")

# 定义存储路径（用于JSON文件存储）
import os
# 获取当前文件所在目录的父目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HISTORY_DIR = os.path.join(BASE_DIR, "data", "history")


class LocalHistoryManager:
    """
    负责会话历史存储，使用JSON文件。
    根据 session_id 进行数据隔离。
    """



    @classmethod
    async def save_state(cls, messages: List[BaseMessage], summary: str, session_id: str):
        """
        异步保存当前对话状态到JSON文件。
        :param messages: LangChain 消息列表
        :param summary: 当前的对话总结
        :param session_id: 会话唯一标识 (private_xxx 或 group_xxx)
        """
        if not session_id:
            logger.warning("⚠️ [History] Cannot save: session_id is missing.")
            return

        # 确保存储目录存在
        os.makedirs(HISTORY_DIR, exist_ok=True)
        
        # 安全处理文件名
        safe_id = "".join([c for c in session_id if c.isalnum() or c in ('_', '-')])
        file_path = os.path.join(HISTORY_DIR, f"{safe_id}.json")
        
        # 准备要保存的数据
        data = {
            "session_id": session_id,
            "summary": summary,
            "messages": messages_to_dict(messages),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
            # print(f"✅ [History] Saved to {file_path}")
        except Exception as e:
            logger.error(f"❌ [History] Save failed for {session_id}: {e}")

    @classmethod
    async def load_state(cls, session_id: str) -> Tuple[List[BaseMessage], str]:
        """
        异步读取会话状态。
        :param session_id: 会话唯一标识
        :return: (messages, summary)
        """
        if not session_id:
            return [], ""

        try:
            # 安全处理文件名
            safe_id = "".join([c for c in session_id if c.isalnum() or c in ('_', '-')])
            file_path = os.path.join(HISTORY_DIR, f"{safe_id}.json")
            
            if not os.path.exists(file_path):
                return [], ""
            
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
                if not content:
                    return [], ""
                
                data = json.loads(content)
                summary = data.get("summary", "")
                msgs_dict = data.get("messages", [])
                
                # 反序列化消息
                messages = messages_from_dict(msgs_dict)
                return messages, summary

        except Exception as e:
            logger.error(f"❌ [History] Load failed for {session_id}: {e}")
            return [], ""

    @classmethod
    def get_existing_summary_sync(cls, session_id: str) -> str:
        """
        同步辅助方法：仅获取 Summary (用于初始化时快速读取)
        """
        if not session_id: return ""

        try:
            # 安全处理文件名
            safe_id = "".join([c for c in session_id if c.isalnum() or c in ('_', '-')])
            file_path = os.path.join(HISTORY_DIR, f"{safe_id}.json")
            
            if not os.path.exists(file_path):
                return ""
            
            with open(file_path, mode='r', encoding='utf-8') as f:
                content = f.read()
                if not content:
                    return ""
                
                data = json.loads(content)
                return data.get("summary", "")
        except Exception as e:
            logger.error(f"❌ [History] Get summary failed for {session_id}: {e}")
            return ""
    
    @classmethod
    async def _migrate_from_json(cls, session_id: str):
        """
        检查JSON文件是否存在（向后兼容旧代码调用）
        """
        if not os.path.exists(HISTORY_DIR):
            return
        
        # 获取文件路径
        safe_id = "".join([c for c in session_id if c.isalnum() or c in ('_', '-')])
        file_path = os.path.join(HISTORY_DIR, f"{safe_id}.json")
        
        if not os.path.exists(file_path):
            return
        
        logger.info(f"✅ [History] JSON file found for {session_id}")

