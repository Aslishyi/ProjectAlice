import os
import json
import aiofiles
from typing import List, Tuple, Dict, Any
from langchain_core.messages import BaseMessage, messages_to_dict, messages_from_dict, HumanMessage, AIMessage
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, SessionHistoryModel

# 定义存储路径（用于迁移旧数据）
HISTORY_DIR = "./data/history"


class LocalHistoryManager:
    """
    负责会话历史存储，使用数据库。
    根据 session_id 进行数据隔离。
    """

    @staticmethod
    def _get_db() -> Session:
        """获取数据库会话"""
        return SessionLocal()

    @classmethod
    async def save_state(cls, messages: List[BaseMessage], summary: str, session_id: str):
        """
        异步保存当前对话状态到数据库。
        :param messages: LangChain 消息列表
        :param summary: 当前的对话总结
        :param session_id: 会话唯一标识 (private_xxx 或 group_xxx)
        """
        if not session_id:
            print("⚠️ [History] Cannot save: session_id is missing.")
            return

        # 将消息对象序列化为JSON字符串
        serialized_msgs = json.dumps(messages_to_dict(messages), ensure_ascii=False)

        try:
            db = cls._get_db()
            
            # 查找现有记录
            history = db.query(SessionHistoryModel).filter_by(session_id=session_id).first()
            
            if history:
                # 更新现有记录
                history.summary = summary
                history.messages = serialized_msgs
            else:
                # 创建新记录
                history = SessionHistoryModel(
                    session_id=session_id,
                    summary=summary,
                    messages=serialized_msgs
                )
                db.add(history)
            
            db.commit()
            db.close()
        except Exception as e:
            print(f"❌ [History] Save failed for {session_id}: {e}")
            try:
                db.rollback()
            except:
                pass
            finally:
                db.close()

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
            db = cls._get_db()
            
            # 查找记录
            history = db.query(SessionHistoryModel).filter_by(session_id=session_id).first()
            
            if not history:
                # 如果数据库中没有，尝试从旧的JSON文件中迁移
                await cls._migrate_from_json(session_id)
                # 再次查询
                history = db.query(SessionHistoryModel).filter_by(session_id=session_id).first()
                
            if not history:
                return [], ""
            
            # 反序列化消息
            msgs_dict = json.loads(history.messages)
            messages = messages_from_dict(msgs_dict)
            
            db.close()
            return messages, history.summary

        except Exception as e:
            print(f"❌ [History] Load failed for {session_id}: {e}")
            try:
                db.close()
            except:
                pass
            return [], ""

    @classmethod
    def get_existing_summary_sync(cls, session_id: str) -> str:
        """
        同步辅助方法：仅获取 Summary (用于初始化时快速读取)
        """
        if not session_id: return ""

        try:
            db = cls._get_db()
            
            # 查找记录
            history = db.query(SessionHistoryModel).filter_by(session_id=session_id).first()
            
            if not history:
                # 如果数据库中没有，尝试从旧的JSON文件中迁移
                import asyncio
                asyncio.run(cls._migrate_from_json(session_id))
                # 再次查询
                history = db.query(SessionHistoryModel).filter_by(session_id=session_id).first()
                
            db.close()
            return history.summary if history else ""
        except Exception as e:
            print(f"❌ [History] Get summary failed for {session_id}: {e}")
            try:
                db.close()
            except:
                pass
            return ""
    
    @classmethod
    async def _migrate_from_json(cls, session_id: str):
        """
        从旧的JSON文件迁移数据到数据库
        """
        if not os.path.exists(HISTORY_DIR):
            return
        
        # 获取旧文件路径
        safe_id = "".join([c for c in session_id if c.isalnum() or c in ('_', '-')])
        file_path = os.path.join(HISTORY_DIR, f"{safe_id}.json")
        
        if not os.path.exists(file_path):
            return
        
        try:
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                content = await f.read()
                if not content:
                    return
                
                data = json.loads(content)
                summary = data.get("summary", "")
                msgs_dict = data.get("messages", [])
                
                # 保存到数据库
                db = cls._get_db()
                
                # 检查是否已经存在
                existing = db.query(SessionHistoryModel).filter_by(session_id=session_id).first()
                if not existing:
                    history = SessionHistoryModel(
                        session_id=session_id,
                        summary=summary,
                        messages=json.dumps(msgs_dict, ensure_ascii=False)
                    )
                    db.add(history)
                    db.commit()
                    print(f"✅ [History] Migrated {session_id} from JSON to database")
                
                db.close()
        except Exception as e:
            print(f"❌ [History] Migration failed for {session_id}: {e}")
            try:
                db = cls._get_db()
                db.rollback()
                db.close()
            except:
                pass

