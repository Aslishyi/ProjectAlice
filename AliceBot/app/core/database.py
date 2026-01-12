# === 数据库配置文件 ===

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from typing import Generator

# 数据库路径
# 获取当前文件所在目录的父目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "project_alice.db")

# 确保数据目录存在
data_dir = os.path.join(BASE_DIR, "data")
os.makedirs(data_dir, exist_ok=True)

# 创建数据库引擎
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False}  # SQLite 多线程支持
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()


# 数据库依赖
def get_db() -> Generator[Session, None, None]:
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 导入数据库模型
from sqlalchemy import Column, String, Text, DateTime, func, JSON, Integer

# 会话历史模型
class SessionHistoryModel(Base):
    __tablename__ = "session_history"
    
    session_id = Column(String(100), primary_key=True, index=True)
    summary = Column(Text, nullable=False, default="")
    messages = Column(Text, nullable=False, default="[]")  # JSON格式存储
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# 转发消息存储模型
class ForwardMessageModel(Base):
    __tablename__ = "forward_messages"
    
    forward_id = Column(String(100), primary_key=True, index=True)  # 转发消息ID
    full_content = Column(JSON, nullable=False)  # 完整的转发消息内容（JSON格式）
    summary = Column(Text, nullable=False, default="")  # 转发消息摘要
    message_count = Column(Integer, nullable=False, default=0)  # 消息数量
    image_count = Column(Integer, nullable=False, default=0)  # 图片数量
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # 创建时间
    accessed_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # 最后访问时间


# 初始化数据库
def init_db():
    """创建所有数据库表"""
    from app.memory.relation_db import UserProfileModel  # 避免循环导入
    Base.metadata.create_all(bind=engine)
