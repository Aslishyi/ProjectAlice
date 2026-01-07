# === 数据库配置文件 ===

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from typing import Generator

# 数据库路径
DB_PATH = "data/project_alice.db"

# 确保数据目录存在
os.makedirs("data", exist_ok=True)

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
from sqlalchemy import Column, String, Text, DateTime, func

# 会话历史模型
class SessionHistoryModel(Base):
    __tablename__ = "session_history"
    
    session_id = Column(String(100), primary_key=True, index=True)
    summary = Column(Text, nullable=False, default="")
    messages = Column(Text, nullable=False, default="[]")  # JSON格式存储
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# 初始化数据库
def init_db():
    """创建所有数据库表"""
    from app.memory.relation_db import UserProfileModel  # 避免循环导入
    Base.metadata.create_all(bind=engine)
