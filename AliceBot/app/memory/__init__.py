"""记忆模块API接口"""

# 从各模块导出关键类和函数
from app.memory.combined_memory import CombinedMemoryManager
from app.memory.local_history import LocalHistoryManager
from app.memory.relation_db import GlobalRelationDB, UserProfile, Relationship
from app.memory.vector_store import VectorMemory
from app.memory.smart_retrieval import MemoryRetrievalTool, get_smart_memory_retriever, initialize_smart_memory_retrieval

# 创建全局实例
from app.memory.combined_memory import combined_memory
from app.memory.relation_db import relation_db
from app.memory.vector_store import vector_db
from app.memory.smart_retrieval import smart_memory_retriever

# 导出所有内容
__all__ = [
    # 类
    "CombinedMemoryManager",
    "LocalHistoryManager",
    "GlobalRelationDB",
    "UserProfile",
    "Relationship",
    "VectorMemory",
    "MemoryRetrievalTool",
    
    # 全局实例
    "combined_memory",
    "relation_db",
    "vector_db",
    "smart_memory_retriever",
    
    # 函数
    "get_smart_memory_retriever",
    "initialize_smart_memory_retrieval",
    
    # 常用别名
    "MemoryManager",   # CombinedMemoryManager的别名
    "memory_manager",  # combined_memory的别名
    "RelationDB",      # GlobalRelationDB的别名
    "VectorDB",        # VectorMemory的别名
    "SmartRetriever",  # MemoryRetrievalTool的别名
    "SmartMemoryRetriever",  # MemoryRetrievalTool的别名
]

# 添加常用别名
MemoryManager = CombinedMemoryManager
RelationDB = GlobalRelationDB
VectorDB = VectorMemory
SmartRetriever = MemoryRetrievalTool
SmartMemoryRetriever = MemoryRetrievalTool
memory_manager = combined_memory  # 为combined_memory实例添加别名

# 版本信息
__version__ = "1.0.0"

# 使用说明
"""
使用示例：

# 导入统一API
from app.memory import memory_manager, relation_db, vector_db

# 更新记忆
await memory_manager.update_memory("user_id", "聊天内容", "角色")

# 获取相关记忆
relevant_memory = await memory_manager.get_relevant_memory("查询", "聊天历史", "用户ID")

# 智能检索记忆
retrieval_result = await memory_manager.smart_retrieve("查询", "聊天历史", "发送者", "用户ID")

# 更新用户关系
relation_db.update_relationship("user_id", "target_id", relationship_data)

# 获取用户资料
user_profile = await relation_db.get_user_profile("user_id")

# 搜索向量存储
results = await vector_db.search("查询", k=3)
"""