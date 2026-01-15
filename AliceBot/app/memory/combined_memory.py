import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_community.memory.kg import ConversationKGMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import config
from app.memory.vector_store import vector_db
from app.memory.relation_db import relation_db
from app.memory.smart_retrieval import get_smart_memory_retriever

# 配置日志
logger = logging.getLogger("CombinedMemory")

class CombinedMemoryManager:
    """
    组合内存管理器，集成了三种LangChain内存方法：
    1. ConversationEntityMemory - 实体记忆
    2. ConversationKGMemory - 知识图谱记忆
    3. VectorStoreRetrieverMemory - 向量存储检索记忆
    """
    
    def __init__(self):
        # 初始化LLM - ChatOpenAI本身支持异步操作
        self.llm = ChatOpenAI(
            model=config.SMALL_MODEL,
            temperature=0.0,
            api_key=config.SMALL_MODEL_API_KEY,
            base_url=config.SMALL_MODEL_URL
        )
        
        # 1. 自定义实体记忆实现（替代ConversationEntityMemory）
        self.entity_store = {}
        self.entity_k = 5
        
        # 2. 初始化知识图谱记忆
        self.kg_memory = ConversationKGMemory(
            llm=self.llm,
            return_messages=True,
            k=10  # 保留最近10个关系
        )
    
    async def update_memory(self, user_input: str, ai_response: str, user_id: str, user_name: str):
        """
        更新所有记忆类型
        """
        # 更新实体记忆（简化实现）
        user_key = f"User {user_name}: {user_input}"
        self.entity_store[user_key] = ai_response
        
        # 保持实体记忆的大小限制
        if len(self.entity_store) > self.entity_k:
            # 删除最旧的条目
            first_key = next(iter(self.entity_store))
            del self.entity_store[first_key]
        
        # 更新知识图谱记忆（将同步方法包装在异步线程中执行）
        import asyncio
        await asyncio.to_thread(self.kg_memory.save_context, {
            "input": f"User {user_name}: {user_input}"
        }, {
            "output": ai_response
        })
        
        # 更新向量存储记忆（直接使用vector_db）
        memory_content = f"User {user_name}: {user_input}\nAI: {ai_response}"
        await vector_db.add_texts([memory_content], metadatas=[{"user_id": user_id}])
    
    async def get_relevant_memory(self, input_text: str, user_id: str) -> Dict[str, Any]:
        """
        获取所有相关记忆
        """
        relevant_memory = {
            "entities": {},
            "knowledge_graph": [],
            "vector_retrieved": []
        }
        
        # 1. 获取实体记忆
        try:
            relevant_memory["entities"] = self.entity_store
        except Exception as e:
            logger.error(f"❌ [Memory] Failed to get entities: {e}")
        
        # 2. 获取知识图谱记忆
        try:
            import asyncio
            kg = await asyncio.to_thread(self.kg_memory.kg.get_triples)
            relevant_memory["knowledge_graph"] = kg
        except Exception as e:
            logger.error(f"❌ [Memory] Failed to get knowledge graph: {e}")
        
        # 3. 获取向量检索记忆
        try:
            vector_results = await vector_db.search(input_text, k=5)
            relevant_memory["vector_retrieved"] = vector_results
        except Exception as e:
            logger.error(f"❌ [Memory] Failed to get vector memory: {e}")
        
        return relevant_memory
    
    async def get_relationship_insights(self, user_name: str, target_name: str) -> List[Dict[str, Any]]:
        """
        获取两个人之间的关系洞察
        """
        insights = []
        
        # 从知识图谱中提取关系
        try:
            import asyncio
            kg = await asyncio.to_thread(self.kg_memory.kg.get_triples)
            for triple in kg:
                if (user_name.lower() in triple[0].lower() or user_name.lower() in triple[2].lower()) and \
                   (target_name.lower() in triple[0].lower() or target_name.lower() in triple[2].lower()):
                    insights.append({
                        "subject": triple[0],
                        "predicate": triple[1],
                        "object": triple[2]
                    })
        except Exception as e:
            logger.error(f"❌ [Memory] Failed to get relationship insights: {e}")
        
        return insights
    
    async def clear_session(self):
        """
        清除会话记忆（保留长期记忆）
        """
        # 清除实体记忆
        self.entity_store = {}
        
        # 知识图谱记忆可以通过重置kg实现
        import asyncio
        await asyncio.to_thread(lambda: setattr(self.kg_memory, 'kg', type(self.kg_memory.kg)()))
    
    async def smart_retrieve(self, query: str, chat_history: str, sender: str, user_id: str) -> Dict[str, Any]:
        """
        智能记忆检索，根据查询和聊天历史自动生成检索问题并检索相关记忆
        
        Args:
            query: 当前查询文本
            chat_history: 聊天历史记录
            sender: 发送者名称
            user_id: 发送者用户ID
            
        Returns:
            包含检索结果的字典
        """
        try:
            # 获取智能记忆检索工具
            retriever = get_smart_memory_retriever()
            if not retriever:
                logger.error("智能记忆检索工具不可用")
                return {
                    "has_relevant_memory": False,
                    "memory_content": "",
                    "questions": []
                }
            
            # 执行智能记忆检索
            result = await retriever.smart_retrieve_for_query(query, chat_history, sender, user_id)
            return result
            
        except Exception as e:
            logger.error(f"智能记忆检索失败: {e}")
            return {
                "has_relevant_memory": False,
                "memory_content": "",
                "questions": []
            }
    
    async def forget_by_semantic(self, query: str, threshold: float = 0.4, user_id: Optional[str] = None) -> int:
        """
        通过语义删除相似记忆
        
        Args:
            query: 用于匹配要删除记忆的查询文本
            threshold: 相似度阈值，小于该值的记忆将被删除
            user_id: 可选的用户ID，用于过滤特定用户的记忆
            
        Returns:
            删除的记忆数量
        """
        try:
            # 在向量存储中删除相似记忆
            deleted_count = await vector_db.delete_by_semantic(query, threshold)
            
            # 从知识图谱中删除相关三元组
            try:
                import asyncio
                kg = await asyncio.to_thread(self.kg_memory.kg.get_triples)
                for triple in kg:
                    if query.lower() in triple[0].lower() or query.lower() in triple[2].lower():
                        await asyncio.to_thread(self.kg_memory.kg.remove_triple, triple)
            except Exception as e:
                logger.error(f"❌ [Memory] Failed to remove knowledge graph triples: {e}")
            
            # 从实体记忆中删除相关实体
            try:
                entities_to_remove = []
                for entity_name, entity_data in self.entity_store.items():
                    if query.lower() in entity_name.lower() or query.lower() in str(entity_data).lower():
                        entities_to_remove.append(entity_name)
                
                for entity_name in entities_to_remove:
                    if entity_name in self.entity_store:
                        del self.entity_store[entity_name]
            except Exception as e:
                logger.error(f"❌ [Memory] Failed to remove entities: {e}")
            
            return deleted_count
        except Exception as e:
            logger.error(f"❌ [Memory] Failed to forget by semantic: {e}")
            return 0
    
    async def forget_by_keyword(self, keyword: str, user_id: Optional[str] = None) -> int:
        """
        通过关键词删除记忆
        
        Args:
            keyword: 要删除的关键词
            user_id: 可选的用户ID，用于过滤特定用户的记忆
            
        Returns:
            删除的记忆数量
        """
        try:
            # 在向量存储中搜索包含关键词的记忆
            search_results = await vector_db.search_by_keyword(keyword, k=20)
            
            # 提取要删除的ID
            ids_to_delete = [result["id"] for result in search_results if keyword.lower() in result["document"].lower()]
            
            # 删除这些记忆
            if ids_to_delete:
                await vector_db.delete(ids_to_delete)
            
            deleted_count = len(ids_to_delete)
            
            # 从知识图谱中删除相关三元组
            try:
                import asyncio
                kg = await asyncio.to_thread(self.kg_memory.kg.get_triples)
                for triple in kg:
                    if keyword.lower() in triple[0].lower() or keyword.lower() in triple[2].lower():
                        await asyncio.to_thread(self.kg_memory.kg.remove_triple, triple)
            except Exception as e:
                logger.error(f"❌ [Memory] Failed to remove knowledge graph triples: {e}")
            
            # 从实体记忆中删除相关实体
            try:
                entities_to_remove = []
                for entity_name, entity_data in self.entity_store.items():
                    if keyword.lower() in entity_name.lower() or keyword.lower() in str(entity_data).lower():
                        entities_to_remove.append(entity_name)
                
                for entity_name in entities_to_remove:
                    if entity_name in self.entity_store:
                        del self.entity_store[entity_name]
            except Exception as e:
                logger.error(f"❌ [Memory] Failed to remove entities: {e}")
            
            return deleted_count
        except Exception as e:
            logger.error(f"❌ [Memory] Failed to forget by keyword: {e}")
            return 0
    
    async def correct_memory(self, incorrect_memory: str, correct_memory: str, user_id: Optional[str] = None) -> bool:
        """
        纠正错误记忆
        
        Args:
            incorrect_memory: 要纠正的错误记忆内容
            correct_memory: 纠正后的正确记忆内容
            user_id: 可选的用户ID，用于过滤特定用户的记忆
            
        Returns:
            是否成功纠正记忆
        """
        try:
            # 1. 首先删除错误记忆
            deleted_count = await self.forget_by_semantic(incorrect_memory, threshold=0.6, user_id=user_id)
            
            # 2. 然后添加正确记忆
            # 更新向量存储记忆
            await vector_db.add_texts([correct_memory])
            
            # 更新知识图谱记忆（需要模拟一个对话来添加）
            try:
                import asyncio
                await asyncio.to_thread(self.kg_memory.save_context,
                    {"input": f"系统修正: {incorrect_memory} 是错误的，正确的应该是 {correct_memory}"},
                    {"output": "已修正记忆"}
                )
            except Exception as e:
                logger.error(f"❌ [Memory] Failed to update knowledge graph: {e}")
            
            # 不需要直接更新实体记忆，因为它会从对话中自动提取
            
            logger.info(f"✅ [Memory] Corrected memory: deleted {deleted_count} old items and added new correct memory")
            return True
        except Exception as e:
            logger.error(f"❌ [Memory] Failed to correct memory: {e}")
            return False
    
    async def clear_all_memory(self, user_id: Optional[str] = None) -> bool:
        """
        清除所有记忆
        
        Args:
            user_id: 可选的用户ID，用于过滤特定用户的记忆
            
        Returns:
            是否成功清除所有记忆
        """
        try:
            # 清除向量存储记忆
            await vector_db.clear_all()
            
            # 清除实体记忆
            self.entity_store = {}
            
            # 清除知识图谱记忆 - 不使用Neo4jGraph
            import asyncio
            await asyncio.to_thread(lambda: setattr(self.kg_memory, 'kg', type(self.kg_memory.kg)()))
            
            logger.info("✅ [Memory] All memory cleared successfully")
            return True
        except Exception as e:
            logger.error(f"❌ [Memory] Failed to clear all memory: {e}")
            return False

# 创建全局实例
combined_memory = CombinedMemoryManager()