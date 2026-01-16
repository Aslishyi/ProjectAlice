# app/core/persona_vector_manager.py
import os
import json
import logging
from typing import List, Dict, Any, Optional
from app.memory.vector_store import vector_db
from app.core.config import config

# 配置日志
logger = logging.getLogger("PersonaVectorManager")

# 人设文件路径
PERSONA_DIR = os.path.join(os.path.dirname(__file__), 'persona')
EXTENDED_PERSONA_FILE = os.path.join(PERSONA_DIR, 'extended_persona.json')
CONTEXTUAL_PERSONA_FILE = os.path.join(PERSONA_DIR, 'contextual_persona.json')

class PersonaVectorManager:
    """
    人设向量管理类，用于将人设信息存储到向量数据库并进行检索
    """
    
    def __init__(self):
        # 使用单独的集合名称存储不同类型的人设信息
        self.extended_persona_collection_name = "extended_persona_collection"
        self.contextual_persona_collection_name = "contextual_persona_collection"
        self.persona_category = "persona"
        
        # 创建单独的向量数据库客户端和集合
        import chromadb
        import threading
        from app.core.config import config
        
        # 创建单独的客户端，使用与主向量数据库相同的路径
        self.client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
        
        # 创建或获取扩展人设专用集合
        self.extended_collection = self.client.get_or_create_collection(
            name=self.extended_persona_collection_name
        )
        
        # 创建或获取场景说话风格专用集合
        self.contextual_collection = self.client.get_or_create_collection(
            name=self.contextual_persona_collection_name
        )
        
        # 初始化嵌入模型
        from openai import AsyncOpenAI
        self.openai_client = AsyncOpenAI(
            api_key=config.SILICONFLOW_API_KEY,
            base_url=config.SILICONFLOW_BASE_URL
        )
        self.embedding_model = config.EMBEDDING_MODEL_NAME
        
        # 初始化互斥锁
        self._lock = threading.Lock()
        
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        手动生成嵌入向量，带缓存
        """
        import asyncio
        from app.utils.cache import cached_embedding_get, cached_embedding_set
        
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # 先检查缓存
        for i, text in enumerate(texts):
            cached_emb = await cached_embedding_get(text, self.embedding_model)
            if cached_emb:
                embeddings.append(cached_emb)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 如果有未缓存的文本，调用API获取嵌入向量
        if uncached_texts:
            response = await self.openai_client.embeddings.create(
                input=uncached_texts,
                model=self.embedding_model
            )
            uncached_embeddings = [data.embedding for data in response.data]
            
            # 将新获取的嵌入向量加入结果并缓存
            for idx, text, emb in zip(uncached_indices, uncached_texts, uncached_embeddings):
                embeddings.insert(idx, emb)
                await cached_embedding_set(text, self.embedding_model, emb)
        
        return embeddings
    
    async def load_and_index_extended_persona(self):
        """
        加载扩展人设文件并将其索引到向量数据库
        """
        logger.info("开始加载并索引扩展人设信息")
        
        try:
            # 加载扩展人设
            with open(EXTENDED_PERSONA_FILE, 'r', encoding='utf-8') as f:
                extended_persona = json.load(f)
            
            # 将扩展人设转换为可索引的文本
            extended_texts = []
            extended_metadatas = []
            extended_ids = []
            
            import hashlib
            from datetime import datetime
            
            # 处理扩展人设
            for category, details in extended_persona.items():
                # 支持不同的JSON格式，不强制要求嵌套结构
                if isinstance(details, dict):
                    for sub_category, sub_details in details.items():
                        if isinstance(sub_details, dict):
                            for key, value in sub_details.items():
                                if isinstance(value, list):
                                    value_str = ", ".join(value)
                                    text = f"{category} - {sub_category} - {key}: {value_str}"
                                else:
                                    text = f"{category} - {sub_category} - {key}: {value}"
                                
                                metadata = {
                                    "category": self.persona_category,
                                    "source": "extended_persona",
                                    "persona_category": category,
                                    "persona_subcategory": sub_category,
                                    "persona_key": key,
                                    "importance": 1.0,
                                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                
                                # 生成唯一ID
                                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                                persona_id = f"extended_persona_{text_hash}"
                                
                                extended_texts.append(text)
                                extended_metadatas.append(metadata)
                                extended_ids.append(persona_id)
                        elif isinstance(sub_details, list):
                            value_str = ", ".join(sub_details)
                            text = f"{category} - {sub_category}: {value_str}"
                            
                            metadata = {
                                "category": self.persona_category,
                                "source": "extended_persona",
                                "persona_category": category,
                                "persona_subcategory": sub_category,
                                "importance": 1.0,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # 生成唯一ID
                            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                            persona_id = f"extended_persona_{text_hash}"
                            
                            extended_texts.append(text)
                            extended_metadatas.append(metadata)
                            extended_ids.append(persona_id)
                        else:
                            text = f"{category} - {sub_category}: {sub_details}"
                            
                            metadata = {
                                "category": self.persona_category,
                                "source": "extended_persona",
                                "persona_category": category,
                                "persona_subcategory": sub_category,
                                "importance": 1.0,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # 生成唯一ID
                            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                            persona_id = f"extended_persona_{text_hash}"
                            
                            extended_texts.append(text)
                            extended_metadatas.append(metadata)
                            extended_ids.append(persona_id)
                else:
                    # 支持直接的键值对格式
                    text = f"{category}: {details}"
                    
                    metadata = {
                        "category": self.persona_category,
                        "source": "extended_persona",
                        "persona_category": category,
                        "importance": 1.0,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # 生成唯一ID
                    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                    persona_id = f"extended_persona_{text_hash}"
                    
                    extended_texts.append(text)
                    extended_metadatas.append(metadata)
                    extended_ids.append(persona_id)
            
            # 将扩展人设信息添加到向量数据库
            if extended_texts:
                logger.info(f"总共准备了 {len(extended_texts)} 条扩展人设信息用于索引")
                # 生成嵌入向量
                embeddings = await self._generate_embeddings(extended_texts)
                
                # 加锁写入
                with self._lock:
                    try:
                        self.extended_collection.upsert(
                            documents=extended_texts,
                            embeddings=embeddings,
                            metadatas=extended_metadatas,
                            ids=extended_ids
                        )
                        logger.info(f"成功索引了 {len(extended_texts)} 条扩展人设信息到专用集合")
                    except Exception as e:
                        logger.error(f"将扩展人设信息写入向量数据库失败: {e}")
        
        except Exception as e:
            logger.error(f"加载并索引扩展人设信息失败: {e}")
    
    async def load_and_index_contextual_persona(self):
        """
        加载场景说话风格文件并将其索引到向量数据库
        """
        logger.info("开始加载并索引场景说话风格信息")
        
        try:
            # 加载场景说话风格
            with open(CONTEXTUAL_PERSONA_FILE, 'r', encoding='utf-8') as f:
                contextual_persona = json.load(f)
            
            # 将场景说话风格转换为可索引的文本
            contextual_texts = []
            contextual_metadatas = []
            contextual_ids = []
            
            import hashlib
            from datetime import datetime
            
            # 处理场景说话风格
            logger.info("开始处理场景说话风格信息")
            
            # 处理情绪维度
            if "情绪维度" in contextual_persona:
                for emotion, details in contextual_persona["情绪维度"].items():
                    if isinstance(details, dict):
                        # 构建情绪说话风格的文本表示
                        style_text = f"【情绪说话风格 - {emotion}】"
                        for key, value in details.items():
                            style_text += f"\n{key}: {value}"
                        
                        metadata = {
                            "category": self.persona_category,
                            "source": "contextual_persona",
                            "persona_type": "emotion_style",
                            "emotion": emotion,
                            "importance": 1.0,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 生成唯一ID
                        text_hash = hashlib.md5(f"style_emotion_{emotion}".encode('utf-8')).hexdigest()
                        persona_id = f"contextual_persona_{text_hash}"
                        
                        # 只添加完整的说话风格文本到索引列表
                        contextual_texts.append(style_text)
                        contextual_metadatas.append(metadata)
                        contextual_ids.append(persona_id)
            
            # 处理关系维度
            if "关系维度" in contextual_persona:
                for relation, details in contextual_persona["关系维度"].items():
                    if isinstance(details, dict):
                        # 构建关系说话风格的文本表示
                        style_text = f"【关系说话风格 - {relation}】"
                        for key, value in details.items():
                            style_text += f"\n{key}: {value}"
                        
                        metadata = {
                            "category": self.persona_category,
                            "source": "contextual_persona",
                            "persona_type": "relation_style",
                            "relation": relation,
                            "importance": 1.0,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 生成唯一ID
                        text_hash = hashlib.md5(f"style_relation_{relation}".encode('utf-8')).hexdigest()
                        persona_id = f"contextual_persona_{text_hash}"
                        
                        # 只添加完整的说话风格文本到索引列表
                        contextual_texts.append(style_text)
                        contextual_metadatas.append(metadata)
                        contextual_ids.append(persona_id)
            
            # 处理场景维度
            if "场景维度" in contextual_persona:
                for scene, details in contextual_persona["场景维度"].items():
                    if isinstance(details, dict):
                        # 构建场景说话风格的文本表示
                        style_text = f"【场景说话风格 - {scene}】"
                        for key, value in details.items():
                            style_text += f"\n{key}: {value}"
                        
                        metadata = {
                            "category": self.persona_category,
                            "source": "contextual_persona",
                            "persona_type": "scene_style",
                            "scene": scene,
                            "importance": 1.0,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 生成唯一ID
                        text_hash = hashlib.md5(f"style_scene_{scene}".encode('utf-8')).hexdigest()
                        persona_id = f"contextual_persona_{text_hash}"
                        
                        # 只添加完整的说话风格文本到索引列表
                        contextual_texts.append(style_text)
                        contextual_metadatas.append(metadata)
                        contextual_ids.append(persona_id)
            
            # 处理综合场景
            if "综合场景" in contextual_persona:
                for scene_key, details in contextual_persona["综合场景"].items():
                    if isinstance(details, dict):
                        # 解析综合场景键
                        try:
                            emotion, relation, scene = scene_key.split("-")
                        except ValueError:
                            continue  # 跳过格式不正确的综合场景
                        
                        # 构建综合说话风格的文本表示
                        style_text = f"【综合说话风格 - {scene_key}】"
                        for key, value in details.items():
                            style_text += f"\n{key}: {value}"
                        
                        metadata = {
                            "category": self.persona_category,
                            "source": "contextual_persona",
                            "persona_type": "comprehensive_style",
                            "emotion": emotion,
                            "relation": relation,
                            "scene": scene,
                            "comprehensive_key": scene_key,
                            "importance": 1.0,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 生成唯一ID
                        text_hash = hashlib.md5(f"style_comprehensive_{scene_key}".encode('utf-8')).hexdigest()
                        persona_id = f"contextual_persona_{text_hash}"
                        
                        # 只添加完整的说话风格文本到索引列表
                        contextual_texts.append(style_text)
                        contextual_metadatas.append(metadata)
                        contextual_ids.append(persona_id)
            
            # 将场景说话风格信息添加到向量数据库
            if contextual_texts:
                logger.info(f"总共准备了 {len(contextual_texts)} 条场景说话风格信息用于索引")
                # 生成嵌入向量
                embeddings = await self._generate_embeddings(contextual_texts)
                
                # 加锁写入
                with self._lock:
                    try:
                        self.contextual_collection.upsert(
                            documents=contextual_texts,
                            embeddings=embeddings,
                            metadatas=contextual_metadatas,
                            ids=contextual_ids
                        )
                        logger.info(f"成功索引了 {len(contextual_texts)} 条场景说话风格信息到专用集合")
                    except Exception as e:
                        logger.error(f"将场景说话风格信息写入向量数据库失败: {e}")
        
        except Exception as e:
            logger.error(f"加载并索引场景说话风格信息失败: {e}")
    
    async def load_and_index_persona(self):
        """
        加载所有人设文件并将其索引到向量数据库
        """
        # 先加载扩展人设
        await self.load_and_index_extended_persona()
        # 再加载场景说话风格
        await self.load_and_index_contextual_persona()
    
    async def search_extended_persona(self, query: str, k: int = 3) -> List[str]:
        """
        根据查询检索相关的扩展人设信息
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            
        Returns:
            List[str]: 检索到的扩展人设信息列表
        """
        try:
            # 生成查询的嵌入向量
            query_embedding = await self._generate_embeddings([query])
            if not query_embedding:
                return []
            
            # 在扩展人设专用集合中检索相关信息
            with self._lock:
                try:
                    results = self.extended_collection.query(
                        query_embeddings=query_embedding,
                        n_results=k,
                        where={"category": self.persona_category}
                    )
                except Exception as query_e:
                    # 处理向量数据库内部错误，可能需要重新索引
                    if "Error creating hnsw segment reader" in str(query_e):
                        logger.error(f"扩展人设向量索引损坏，正在重新索引...")
                        # 重新索引扩展人设
                        await self.load_and_index_extended_persona()
                        # 重新尝试检索
                        results = self.extended_collection.query(
                            query_embeddings=query_embedding,
                            n_results=k,
                            where={"category": self.persona_category}
                        )
                    else:
                        raise query_e
            
            # 提取检索结果
            retrieved_docs = []
            if results and "documents" in results:
                for doc_list in results["documents"]:
                    retrieved_docs.extend(doc_list)
            
            logger.info(f"成功从扩展人设专用集合中检索到 {len(retrieved_docs)} 条相关信息")
            return retrieved_docs
        except Exception as e:
            logger.error(f"检索扩展人设信息失败: {e}")
            return []
    
    async def search_contextual_persona(self, query: str, k: int = 3) -> List[str]:
        """
        根据查询检索相关的场景说话风格信息
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            
        Returns:
            List[str]: 检索到的场景说话风格信息列表
        """
        try:
            # 生成查询的嵌入向量
            query_embedding = await self._generate_embeddings([query])
            if not query_embedding:
                return []
            
            # 在场景说话风格专用集合中检索相关信息
            with self._lock:
                try:
                    results = self.contextual_collection.query(
                        query_embeddings=query_embedding,
                        n_results=k,
                        where={"category": self.persona_category}
                    )
                except Exception as query_e:
                    # 处理向量数据库内部错误，可能需要重新索引
                    if "Error creating hnsw segment reader" in str(query_e):
                        logger.error(f"场景说话风格向量索引损坏，正在重新索引...")
                        # 重新索引场景说话风格
                        await self.load_and_index_contextual_persona()
                        # 重新尝试检索
                        results = self.contextual_collection.query(
                            query_embeddings=query_embedding,
                            n_results=k,
                            where={"category": self.persona_category}
                        )
                    else:
                        raise query_e
            
            # 提取检索结果
            retrieved_docs = []
            if results and "documents" in results:
                for doc_list in results["documents"]:
                    retrieved_docs.extend(doc_list)
            
            logger.info(f"成功从场景说话风格专用集合中检索到 {len(retrieved_docs)} 条说话风格信息")
            return retrieved_docs
        except Exception as e:
            logger.error(f"检索场景说话风格信息失败: {e}")
            return []
    
    async def search_persona(self, query: str, k: int = 3) -> List[str]:
        """
        根据查询检索相关的所有人设信息（扩展人设 + 场景说话风格）
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            
        Returns:
            List[str]: 检索到的人设信息列表
        """
        # 分别检索扩展人设和场景说话风格
        extended_results = await self.search_extended_persona(query, k)
        contextual_results = await self.search_contextual_persona(query, k)
        
        # 合并结果
        all_results = extended_results + contextual_results
        
        logger.info(f"成功从所有人设集合中检索到 {len(all_results)} 条相关信息")
        return all_results
    
    async def update_persona_vector_store(self):
        """
        更新人设向量存储
        """
        logger.info("开始更新人设向量存储")
        
        try:
            # 清理现有扩展人设向量
            with self._lock:
                try:
                    # 获取所有现有扩展人设文档ID
                    existing_extended_docs = self.extended_collection.get(where={"category": self.persona_category})
                    if existing_extended_docs and "ids" in existing_extended_docs:
                        existing_extended_ids = existing_extended_docs["ids"]
                        if existing_extended_ids:
                            # 删除所有现有扩展人设文档
                            self.extended_collection.delete(ids=existing_extended_ids)
                            logger.info(f"成功删除了 {len(existing_extended_ids)} 条旧的扩展人设信息")
                except Exception as get_e:
                    # 处理向量数据库内部错误，可能需要重新创建集合
                    if "Error creating hnsw segment reader" in str(get_e):
                        logger.error(f"扩展人设向量集合损坏，正在重新创建...")
                        # 删除损坏的集合并重新创建
                        self.client.delete_collection(self.extended_persona_collection_name)
                        self.extended_collection = self.client.create_collection(
                            name=self.extended_persona_collection_name
                        )
                    else:
                        raise get_e
            
            # 清理现有场景说话风格向量
            with self._lock:
                try:
                    # 获取所有现有场景说话风格文档ID
                    existing_contextual_docs = self.contextual_collection.get(where={"category": self.persona_category})
                    if existing_contextual_docs and "ids" in existing_contextual_docs:
                        existing_contextual_ids = existing_contextual_docs["ids"]
                        if existing_contextual_ids:
                            # 删除所有现有场景说话风格文档
                            self.contextual_collection.delete(ids=existing_contextual_ids)
                            logger.info(f"成功删除了 {len(existing_contextual_ids)} 条旧的场景说话风格信息")
                except Exception as get_e:
                    # 处理向量数据库内部错误，可能需要重新创建集合
                    if "Error creating hnsw segment reader" in str(get_e):
                        logger.error(f"场景说话风格向量集合损坏，正在重新创建...")
                        # 删除损坏的集合并重新创建
                        self.client.delete_collection(self.contextual_persona_collection_name)
                        self.contextual_collection = self.client.create_collection(
                            name=self.contextual_persona_collection_name
                        )
                    else:
                        raise get_e
            
            # 重新索引扩展人设
            await self.load_and_index_extended_persona()
            
            # 重新索引场景说话风格
            await self.load_and_index_contextual_persona()
            
            logger.info("人设向量存储更新成功")
        except Exception as e:
            logger.error(f"更新人设向量存储失败: {e}")
            
    async def health_check(self):
        """
        检查人设向量存储的健康状态，如果发现问题则自动修复
        """
        logger.info("开始人设向量存储健康检查")
        
        try:
            # 检查扩展人设集合
            with self._lock:
                try:
                    extended_count = self.extended_collection.count()
                    logger.info(f"扩展人设集合健康检查通过，包含 {extended_count} 条文档")
                except Exception as e:
                    logger.error(f"扩展人设集合健康检查失败: {e}")
                    # 重新创建集合并索引
                    if "Error creating hnsw segment reader" in str(e):
                        logger.info("正在修复扩展人设集合...")
                        self.client.delete_collection(self.extended_persona_collection_name)
                        self.extended_collection = self.client.create_collection(
                            name=self.extended_persona_collection_name
                        )
                        await self.load_and_index_extended_persona()
                        logger.info("扩展人设集合修复成功")
            
            # 检查场景说话风格集合
            with self._lock:
                try:
                    contextual_count = self.contextual_collection.count()
                    logger.info(f"场景说话风格集合健康检查通过，包含 {contextual_count} 条文档")
                except Exception as e:
                    logger.error(f"场景说话风格集合健康检查失败: {e}")
                    # 重新创建集合并索引
                    if "Error creating hnsw segment reader" in str(e):
                        logger.info("正在修复场景说话风格集合...")
                        self.client.delete_collection(self.contextual_persona_collection_name)
                        self.contextual_collection = self.client.create_collection(
                            name=self.contextual_persona_collection_name
                        )
                        await self.load_and_index_contextual_persona()
                        logger.info("场景说话风格集合修复成功")
            
            logger.info("人设向量存储健康检查完成")
            return True
        except Exception as e:
            logger.error(f"人设向量存储健康检查失败: {e}")
            return False


# 创建全局实例
persona_vector_manager = PersonaVectorManager()