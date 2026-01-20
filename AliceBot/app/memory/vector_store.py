import chromadb
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import math
import threading
from openai import OpenAI
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from app.core.config import config
from app.utils.cache import cached_embedding_get, cached_embedding_set

# 配置日志
logger = logging.getLogger("VectorStore")


class VectorMemory(VectorStore):
    """
    向量存储器，实现了LangChain的VectorStore接口
    """
    
    def __init__(self):
        # 1. 检查并创建向量数据库目录
        import os
        import tempfile
        db_path = config.VECTOR_DB_PATH
        
        # 创建目录（如果不存在）
        if not os.path.exists(db_path):
            try:
                os.makedirs(db_path, exist_ok=True)
                logger.info(f"[VectorStore] Created vector database directory: {db_path}")
            except OSError as e:
                logger.error(f"[VectorStore] Failed to create vector database directory: {e}")
                raise
        
        # 检查目录是否可写
        try:
            with tempfile.NamedTemporaryFile(dir=db_path, delete=True):
                pass
            logger.info(f"[VectorStore] Vector database directory is writable: {db_path}")
        except OSError as e:
            logger.error(f"[VectorStore] Vector database directory is not writable: {e}")
            logger.error(f"[VectorStore] Please check directory permissions and ensure no other process is locking the database")
            raise
        
        # 2. 初始化 ChromaDB，禁用遥测并确保可写
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=chromadb.config.Settings(
                anonymized_telemetry=False
            )
        )
        self._lock = threading.Lock()  # 初始化互斥锁

        # 2. 初始化异步 OpenAI 客户端
        from openai import AsyncOpenAI
        self.openai_client = AsyncOpenAI(
            api_key=config.SILICONFLOW_API_KEY,
            base_url=config.SILICONFLOW_BASE_URL
        )

        # 3. 初始化嵌入函数
        self.embedding_model = config.EMBEDDING_MODEL_NAME

        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME
        )
        
        # 4. 标记清理任务未启动
        self._cleanup_task_started = False
    
    def start_cleanup_task(self):
        """手动启动定时清理任务
        
        只有当有运行的事件循环时才能调用此方法
        """
        if not self._cleanup_task_started:
            import asyncio
            asyncio.create_task(self._start_cleanup_task())
            self._cleanup_task_started = True
            logger.info("[VectorStore] 定时清理任务已启动")

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """手动生成嵌入向量，带缓存"""
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

    async def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None,
                  ids: Optional[List[str]] = None, **kwargs) -> List[str]:
        """添加文本到向量存储"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not texts: return []

        # 如果没有提供ID，生成唯一ID
        if not ids:
            ids = [f"mem_{hash(t)}" for t in texts]

        final_metadatas = []
        if metadatas:
            for m in metadatas:
                if "importance" not in m: m["importance"] = 1
                if "created_at" not in m: m["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                final_metadatas.append(m)
        else:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            final_metadatas = [{"source": "interaction", "importance": 1, "created_at": now_str}] * len(texts)

        # 手动生成嵌入向量（异步）
        embeddings = await self._generate_embeddings(texts)

        # 加锁写入
        with self._lock:
            try:
                self.collection.upsert(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=final_metadatas,
                    ids=ids
                )
                return ids
            except Exception as e:
                logger.error(f"[{ts}] ❌ [VectorStore Write Error] {e}")
                return []

    async def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """相似性搜索"""
        with self._lock:
            try:
                # 手动生成查询嵌入向量（异步）
                query_embedding = await self._generate_embeddings([query])[0]
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k * 3,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                logger.error(f"[VectorStore Error] Search failed: {e}")
                return []

        if not results["documents"]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        scored_candidates = []

        for doc, meta, dist in zip(docs, metas, dists):
            # 计算语义相似度得分（距离越小越相似）
            semantic_score = 1.0 / (1.0 + dist)
            document = Document(page_content=doc, metadata=meta)
            scored_candidates.append((document, semantic_score))

        # 按得分排序，取前k个
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [doc_score[0] for doc_score in scored_candidates[:k]]

    async def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[tuple[Document, float]]:
        """带分数的相似性搜索"""
        with self._lock:
            try:
                # 手动生成查询嵌入向量（异步）
                query_embedding = await self._generate_embeddings([query])[0]
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k * 3,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                logger.error(f"[VectorStore Error] Search failed: {e}")
                return []

        if not results["documents"]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        scored_candidates = []

        for doc, meta, dist in zip(docs, metas, dists):
            # 计算语义相似度得分（距离越小越相似）
            semantic_score = 1.0 / (1.0 + dist)
            document = Document(page_content=doc, metadata=meta)
            scored_candidates.append((document, semantic_score))

        # 按得分排序，取前k个
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:k]

    def _calculate_time_decay(self, created_at_str: str, half_life_hours: float = 48.0) -> float:
        """
        计算时间衰减因子，优化的衰减算法
        """
        try:
            mem_time = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            delta_hours = (datetime.now() - mem_time).total_seconds() / 3600.0
            
            # 优化的衰减算法：前24小时衰减较慢，之后加速衰减
            if delta_hours < 24:
                # 前24小时衰减较慢，half_life为96小时
                decay = max(0.2, math.pow(0.5, delta_hours / 96.0))
            else:
                # 24小时后加速衰减，使用指定的half_life
                decay = max(0.2, math.pow(0.5, delta_hours / half_life_hours))
            
            return decay
        except:
            return 1.0

    async def search(self, query: str, k: int = 3, categories: List[str] = None, source_boosts: Dict[str, float] = None, importance_threshold: float = 0.5) -> List[str]:
        """
        自定义搜索，考虑时间衰减和重要性，带缓存，支持分类筛选和自定义来源权重
        
        Args:
            query: 搜索查询
            k: 返回结果数量
            categories: 可选，指定要搜索的分类
            source_boosts: 可选，自定义来源权重
            importance_threshold: 可选，重要性阈值，过滤低于此阈值的记忆
            
        Returns:
            List[str]: 搜索结果列表
        """
        import asyncio
        from app.utils.cache import cached_context_get, cached_context_set
        
        # 构建缓存键，考虑所有参数
        cache_params = [f"{k}"]
        if categories:
            cache_params.extend(sorted(categories))
        if source_boosts:
            cache_params.extend([f"{k}:{v}" for k, v in sorted(source_boosts.items())])
        cache_params.append(f"{importance_threshold}")
        cache_key = f"vector_search:{hash(query)}:{':'.join(map(str, cache_params))}"
        
        # 先检查上下文缓存
        cached_results = await cached_context_get(cache_key)
        if cached_results:
            return cached_results
        
        # 生成查询嵌入向量（在锁外进行，提高并发性能）
        query_embedding = await self._generate_embeddings([query])
        query_embedding = query_embedding[0] if query_embedding else []
        
        with self._lock:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k * 5,  # 增加候选数量，提高选择质量
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                logger.error(f"[VectorStore Error] Search failed: {e}")
                return []

        if not results["documents"]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        scored_candidates = []
        seen_docs = set()  # 用于去重
        
        # 默认来源权重
        default_source_boosts = {
            "user_profile": 1.8,  # 提高用户资料的权重
            "chat_history": 1.3,  # 提高聊天历史的权重
            "interaction": 1.0,
            "system": 0.9
        }
        
        # 合并自定义来源权重
        final_source_boosts = default_source_boosts.copy()
        if source_boosts:
            final_source_boosts.update(source_boosts)

        for doc, meta, dist in zip(docs, metas, dists):
            # 去重
            if doc in seen_docs:
                continue
            seen_docs.add(doc)
            
            # 分类筛选
            if categories:
                doc_category = meta.get("category", "")
                if doc_category not in categories:
                    continue
            
            # 重要性阈值过滤
            importance = float(meta.get("importance", 1))
            if importance < importance_threshold:
                continue
            
            # 计算语义相似度得分（距离越小越相似）
            semantic_score = 1.0 / (1.0 + dist)
            
            # 计算时间衰减因子
            created_at = meta.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            time_score = self._calculate_time_decay(created_at)
            
            # 计算重要性权重
            imp_boost = 1.0 + (importance * 0.3)  # 增加重要性的影响
            
            # 计算来源权重
            source = meta.get("source", "interaction")
            source_boost = final_source_boosts.get(source, 1.0)
            
            # 计算最终得分
            final_score = semantic_score * time_score * imp_boost * source_boost
            
            # 如果文档包含关键词，给予额外奖励
            if query.lower() in doc.lower():
                final_score *= 1.1  # 关键词匹配奖励
            
            scored_candidates.append((final_score, doc))

        # 按得分排序，取前k个
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_docs = [item[1] for item in scored_candidates[:k]]
        
        # 将结果存入上下文缓存
        await cached_context_set(cache_key, top_docs, ttl=3600)  # 增加缓存时间到1小时

        return top_docs
        
    async def search_by_category(self, category: str, query: str = None, k: int = 5) -> List[str]:
        """
        按分类搜索记忆
        
        Args:
            category: 要搜索的分类
            query: 可选，搜索查询
            k: 返回结果数量
            
        Returns:
            List[str]: 搜索结果列表
        """
        if not query:
            # 如果没有查询词，直接使用分类名作为查询
            query = category
        
        return await self.search(query, k=k, categories=[category])

    async def delete_by_semantic(self, query: str, threshold: float = 0.3):
        """通过语义删除相似项"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            try:
                # 手动生成查询嵌入向量（异步）
                query_embedding = await self._generate_embeddings([query])[0]
                
                # 使用查询获取相似的文档
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=10,
                    include=["documents", "metadatas", "distances", "embeddings"]
                )
                
                if not results["documents"] or not results["documents"][0]:
                    return 0
                
                # 计算查询与所有文档的相似度
                import numpy as np
                query_emb = np.array(query_embedding)
                ids_to_delete = []
                
                # 先获取所有文档ID和内容的映射，以便快速查找
                all_docs = self.collection.get()
                doc_id_map = {}
                for doc_id, doc_content in zip(all_docs["ids"], all_docs["documents"]):
                    doc_id_map[doc_content] = doc_id
                
                for i in range(len(results["documents"][0])):
                    doc = results["documents"][0][i]
                    embedding = results["embeddings"][0][i]
                    distance = results["distances"][0][i]
                    
                    doc_emb = np.array(embedding)
                    # 计算余弦相似度
                    similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                    
                    logger.debug(f"Doc {i}: {doc[:50]}..., Similarity: {similarity:.4f}, Distance: {distance:.4f}")
                    
                    # 余弦相似度大于阈值时删除
                    if similarity > threshold:
                        # 使用文档内容查找对应的ID
                        if doc in doc_id_map:
                            ids_to_delete.append(doc_id_map[doc])
                            logger.debug(f"Found ID {doc_id_map[doc]} for document")
                
                logger.debug(f"Total ids to delete: {len(ids_to_delete)}")
                logger.debug(f"Ids to delete: {ids_to_delete}")
                
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                    logger.info(f"[{ts}] 🧹 [Memory] Deleted {len(ids_to_delete)} items.")
                    return len(ids_to_delete)
                return 0
            except Exception as e:
                logger.error(f"[{ts}] [VectorStore] Delete error: {e}")
                return 0

    # 实现LangChain VectorStore接口的其他必要方法
    def as_retriever(self, **kwargs):
        """转换为检索器"""
        from langchain_classic.memory.vectorstore import VectorStoreRetriever
        return VectorStoreRetriever(vectorstore=self, **kwargs)

    @classmethod
    async def from_documents(cls, documents: List[Document], embedding, **kwargs):
        """从文档创建"""
        instance = cls()
        texts = [doc.page_content for doc in documents]
        metas = [doc.metadata for doc in documents]
        await instance.add_texts(texts, metas)
        return instance

    async def delete(self, ids: List[str], **kwargs):
        """删除指定ID的文档"""
        with self._lock:
            try:
                self.collection.delete(ids=ids)
                return True
            except Exception as e:
                logger.error(f"❌ [VectorStore] Delete error: {e}")
                return False
                
    async def search_by_keyword(self, keyword: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        通过关键词搜索记忆
        
        Args:
            keyword: 要搜索的关键词
            k: 返回结果的数量
            
        Returns:
            包含document、metadata和distance的字典列表
        """
        with self._lock:
            try:
                # 获取所有文档内容和元数据
                all_docs = self.collection.get()
                
                if not all_docs["documents"]:
                    return []
                
                # 手动生成关键词嵌入向量（用于计算相似度）
                keyword_embedding = await self._generate_embeddings([keyword])[0]
                
                # 过滤包含关键词的文档
                import numpy as np
                keyword_emb = np.array(keyword_embedding)
                matching_docs = []
                
                # 遍历所有文档，查找包含关键词的文档
                for i in range(len(all_docs["documents"])):
                    doc = all_docs["documents"][i]
                    doc_id = all_docs["ids"][i]
                    meta = all_docs["metadatas"][i] if "metadatas" in all_docs and all_docs["metadatas"] else {}
                    
                    # 检查关键词是否在文档中
                    if keyword.lower() in doc.lower():
                        # 如果有embeddings，计算相似度用于排序
                        if "embeddings" in all_docs and all_docs["embeddings"]:
                            embedding = all_docs["embeddings"][i]
                            doc_emb = np.array(embedding)
                            similarity = np.dot(keyword_emb, doc_emb) / (np.linalg.norm(keyword_emb) * np.linalg.norm(doc_emb))
                            distance = 1 - similarity
                        else:
                            # 如果没有embeddings，默认距离为0.5
                            distance = 0.5
                        
                        matching_docs.append({
                            "id": doc_id,  # 使用实际的文档ID
                            "document": doc,
                            "metadata": meta,
                            "distance": distance
                        })
                
                # 按距离排序并取前k个
                matching_docs.sort(key=lambda x: x["distance"])
                return matching_docs[:k]
                
            except Exception as e:
                logger.error(f"❌ [VectorStore] Keyword search error: {e}")
                return []
                
    async def clear_all(self) -> bool:
        """
        清除向量存储中的所有记忆
        
        Returns:
            是否成功清除所有记忆
        """
        with self._lock:
            try:
                # 直接删除整个集合并重新创建
                self.client.delete_collection(name=self.collection.name)
                self.collection = self.client.get_or_create_collection(name=config.COLLECTION_NAME)
                return True
            except Exception as e:
                logger.error(f"❌ [VectorStore] Clear all error: {e}")
                return False

    @classmethod
    async def from_texts(cls, texts: List[str], embedding, metadatas: Optional[List[dict]] = None, **kwargs):
        """从文本创建向量存储"""
        instance = cls()
        await instance.add_texts(texts, metadatas)
        return instance
    
    async def _start_cleanup_task(self):
        """启动定时清理任务"""
        import time
        import asyncio
        from app.core.config import config
        
        # 获取清理间隔（默认6小时）
        cleanup_interval = getattr(config, 'VECTOR_DB_CLEANUP_INTERVAL', 6 * 3600)
        
        while True:
            try:
                # 等待指定时间
                await asyncio.sleep(cleanup_interval)
                
                # 执行清理
                await self._perform_cleanup()
            except Exception as e:
                logger.error(f"[VectorStore] 定时清理任务失败: {e}")
                # 发生错误后，等待较短时间后重试
                await asyncio.sleep(3600)  # 1小时后重试
    
    async def _perform_cleanup(self):
        """执行实际的清理操作"""
        logger.info("[VectorStore] 开始执行定时清理任务")
        
        # 1. 清理过时的记忆（超过30天的记忆）
        try:
            # 获取所有文档
            all_docs = self.collection.get(include=["documents", "metadatas"])
            
            if not all_docs["documents"]:
                logger.info("[VectorStore] 没有需要清理的文档")
                return
            
            from datetime import datetime, timedelta
            current_time = datetime.now()
            old_doc_ids = []
            
            # 找出超过30天的文档
            for i, metadata in enumerate(all_docs["metadatas"]):
                created_at = metadata.get("created_at")
                if created_at:
                    try:
                        doc_time = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                        if current_time - doc_time > timedelta(days=30):
                            old_doc_ids.append(all_docs["ids"][i])
                    except Exception:
                        # 如果日期格式不正确，跳过
                        continue
            
            # 删除过时的文档
            if old_doc_ids:
                self.collection.delete(ids=old_doc_ids)
                logger.info(f"[VectorStore] 删除了 {len(old_doc_ids)} 个过时的文档")
        except Exception as e:
            logger.error(f"[VectorStore] 清理过时文档失败: {e}")
        
        # 2. 使用语义相似度清理重复内容
        try:
            # 获取一些示例文档作为查询，用于找出相似的文档
            sample_docs = self.collection.get(n_results=10, include=["documents"])
            
            for doc in sample_docs["documents"]:
                if doc:
                    # 删除与示例文档相似度过高的文档（超过0.9）
                    deleted_count = await self.delete_by_semantic(doc, threshold=0.9)
                    if deleted_count > 0:
                        logger.info(f"[VectorStore] 通过语义删除了 {deleted_count} 个重复文档")
        except Exception as e:
            logger.error(f"[VectorStore] 语义清理失败: {e}")
        
        logger.info("[VectorStore] 定时清理任务完成")


vector_db = VectorMemory()
