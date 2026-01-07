import chromadb
from typing import List, Dict, Any, Optional
from datetime import datetime
import math
import threading
from openai import OpenAI
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from app.core.config import config


class VectorMemory(VectorStore):
    """
    向量存储器，实现了LangChain的VectorStore接口
    """
    
    def __init__(self):
        # 1. 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
        self._lock = threading.Lock()  # 初始化互斥锁

        # 2. 初始化 OpenAI 客户端
        self.openai_client = OpenAI(
            api_key=config.SILICONFLOW_API_KEY,
            base_url=config.SILICONFLOW_BASE_URL
        )

        # 3. 初始化嵌入函数
        self.embedding_model = config.EMBEDDING_MODEL_NAME

        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME
        )

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """手动生成嵌入向量"""
        texts = [t.replace("\n", " ") for t in texts]
        response = self.openai_client.embeddings.create(
            input=texts,
            model=self.embedding_model
        )
        return [data.embedding for data in response.data]

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None,
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

        # 手动生成嵌入向量
        embeddings = self._generate_embeddings(texts)

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
                print(f"[{ts}] ❌ [VectorStore Write Error] {e}")
                return []

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """相似性搜索"""
        with self._lock:
            try:
                # 手动生成查询嵌入向量
                query_embedding = self._generate_embeddings([query])[0]
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k * 3,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                print(f"[VectorStore Error] Search failed: {e}")
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

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[tuple[Document, float]]:
        """带分数的相似性搜索"""
        with self._lock:
            try:
                # 手动生成查询嵌入向量
                query_embedding = self._generate_embeddings([query])[0]
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k * 3,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                print(f"[VectorStore Error] Search failed: {e}")
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
        """计算时间衰减因子"""
        try:
            mem_time = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            delta_hours = (datetime.now() - mem_time).total_seconds() / 3600.0
            decay = max(0.3, math.pow(0.5, delta_hours / half_life_hours))
            return decay
        except:
            return 1.0

    def search(self, query: str, k: int = 3) -> List[str]:
        """自定义搜索，考虑时间衰减和重要性"""
        with self._lock:
            try:
                # 手动生成查询嵌入向量
                query_embedding = self._generate_embeddings([query])[0]
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k * 3,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                print(f"[VectorStore Error] Search failed: {e}")
                return []

        if not results["documents"]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        scored_candidates = []

        for doc, meta, dist in zip(docs, metas, dists):
            semantic_score = 1.0 / (1.0 + dist)
            created_at = meta.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            time_score = self._calculate_time_decay(created_at)
            importance = float(meta.get("importance", 1))
            imp_boost = 1.0 + (importance * 0.15)

            final_score = semantic_score * time_score * imp_boost
            scored_candidates.append((final_score, doc))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_docs = [item[1] for item in scored_candidates[:k]]

        return top_docs

    def delete_by_semantic(self, query: str, threshold: float = 0.3):
        """通过语义删除相似项"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            try:
                # 手动生成查询嵌入向量
                query_embedding = self._generate_embeddings([query])[0]
                
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
                    
                    print(f"DEBUG: Doc {i}: {doc[:50]}..., Similarity: {similarity:.4f}, Distance: {distance:.4f}")
                    
                    # 余弦相似度大于阈值时删除
                    if similarity > threshold:
                        # 使用文档内容查找对应的ID
                        if doc in doc_id_map:
                            ids_to_delete.append(doc_id_map[doc])
                            print(f"DEBUG: Found ID {doc_id_map[doc]} for document")
                
                print(f"DEBUG: Total ids to delete: {len(ids_to_delete)}")
                print(f"DEBUG: Ids to delete: {ids_to_delete}")
                
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                    print(f"[{ts}] 🧹 [Memory] Deleted {len(ids_to_delete)} items.")
                    return len(ids_to_delete)
                return 0
            except Exception as e:
                print(f"[{ts}] [VectorStore] Delete error: {e}")
                return 0

    # 实现LangChain VectorStore接口的其他必要方法
    def as_retriever(self, **kwargs):
        """转换为检索器"""
        from langchain_classic.memory.vectorstore import VectorStoreRetriever
        return VectorStoreRetriever(vectorstore=self, **kwargs)

    @classmethod
    def from_documents(cls, documents: List[Document], embedding, **kwargs):
        """从文档创建"""
        instance = cls()
        texts = [doc.page_content for doc in documents]
        metas = [doc.metadata for doc in documents]
        instance.add_texts(texts, metas)
        return instance

    def delete(self, ids: List[str], **kwargs):
        """删除指定ID的文档"""
        with self._lock:
            try:
                self.collection.delete(ids=ids)
                return True
            except Exception as e:
                print(f"❌ [VectorStore] Delete error: {e}")
                return False
                
    def search_by_keyword(self, keyword: str, k: int = 10) -> List[Dict[str, Any]]:
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
                keyword_embedding = self._generate_embeddings([keyword])[0]
                
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
                print(f"❌ [VectorStore] Keyword search error: {e}")
                return []
                
    def clear_all(self) -> bool:
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
                print(f"❌ [VectorStore] Clear all error: {e}")
                return False

    @classmethod
    def from_texts(cls, texts: List[str], embedding, metadatas: Optional[List[dict]] = None, **kwargs):
        """从文本创建向量存储"""
        instance = cls()
        instance.add_texts(texts, metadatas)
        return instance


vector_db = VectorMemory()
