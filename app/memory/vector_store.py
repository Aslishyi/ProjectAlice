import chromadb
from typing import List
from datetime import datetime
import math
import threading # 引入 threading
from openai import OpenAI
from app.core.config import config


class VectorMemory:
    def __init__(self):
        # 1. 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
        self._lock = threading.Lock() # 初始化互斥锁

        # 2. 初始化 OpenAI 客户端
        self.openai_client = OpenAI(
            api_key=config.SILICONFLOW_API_KEY,
            base_url=config.SILICONFLOW_BASE_URL
        )

        # 3. 适配器
        class SiliconFlowAdapter:
            def __init__(self, client, model_name):
                self.client = client
                self.model_name = model_name

            def _embed(self, texts: List[str]) -> List[List[float]]:
                texts = [t.replace("\n", " ") for t in texts]
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model_name
                )
                return [data.embedding for data in response.data]

            def __call__(self, input: List[str]) -> List[List[float]]:
                return self._embed(input)

            def embed_documents(self, input: List[str]) -> List[List[float]]:
                return self._embed(input)

            def embed_query(self, input: List[str]) -> List[List[float]]:
                return self._embed(input)

            def name(self):
                return "siliconflow_native_adapter"

        self.embedding_fn = SiliconFlowAdapter(
            self.openai_client,
            config.EMBEDDING_MODEL_NAME
        )

        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

    def add_texts(self, texts: List[str], metadatas: List[dict] = None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not texts: return

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

        # 加锁写入
        with self._lock:
            try:
                self.collection.upsert(
                    documents=texts,
                    metadatas=final_metadatas,
                    ids=ids
                )
            except Exception as e:
                print(f"[{ts}] ❌ [VectorStore Write Error] {e}")

    def _calculate_time_decay(self, created_at_str: str, half_life_hours: float = 48.0) -> float:
        try:
            mem_time = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
            delta_hours = (datetime.now() - mem_time).total_seconds() / 3600.0
            decay = max(0.3, math.pow(0.5, delta_hours / half_life_hours))
            return decay
        except:
            return 1.0

    def search(self, query: str, k: int = 3) -> List[str]:
        # 查询通常不需要严格加锁，但为了防止读取时正好在发生 vacuum，加上更安全
        # 如果追求性能，查询可以不加锁，只要 Chroma 版本够新
        with self._lock:
            try:
                results = self.collection.query(
                    query_texts=[query],
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

    def delete_by_semantic(self, query: str, threshold: float = 0.25):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=5
                )
                ids_to_delete = []
                if results["ids"]:
                    for id_val, dist in zip(results["ids"][0], results["distances"][0]):
                        if dist < threshold:
                            ids_to_delete.append(id_val)

                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                    print(f"[{ts}] 🧹 [Memory] Deleted {len(ids_to_delete)} items.")
                    return len(ids_to_delete)
                return 0
            except Exception as e:
                print(f"[{ts}] [VectorStore] Delete error: {e}")
                return 0


vector_db = VectorMemory()
