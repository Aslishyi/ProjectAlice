import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import config
from app.memory.vector_store import vector_db
from app.core.global_store import global_store
from app.utils.cache import cached_llm_invoke

logger = logging.getLogger("DreamCycle")

# --- 记忆固化 Prompt ---
CONSOLIDATION_PROMPT = """
你是 Alice 的潜意识整理者。你的任务是将碎片化的短期记忆合并为有价值的长期记忆。

【待整理的记忆碎片】
{fragments}

【任务要求】
1. 分析这些碎片之间是否存在关联（例如：都是关于饮食偏好、都是关于某个特定项目、或者是连续的事件）。
2. 如果存在关联，请将它们**概括**为一条简洁的、包含核心信息的陈述句。
3. 概括后的记忆应当去除时间状语（如“刚才”、“今天”），转变为持久的事实描述。
4. 如果碎片之间没有明显关联，或者信息太杂乱无法合并，请输出 "SKIP"。

【输出示例】
输入碎片: ["用户说今天想吃辣", "中午点了麻辣烫", "晚上还在找火锅店"]
输出: 用户非常喜欢吃辣的食物，尤其是麻辣烫和火锅。

请输出结果 (纯文本):
"""


class DreamCycle:
    def __init__(self, interval_seconds=1800):
        """
        :param interval_seconds: 做梦循环的间隔，默认 30 分钟 (1800秒)
        """
        self.interval = interval_seconds
        self.running = False
        self._task = None

        # 专门用于整理记忆的 LLM，可以使用便宜的模型 (如 gpt-3.5-turbo 或 qwen-turbo)
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=0.1,
            api_key=config.MODEL_API_KEY,
            base_url=config.MODEL_URL
        )

    async def start(self):
        # 在Windows上使用文件锁确保只有一个进程能启动DreamCycle
        lock_file_path = os.path.join(os.path.dirname(__file__), "dream_lock.lock")
        lock_file = None
        
        try:
            # 尝试打开锁文件
            lock_file = open(lock_file_path, 'w')
            
            # 检查操作系统类型
            if os.name == 'nt':  # Windows
                # 在Windows上使用msvcrt.lock来获取文件锁
                import msvcrt
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    # 如果成功获取锁，保存文件对象并启动DreamCycle
                    self._lock_file = lock_file
                    self.running = True
                    self._task = asyncio.create_task(self._dream_loop())
                    logger.info("💤 [Dream] Background memory consolidation module started.")
                except IOError:
                    # 无法获取锁，说明已经有其他进程在运行DreamCycle
                    logger.info("💤 [DreamCycle] Already running in another process. Skipping startup.")
                    lock_file.close()
                    return
            else:  # 非Windows
                # 如果是在非Windows平台，使用fcntl
                try:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # 如果成功获取锁，保存文件对象并启动DreamCycle
                    self._lock_file = lock_file
                    self.running = True
                    self._task = asyncio.create_task(self._dream_loop())
                    logger.info("💤 [Dream] Background memory consolidation module started.")
                except (BlockingIOError, IOError):
                    # 无法获取锁，说明已经有其他进程在运行DreamCycle
                    logger.info("💤 [DreamCycle] Already running in another process. Skipping startup.")
                    lock_file.close()
                    return
        except Exception as e:
            # 处理其他可能的异常
            logger.error(f"💤 [DreamCycle] Error during startup: {e}")
            if lock_file:
                lock_file.close()
            return

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # 释放文件锁
        if hasattr(self, '_lock_file') and self._lock_file:
            try:
                import msvcrt
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            except ImportError:
                try:
                    import fcntl
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                except:
                    pass
            finally:
                self._lock_file.close()
                logger.info("💤 [Dream] Background memory consolidation module stopped.")

    async def _dream_loop(self):
        while self.running:
            try:
                # 等待下一个周期
                await asyncio.sleep(self.interval)

                # 1. 检查活跃度：如果用户最近 5 分钟还在说话，不要做梦，避免数据库锁冲突
                last_active_str = global_store.get_emotion_snapshot().last_updated
                last_active = datetime.strptime(last_active_str, "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - last_active).total_seconds() < 300:
                    logger.info("💤 [Dream] User is active. Postponing dream cycle.")
                    continue

                logger.info("💤 [Dream] Entering REM sleep (Memory Optimization)...")

                # 2. 执行清理逻辑
                deleted_count = self._prune_garbage_memories(days_threshold=3)

                # 3. 执行固化逻辑
                consolidated_count = await self._consolidate_memories()

                # 4. 恢复体力 (作为奖励)
                if deleted_count > 0 or consolidated_count > 0:
                    global_store.update_emotion(0, 0, stamina_delta=30.0)
                    logger.info(
                        f"💤 [Dream] Cycle Done. Pruned: {deleted_count}, Consolidated: {consolidated_count}. Stamina Recovered.")
                else:
                    logger.info("💤 [Dream] Deep sleep. No memories needed processing.")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ [Dream Error] {e}", exc_info=True)

    def _prune_garbage_memories(self, days_threshold: int = 3) -> int:
        """
        清理逻辑：删除 [importance=1] 且 [创建时间 > 3天] 的记忆
        """
        try:
            # Chroma API 获取所有 metadata (limit 设大一点以覆盖)
            # 注意：如果数据量巨大，这里需要分页处理，Demo 中暂且一次性获取
            result = vector_db.collection.get(include=["metadatas"])

            ids = result["ids"]
            metadatas = result["metadatas"]

            ids_to_delete = []
            now = datetime.now()
            cutoff_date = now - timedelta(days=days_threshold)

            for i, meta in enumerate(metadatas):
                # 检查 Importance (如果没有字段，默认为 1)
                importance = meta.get("importance", 1)

                # 只清理低权重记忆
                if importance > 1:
                    continue

                # 检查时间
                created_at_str = meta.get("created_at")
                if created_at_str:
                    try:
                        mem_time = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                        if mem_time < cutoff_date:
                            ids_to_delete.append(ids[i])
                    except ValueError:
                        continue  # 时间格式不对则跳过

            if ids_to_delete:
                logger.info(f"🧹 [Dream] Pruning {len(ids_to_delete)} garbage memories...")
                vector_db.collection.delete(ids=ids_to_delete)
                return len(ids_to_delete)

            return 0

        except Exception as e:
            logger.error(f"Error in pruning: {e}")
            return 0

    async def _consolidate_memories(self) -> int:
        """
        固化逻辑：
        1. 找出最近 24 小时产生的、importance=2 (Context) 或 3 (Preference) 的记忆。
        2. 如果碎片数量 > 3，尝试让 LLM 总结。
        3. 如果总结成功，写入一条 importance=4 的新记忆，并删除旧碎片。
        """
        try:
            # 1. 获取最近记忆
            result = vector_db.collection.get(include=["documents", "metadatas"])
            ids = result["ids"]
            docs = result["documents"]
            metadatas = result["metadatas"]

            candidates = []  # list of (id, doc, meta)
            now = datetime.now()

            # 筛选：最近 24 小时 且 重要性为 2 或 3
            for i, meta in enumerate(metadatas):
                imp = meta.get("importance", 1)
                if imp not in [2, 3]:
                    continue

                c_time_str = meta.get("created_at")
                if not c_time_str: continue

                try:
                    mem_time = datetime.strptime(c_time_str, "%Y-%m-%d %H:%M:%S")
                    # 只看最近 24 小时
                    if (now - mem_time).total_seconds() < 86400:
                        candidates.append((ids[i], docs[i]))
                except:
                    continue

            # 如果碎片太少，没必要总结
            if len(candidates) < 4:
                return 0

            # 2. 准备 Prompt 数据 (取前 10 条处理，避免 token 爆炸)
            batch = candidates[:10]
            batch_texts = [item[1] for item in batch]
            batch_ids = [item[0] for item in batch]

            fragments_text = json.dumps(batch_texts, ensure_ascii=False, indent=2)

            # 3. LLM 思考
            logger.info(f"🧠 [Dream] Attempting to consolidate {len(batch)} fragments...")

            prompt = CONSOLIDATION_PROMPT.format(fragments=fragments_text)
            response = await cached_llm_invoke(self.llm, [SystemMessage(content=prompt)], temperature=self.llm.temperature)
            result_text = response.content.strip()

            # 4. 处理结果
            if "SKIP" in result_text or len(result_text) < 5:
                # 无法合并，保持原样
                return 0

            # 5. 执行“新陈代谢”
            logger.info(f"✨ [Dream] Consolidation Success: '{result_text}'")

            # A. 写入新记忆 (Importance = 4, 表示这是经过深思熟虑的事实)
            new_metadata = {
                "source": "dream_consolidation",
                "importance": 4,
                "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "consolidated_from_count": len(batch)
            }
            await vector_db.add_texts([result_text], [new_metadata])

            # B. 删除旧碎片 (物理删除，释放空间)
            # vector_db.collection.delete(ids=batch_ids) # 暂时注释掉，为了调试安全。确认稳定后取消注释。
            # 这里我们做一个折中：不删除，而是将其 importance 降级为 0，等待下次 Pruning 清理
            # 但 Chroma update 比较麻烦，所以直接删除是比较干净的做法。
            # 生产环境建议开启删除：
            vector_db.collection.delete(ids=batch_ids)

            return 1

        except Exception as e:
            logger.error(f"Error in consolidation: {e}")
            return 0


# 单例导出
dream_machine = DreamCycle(interval_seconds=1800)
