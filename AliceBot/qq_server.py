# === Python代码文件: qq_server.py ===

# 首先配置日志和警告过滤
import logging
import warnings
import builtins
from langchain_core._api.deprecation import LangChainDeprecationWarning
from datetime import datetime
# 添加调试日志
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 先创建一个临时日志器来记录启动时的调试信息
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
temp_logger = logging.getLogger("DebugLogger")
temp_logger.debug(f"Current working directory: {os.getcwd()}")
temp_logger.debug(f"BASE_DIR in qq_server.py: {BASE_DIR}")

# 导入配置以查看实际路径
from app.core.config import config
temp_logger.debug(f"VECTOR_DB_PATH: {config.VECTOR_DB_PATH}")
temp_logger.debug(f"LOG_DIR: {config.LOG_DIR}")

# 过滤第三方库警告
warnings.filterwarnings("ignore", category=builtins.UserWarning, module="langchain_tavily")
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

import uvicorn
import asyncio
import uuid
import re
import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from langchain_core.messages import HumanMessage, AIMessage

from app.core.global_store import global_store
from app.graph.graph_builder import build_graph
from app.memory.relation_db import relation_db
from app.memory.local_history import LocalHistoryManager
from app.background.dream import dream_machine
from app.utils.qq_utils import parse_onebot_array_msg
from app.plugins.emoji_plugin.emoji_service import get_emoji_service
from app.plugins.emoji_plugin.emoji_manager import get_emoji_manager  # 兼容旧代码
from app.core.database import SessionLocal, ForwardMessageModel

# 配置根日志记录器
log_directory = os.path.join(os.path.dirname(__file__), "log")
log_file = os.path.join(log_directory, "logfile.log")

# 创建日志格式
log_format = logging.Formatter(
    "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 配置根日志记录器
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 清除现有处理器
root_logger.handlers.clear()

# 添加控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
root_logger.addHandler(console_handler)

# 添加文件处理器
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(log_format)
root_logger.addHandler(file_handler)

# 禁用Chromadb遥测日志
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)

logger = logging.getLogger("QQServer")


# --- 新增：会话活跃管理器 ---
class SessionManager:
    def __init__(self):
        # 记录 session_id -> {last_active: timestamp, type: 'group'/'private', target_id: str, self_id: str}
        self.sessions = {}
        self.lock = asyncio.Lock()

    async def update_activity(self, session_id: str, msg_type: str, target_id: str, self_id: str):
        async with self.lock:
            self.sessions[session_id] = {
                "last_active": time.time(),
                "type": msg_type,
                "target_id": target_id,
                "self_id": self_id
            }

    async def get_active_sessions(self, timeout_seconds=3600):
        """获取最近活跃的会话"""
        now = time.time()
        active = []
        async with self.lock:
            # 清理过期的 session (比如超过12小时没说话就不再主动搭理了)
            to_remove = []
            for sid, data in self.sessions.items():
                if now - data["last_active"] > 43200:  # 12小时
                    to_remove.append(sid)
                    continue
                active.append((sid, data))

            for sid in to_remove:
                del self.sessions[sid]
        return active


session_manager = SessionManager()


class MessageBuffer:
    """
    增强版消息批处理器
    支持基于时间和数量的双重批处理条件，以及不同会话类型的智能策略
    """
    def __init__(self):
        # 基础配置
        self.buffers = {}
        self.lock = asyncio.Lock()
        
        # 批处理策略配置 - 优化后的配置，进一步减少等待时间
        self.strategies = {
            "group": {
                "wait_time": 0.3,  # 群聊等待时间进一步缩短，提高响应速度
                "max_batch_size": 8,  # 群聊合并消息数量减少，加快响应
                "max_wait_time": 1.0,  # 最长等待时间进一步缩短
                "same_user_merge_window": 20,  # 同一用户消息合并窗口缩短
                "batch_merge_window": 0.8  # 批次合并窗口缩短
            },
            "private": {
                "wait_time": 0.5,  # 私聊等待时间进一步缩短
                "max_batch_size": 3,  # 私聊合并消息数量减少，保持对话流畅
                "max_wait_time": 1.2,  # 最长等待时间缩短
                "same_user_merge_window": 40,  # 同一用户消息合并窗口缩短
                "batch_merge_window": 1.0  # 批次合并窗口缩短
            }
        }

    def _get_session_type(self, session_id: str) -> str:
        """根据会话ID判断会话类型"""
        if session_id.startswith("group_"):
            return "group"
        elif session_id.startswith("private_"):
            return "private"
        return "private"  # 默认按私聊处理

    def _get_strategy(self, session_id: str) -> dict:
        """获取会话的批处理策略"""
        session_type = self._get_session_type(session_id)
        return self.strategies[session_type]

    async def add(self, session_id: str, message_data: dict, callback):
        async with self.lock:
            # 初始化会话缓冲区
            if session_id not in self.buffers:
                strategy = self._get_strategy(session_id)
                self.buffers[session_id] = {
                    "msgs": [],
                    "task": None,
                    "strategy": strategy,
                    "start_time": datetime.now()  # 记录批次开始时间
                }

            buffer = self.buffers[session_id]
            buffer["msgs"].append(message_data)

            # 如果任务已经存在，取消它
            if buffer["task"]:
                buffer["task"].cancel()

            # 检查是否达到最大批次大小
            if len(buffer["msgs"]) >= buffer["strategy"]["max_batch_size"]:
                # 立即处理批次
                msgs = buffer["msgs"]
                del self.buffers[session_id]
                asyncio.create_task(self._process_batch(session_id, msgs, callback))
                return

            # 检查是否超过最长等待时间
            elapsed_time = (datetime.now() - buffer["start_time"]).total_seconds()
            if elapsed_time >= buffer["strategy"]["max_wait_time"]:
                # 立即处理批次
                msgs = buffer["msgs"]
                del self.buffers[session_id]
                asyncio.create_task(self._process_batch(session_id, msgs, callback))
                return

            # 创建新的延迟处理任务
            buffer["task"] = asyncio.create_task(
                self._flush_timer(session_id, callback)
            )

    async def _flush_timer(self, session_id, callback):
        try:
            async with self.lock:
                if session_id not in self.buffers:
                    return
                # 获取会话的等待时间
                wait_time = self.buffers[session_id]["strategy"]["wait_time"]
            
            await asyncio.sleep(wait_time)
            
            async with self.lock:
                if session_id in self.buffers:
                    msgs = self.buffers[session_id]["msgs"]
                    del self.buffers[session_id]
                    await self._process_batch(session_id, msgs, callback)
        except asyncio.CancelledError:
            pass

    async def _process_batch(self, session_id: str, msgs: list, callback):
        """处理消息批次，根据会话类型进行优化"""
        session_type = self._get_session_type(session_id)
        if session_type == "group":
            # 群聊场景下的优化处理
            optimized_msgs = self._optimize_group_messages(msgs)
            await callback(session_id, optimized_msgs)
        else:
            # 私聊场景直接处理
            await callback(session_id, msgs)

    def _optimize_group_messages(self, messages: list) -> list:
        """
        群聊消息优化：
        1. 合并同一用户的连续消息
        2. 按用户分组处理不同用户的消息
        3. 保留消息的时间顺序
        """
        if not messages:
            return []

        # 按用户ID分组并保留时间顺序
        user_groups = {}
        # 按时间排序消息
        sorted_messages = sorted(messages, key=lambda x: x["time"])
        
        for msg in sorted_messages:
            user_id = msg["sender"]["user_id"]
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(msg)

        optimized = []
        
        # 对每个用户的消息进行合并
        for user_id, user_msgs in user_groups.items():
            if not user_msgs:
                continue
                
            # 合并同一用户的连续消息
            merged_user_messages = []
            current_batch = [user_msgs[0]]
            
            for msg in user_msgs[1:]:
                time_diff = msg["time"] - current_batch[-1]["time"]
                if time_diff < self.strategies["group"]["same_user_merge_window"]:
                    current_batch.append(msg)
                else:
                    # 合并当前批次
                    merged_msg = self._merge_messages(current_batch)
                    merged_user_messages.append(merged_msg)
                    current_batch = [msg]
            
            # 处理最后一个批次
            if current_batch:
                merged_msg = self._merge_messages(current_batch)
                merged_user_messages.append(merged_msg)
            
            optimized.extend(merged_user_messages)
        
        # 按时间顺序重新排序所有合并后的消息
        optimized.sort(key=lambda x: x["time"])
        
        return optimized

    def _merge_messages(self, messages: list) -> dict:
        """
        合并多条消息为一条
        """
        if not messages:
            return {}
        elif len(messages) == 1:
            return messages[0]
        
        # 创建合并后的消息
        merged_msg = messages[0].copy()
        merged_content = ""
        all_images = []
        
        for msg in messages:
            content, images, _ = parse_onebot_array_msg(msg.get("message", ""))
            if content:
                merged_content += content + " "
            all_images.extend(images)
        
        # 构建合并后的消息内容
        final_content = merged_content.strip()
        
        # 如果有图片，添加图片信息
        if all_images:
            final_content += " [图片]"
        
        merged_msg["message"] = final_content
        merged_msg["time"] = messages[0]["time"]  # 保留最早的时间戳
        merged_msg["is_merged"] = True  # 标记为合并消息
        merged_msg["merged_count"] = len(messages)  # 记录合并的消息数量
        
        return merged_msg


class QQBotManager:
    def __init__(self):
        self.connections: dict[str, WebSocket] = {}
        self.graph = build_graph()
        self.msg_buffer = MessageBuffer()
        self.api_futures: dict[str, asyncio.Future] = {}
        # 增加一个锁，防止同一个 Session 同时运行 Reactive 和 Proactive 导致混乱
        self.session_locks: dict[str, asyncio.Lock] = {}

    def get_session_lock(self, session_id: str):
        if session_id not in self.session_locks:
            self.session_locks[session_id] = asyncio.Lock()
        return self.session_locks[session_id]

    async def call_api(self, self_id: str, action: str, params: dict):
        if self_id not in self.connections: return None
        echo_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self.api_futures[echo_id] = future

        try:
            await self.connections[self_id].send_json({"action": action, "params": params, "echo": echo_id})
            return await asyncio.wait_for(future, timeout=5.0)
        except (asyncio.TimeoutError, Exception) as e:
            logger.error(f"❌ [API Error] {action}: {e}")
            if echo_id in self.api_futures: del self.api_futures[echo_id]
            return None

    async def send_msg(self, self_id: str, target_type: str, target_id: int, message: str):
        if self_id not in self.connections or not message: return
        payload = {
            "action": "send_msg",
            "params": {
                "message_type": target_type,
                "user_id": target_id if target_type == 'private' else None,
                "group_id": target_id if target_type == 'group' else None,
                "message": message
            }
        }
        try:
            await self.connections[self_id].send_json(payload)
            logger.info(f"🗣️ [Reply] -> {target_id}: {message[:50]}...")
        except Exception as e:
            logger.error(f"❌ [Send Error] {e}")

    async def resolve_mentions(self, text: str, self_id: str, group_id: str = "") -> str:
        # ... (保持原样，未修改) ...
        matches = re.findall(r"\[Mention:(\d+)\]", text)
        if not matches:
            return text
        unique_qqs = set(matches)
        for qq in unique_qqs:
            nickname = "未知用户"
            if group_id:
                info = await self.call_api(self_id, "get_group_member_info",
                                           {"group_id": int(group_id), "user_id": int(qq)})
                if info and "data" in info:
                    nickname = info["data"].get("card") or info["data"].get("nickname") or str(qq)
            if nickname == "未知用户" or nickname == str(qq):
                info = await self.call_api(self_id, "get_stranger_info", {"user_id": int(qq)})
                if info and "data" in info:
                    nickname = info["data"].get("nickname") or str(qq)
            pattern = f"\\[Mention:{qq}\\]"
            replacement = f"[@{nickname}](ID:{qq})"
            text = re.sub(pattern, replacement, text)
        return text

    # 修改 qq_server.py 文件中的 handle_graph_output 函数
    async def handle_graph_output(self, inputs: dict, self_id: str, msg_type: str, group_id: str, user_qq: str):
        """
        统一处理 Graph 的流式输出，无论是 Reactive 还是 Proactive
        """
        try:
            # 添加去重机制，避免重复发送相同的回复
            sent_messages = set()
            
            async for output in self.graph.astream(inputs):
                for node_name, node_val in output.items():
                    # 🚀 关键修改：监听 agent、proactive 和 saver 三个节点的输出
                    # saver 节点包含工具执行完成后的最终回复
                    if node_name in ["agent", "proactive", "saver"]:

                        # 检查 proactive 是否决定沉默
                        if node_name == "proactive" and node_val.get("next_step") == "silent":
                            continue

                        thought = node_val.get("internal_monologue")
                        if thought: logger.info(f"💭 [{node_name.upper()}] {thought}")

                        # 处理 emoji_reply 字段（直接发送表情包）
                        emoji_reply = node_val.get("emoji_reply")
                        if emoji_reply:
                            try:
                                target = int(group_id) if msg_type == "group" else int(user_qq)
                                # 使用file:///协议格式，确保OneBot客户端能正确识别本地文件路径
                                img_cq = f'[CQ:image,file=file:///{emoji_reply}]'
                                
                                # 检查是否已经发送过相同的表情包
                                if img_cq not in sent_messages:
                                    logger.info(f"📷 发送表情包回复: {emoji_reply}")
                                    await self.send_msg(self_id, msg_type, target, img_cq)
                                    sent_messages.add(img_cq)
                                    
                                    # 更新最后活跃时间
                                    session_key = f"{msg_type}_{target}"
                                    await session_manager.update_activity(session_key, msg_type, str(target), self_id)
                                continue
                            except Exception as e:
                                logger.error(f"❌ 处理表情包回复失败: {e}")

                        msgs = node_val.get("messages", [])
                        if msgs and isinstance(msgs[-1], AIMessage):
                            original_reply = msgs[-1].content
                            final_send_content = original_reply

                            # 群聊且是标准回复时，加个At (Proactive模式通常不At，更像随口一说，这里可以根据 node_name 区分)
                            if msg_type == "group" and user_qq and node_name == "agent":
                                final_send_content = f"[CQ:at,qq={user_qq}] {original_reply}"

                            # 群聊场景下的主动回复优化：
                            # 1. 通常不@某人，除非是直接针对特定内容的回复
                            # 2. 保持简洁，避免占屏
                            # 3. 根据内容判断是否需要更自然的表达
                            if msg_type == "group":
                                # 群聊主动回复：避免@，保持自然，融入群体
                                # 可以在内容前添加一些轻松的表情或语气词，增加自然感
                                if node_name == "proactive":
                                    # 主动发起的群聊回复，更加自然随意
                                    final_send_content = final_send_content
                                else:
                                    # 针对特定内容的回复，可以考虑@
                                    final_send_content = f"[CQ:at,qq={user_qq}] {original_reply}"

                            try:
                                # 处理回复中的表情包标记
                                final_content = final_send_content
                                
                                # 查找所有表情包标记 [表情: 哈希值]
                                import re
                                emoji_pattern = r'\[表情: (\w+)\]'
                                emoji_matches = re.findall(emoji_pattern, final_content)
                                
                                target = int(group_id) if msg_type == "group" else int(user_qq)
                                
                                # 检查是否已经发送过相同的回复
                                if final_content not in sent_messages:
                                    if emoji_matches:
                                        emoji_manager = get_emoji_manager()
                                        if emoji_manager:
                                            # 分离文字内容和表情包
                                            text_content = re.sub(emoji_pattern, '', final_content).strip()
                                            
                                            # 先发送文字消息（如果有）
                                            if text_content:
                                                if text_content not in sent_messages:
                                                    await self.send_msg(self_id, msg_type, target, text_content)
                                                    sent_messages.add(text_content)
                                            
                                            # 然后分开发送每个表情包
                                            for emoji_hash in emoji_matches:
                                                try:
                                                    emoji_info = emoji_manager.get_emoji(emoji_hash)
                                                    if emoji_info and emoji_info.file_path:
                                                        # 使用本地文件路径生成CQ码，避免base64数据过长
                                                        img_path = emoji_info.file_path
                                                        # 使用file:///协议格式，确保OneBot客户端能正确识别本地文件路径
                                                        img_cq = f'[CQ:image,file=file:///{img_path}]'
                                                        if img_cq not in sent_messages:
                                                            logger.info(f"📷 发送表情包: {emoji_hash} -> 文件路径: {img_path}")
                                                            await self.send_msg(self_id, msg_type, target, img_cq)
                                                            sent_messages.add(img_cq)
                                                except Exception as e:
                                                    logger.error(f"❌ 处理表情包失败: {e}")
                                    else:
                                        # 如果没有表情包，直接发送文字消息
                                        if final_content.strip():
                                            logger.info(f"🗣️ [Reply] -> {target}: {final_content[:50]}...")
                                            await self.send_msg(self_id, msg_type, target, final_content)
                                            sent_messages.add(final_content)

                                    # 更新最后活跃时间，防止 Proactive 刚说完又触发 Proactive
                                    session_key = f"{msg_type}_{target}"
                                    await session_manager.update_activity(session_key, msg_type, str(target), self_id)

                            except ValueError:
                                pass
        except Exception as e:
            logger.error(f"❌ [Graph Error] {e}", exc_info=True)

    # --- 核心逻辑 2: 用户消息入口 (Reactive) ---
    async def process_batch(self, session_id: str, raw_messages: list):
        if not raw_messages: return

        # 获取锁
        lock = self.get_session_lock(session_id)
        async with lock:
            first_msg = raw_messages[0]
            self_id = str(first_msg.get("self_id", "default"))
            msg_type = first_msg.get("message_type")
            group_id = str(first_msg.get("group_id", ""))
            sender = first_msg.get("sender", {})
            user_qq = str(sender.get("user_id"))
            user_nickname = sender.get("card") or sender.get("nickname") or user_qq

            # 更新活跃状态
            target_id = group_id if msg_type == "group" else user_qq
            await session_manager.update_activity(session_id, msg_type, target_id, self_id)

            # 记录原始消息数据，用于调试合并转发消息
            # 记录完整的原始消息结构
            logger.info(f"📦 [Raw Msg Full] {user_nickname}: {raw_messages}")
            # 记录消息类型和message字段
            for i, msg in enumerate(raw_messages):
                logger.info(f"📦 [Raw Msg {i}] Type: {msg.get('type')}, Message: {msg.get('message')}")
            
            # 解析消息批次
            full_text, image_urls, is_mentioned = await self._parse_message_batch(raw_messages, self_id, user_qq, user_nickname)

            logger.info(f"📦 [Msg] {user_nickname}: {full_text[:50]}... [URLs: {len(image_urls)}]")
            
            # 不再需要自动保存，因为已经在_parse_message_batch中处理

        # 构建输入参数
        inputs = await self._build_reactive_inputs(
            session_id=session_id,
            full_text=full_text,
            image_urls=image_urls,
            user_qq=user_qq,
            user_nickname=user_nickname,
            msg_type=msg_type,
            is_mentioned=is_mentioned
        )

        await self.handle_graph_output(inputs, self_id, msg_type, group_id, user_qq)

    async def _parse_message_batch(self, raw_messages: list, self_id: str, user_qq: str, user_nickname: str):
        """解析消息批次，提取文本、图片URL和是否被提及"""
        full_text = ""
        image_urls = []
        emoji_descriptions = []
        is_mentioned = False
        processed_reply_ids = set()

        # 收集所有需要处理的引用消息ID
        reply_ids_to_process = []
        # 处理转发消息ID列表
        forward_ids_to_process = []
        
        for item in raw_messages:
            # 解析单条消息
            t, imgs, reply_id = parse_onebot_array_msg(item.get("message", ""))
            
            # 检查是否包含转发消息
            if "[合并转发消息(ID:" in t:
                # 提取转发ID
                import re
                match = re.search(r'\[合并转发消息\(ID:(\d+)\)\]', t)
                if match:
                    forward_id = match.group(1)
                    forward_ids_to_process.append(forward_id)
            
            full_text += t + " "
            
            # 检查图片是否为表情包
            import io
            import base64
            from PIL import Image
            
            for img_url in imgs:
                try:
                    # 下载图片并判断是否为表情包
                    emoji_service = get_emoji_service()
                    if emoji_service:
                        # 使用emoji_service处理图片
                        result = await emoji_service.process_emoji(img_url, user_qq, user_nickname)
                        
                        if result.get("success", False):
                            # 将表情包情绪标签添加到文本中，而不是详细描述
                            emotions = result.get("emotions", ["未知"])
                            emoji_desc = "、".join(emotions)
                            full_text += f"【表情包: {emoji_desc}】\n"
                            emoji_descriptions.append(emoji_desc)
                        else:
                            # 如果不是表情包或处理失败，正常添加到图片列表
                            image_urls.append(img_url)
                    else:
                        # 如果EmojiService不可用，将图片添加到普通图片列表
                        logger.warning(f"⚠️ EmojiService不可用，将图片{img_url[:30]}...视为普通图片处理")
                        image_urls.append(img_url)
                except Exception as e:
                    logger.error(f"❌ 处理图片时发生错误: {e}")
                    # 发生错误时，仍将图片添加到列表
                    image_urls.append(img_url)

            # 收集引用消息ID
            if reply_id and reply_id not in processed_reply_ids:
                processed_reply_ids.add(reply_id)
                reply_ids_to_process.append(reply_id)

            # 检查是否被@
            raw_arr = item.get("message", [])
            if isinstance(raw_arr, list):
                for seg in raw_arr:
                    if seg.get("type") == "at" and str(seg.get("data", {}).get("qq", "")) == self_id:
                        is_mentioned = True
            
            # 检查是否包含forward类型的消息段，提取转发ID
            if isinstance(raw_arr, list):
                for seg in raw_arr:
                    if seg.get("type") == "forward":
                        forward_data = seg.get("data", {})
                        forward_id = forward_data.get("id") or forward_data.get("forward_id")
                        if forward_id:
                            forward_ids_to_process.append(str(forward_id))

        # 并行处理所有引用消息（性能优化）
        if reply_ids_to_process:
            # 创建所有API调用任务
            api_tasks = [self.call_api(self_id, "get_msg", {"message_id": rid}) for rid in reply_ids_to_process]
            # 并行执行所有API调用
            msg_data_list = await asyncio.gather(*api_tasks, return_exceptions=True)
            
            # 处理API调用结果
            for i, msg_data in enumerate(msg_data_list):
                if isinstance(msg_data, Exception):
                    logger.error(f"获取引用消息失败: {msg_data}")
                    continue
                
                if msg_data and "data" in msg_data:
                    ref_msg = msg_data["data"].get("message", "")
                    ref_text, ref_imgs, _ = parse_onebot_array_msg(ref_msg)
                    full_text += f"【引用: {ref_text}】\n"
                    
                    # 处理引用消息中的图片
                    for ref_img_url in ref_imgs:
                        image_urls.append(ref_img_url)

        # 循环处理所有转发消息（包括嵌套转发）
        processed_forward_ids = set()
        while forward_ids_to_process:
            # 去重转发ID，排除已处理的
            unique_forward_ids = [fid for fid in list(set(forward_ids_to_process)) if fid not in processed_forward_ids]
            if not unique_forward_ids:
                break
                
            logger.info(f"📦 [Forward] 处理{len(unique_forward_ids)}个转发消息ID: {unique_forward_ids}")
            
            # 创建所有API调用任务
            api_tasks = [self.call_api(self_id, "get_forward_msg", {"id": fid}) for fid in unique_forward_ids]
            # 并行执行所有API调用
            forward_data_list = await asyncio.gather(*api_tasks, return_exceptions=True)
            
            # 处理API调用结果
            for i, forward_data in enumerate(forward_data_list):
                forward_id = unique_forward_ids[i]
                processed_forward_ids.add(forward_id)
                
                if isinstance(forward_data, Exception):
                    logger.error(f"获取转发消息{forward_id}失败: {forward_data}")
                    continue
                
                if forward_data and "data" in forward_data:
                    # 解析转发消息内容
                    forward_msg_data = forward_data["data"]
                    
                    # 确保forward_msg_data是有效的字典
                    if not isinstance(forward_msg_data, dict):
                        logger.error(f"转发消息{forward_id}数据格式无效: {type(forward_msg_data)}")
                        continue
                    
                    # 保存完整的转发消息到数据库
                    try:
                        with SessionLocal() as db:
                            # 计算转发消息摘要
                            messages = forward_msg_data.get("messages", [])
                            msg_count = len(messages)
                            image_count = 0
                            summary_text = ""
                            
                            # 生成摘要
                            for i, msg_item in enumerate(messages[:3]):
                                sender_name = msg_item.get("sender", {}).get("nickname", msg_item.get("sender", {}).get("name", "未知用户"))
                                msg_content = msg_item.get("message", "")
                                msg_text, msg_imgs, _ = parse_onebot_array_msg(msg_content)
                                
                                if msg_text:
                                    if len(msg_text) > 30:
                                        msg_text = msg_text[:30] + "..."
                                    summary_text += f"{sender_name}: {msg_text}\n"
                                
                                if msg_imgs:
                                    image_count += len(msg_imgs)
                            
                            if msg_count > 3:
                                summary_text += f"... 共{msg_count}条消息，{image_count}张图片 ..."
                            
                            # 检查是否已存在
                            existing_forward = db.query(ForwardMessageModel).filter(ForwardMessageModel.forward_id == forward_id).first()
                            
                            if existing_forward:
                                # 更新现有记录
                                existing_forward.full_content = forward_msg_data
                                existing_forward.summary = summary_text
                                existing_forward.message_count = msg_count
                                existing_forward.image_count = image_count
                                db.commit()
                                logger.info(f"📦 [DB Update] Forward message {forward_id} updated in database")
                            else:
                                # 创建新记录
                                new_forward = ForwardMessageModel(
                                    forward_id=forward_id,
                                    full_content=forward_msg_data,
                                    summary=summary_text,
                                    message_count=msg_count,
                                    image_count=image_count
                                )
                                db.add(new_forward)
                                db.commit()
                                logger.info(f"📦 [DB Save] Forward message {forward_id} saved to database")
                    except Exception as e:
                        logger.error(f"❌ [DB Error] Failed to save forward message: {e}")
                    
                    # 添加转发消息的整体标题
                    full_text += f"\n【合并转发消息(ID:{forward_id})内容】\n"
                    
                    # 解析转发的每条消息
                    if "messages" in forward_msg_data:
                        messages = forward_msg_data["messages"]
                        total_images = 0
                        msg_count = len(messages)
                        
                        # 转发消息优化配置
                        MAX_FORWARD_MSG_DISPLAY = 10  # 最大显示消息数
                        TRUNCATE_MSG_LENGTH = 50     # 单条消息截断长度
                        
                        # 转发消息优化：只保留关键信息，减少Token消耗
                        # 对于超过MAX_FORWARD_MSG_DISPLAY条的转发消息，只保留前3条和后3条
                        display_messages = messages
                        if msg_count > MAX_FORWARD_MSG_DISPLAY:
                            display_messages = messages[:3] + messages[-3:]
                            
                        for i, msg_item in enumerate(display_messages):
                            sender_name = msg_item.get("sender", {}).get("nickname", msg_item.get("sender", {}).get("name", "未知用户"))
                            msg_content = msg_item.get("message", "")
                            
                            # 解析单条消息
                            msg_text, msg_imgs, _ = parse_onebot_array_msg(msg_content)
                            
                            # 限制单条消息文本长度
                            if msg_text:
                                if len(msg_text) > TRUNCATE_MSG_LENGTH:
                                    msg_text = msg_text[:TRUNCATE_MSG_LENGTH] + "..."
                                full_text += f"【{sender_name}】: {msg_text}\n"
                            
                            if msg_imgs:
                                # 保存转发消息中的图片URL
                                for img_url in msg_imgs:
                                    image_urls.append(img_url)
                                total_images += len(msg_imgs)
                                full_text += f" [{len(msg_imgs)}张图片]\n"
                        
                        # 如果是长消息，添加省略提示
                        if msg_count > 10:
                            omitted_count = msg_count - 6
                            full_text += f"... 省略了{omitted_count}条消息 ...\n"
                        
                        # 检查嵌套转发消息（需要检查所有消息，而不仅是显示的）
                        for msg_item in messages:
                            msg_content = msg_item.get("message", "")
                            if isinstance(msg_content, list):
                                for seg in msg_content:
                                    if isinstance(seg, dict) and seg.get("type") == "forward":
                                        nested_forward_id = seg.get("data", {}).get("id") or seg.get("data", {}).get("forward_id")
                                        if nested_forward_id and nested_forward_id not in processed_forward_ids:
                                            # 将嵌套转发ID添加到待处理列表
                                            forward_ids_to_process.append(str(nested_forward_id))
                                            logger.info(f"📦 [Nested Forward] 发现嵌套转发消息，ID: {nested_forward_id}")
                        
                        # 添加总图片数量信息
                        if total_images > 0:
                            logger.info(f"📦 [Forward] 转发消息{forward_id}中包含{total_images}张图片")
                    
                    logger.info(f"📦 [Forward] 成功解析转发消息{forward_id}，包含{len(forward_msg_data.get('messages', []))}条消息")
        
        # 移除已处理的转发ID
        forward_ids_to_process = [fid for fid in forward_ids_to_process if fid not in processed_forward_ids]

        # 清理文本
        full_text = full_text.strip()
        if not full_text and image_urls and not emoji_descriptions:
            full_text = "[图片]"

        return full_text, image_urls, is_mentioned

    async def _build_reactive_inputs(self, session_id: str, full_text: str, image_urls: list,
                                    user_qq: str, user_nickname: str, msg_type: str, is_mentioned: bool):
        """构建响应式模式的输入参数"""
        profile = await relation_db.get_user_profile(user_qq=user_qq, current_name=user_nickname)
        history_msgs, history_summary = await LocalHistoryManager.load_state(session_id)

        human_msg = HumanMessage(
            content=f"[{user_nickname}]: {full_text}",
            additional_kwargs={"image_urls": image_urls}
        )

        return {
            "messages": history_msgs + [human_msg],
            "conversation_summary": history_summary,
            "visual_input": None,
            "image_urls": image_urls,
            "session_id": session_id,
            "sender_qq": user_qq,
            "sender_name": user_nickname,
            "is_group": (msg_type == "group"),
            "is_mentioned": is_mentioned,
            "user_profile": profile.model_dump(),
            "should_reply": False,
            "is_proactive_mode": False,
            "global_emotion_snapshot": global_store.get_emotion_snapshot().model_dump(),
            "psychological_context": {},
            "current_image_artifact": None,
            "tool_call": {},
            "emotion": {"current_mood": "Calm"},
            "last_interaction_ts": time.time()
        }

    # --- 核心逻辑 3: 主动触发入口 (Proactive Trigger) ---
    async def run_proactive_check(self):
        """后台任务：遍历活跃会话，尝试主动触发"""
        logger.info("🕵️ [Proactive] Background task started.")
        while True:
            try:
                # 每 60 秒检查一次 (可以根据需要调整频率)
                await asyncio.sleep(60)

                # 获取活跃会话
                active_list = await session_manager.get_active_sessions()

                for session_id, data in active_list:
                    # 如果最近 5 分钟内有过交互，或者正在处理消息，先跳过，避免打扰
                    # Proactive Agent 内部也有 silence 判断，但这里做第一层过滤更省资源
                    silence_duration = time.time() - data["last_active"]

                    # 为群聊和私聊设置不同的触发条件
                    # 群聊场景：需要更长的沉默时间，避免过度活跃
                    # 私聊场景：可以更频繁地主动互动，增加亲密感
                    current_hour = time.localtime().tm_hour
                    current_weekday = time.localtime().tm_wday  # 0-6，0是周一
                    
                    if data["type"] == "group":
                        # 群聊沉默超过10分钟才触发，且只在活跃群里（最近2小时有互动）
                        # 增加：避免在深夜（23:00-07:00）打扰群聊
                        # 周末可以适当放宽时间限制，因为大家可能更活跃
                        is_weekend = current_weekday in [5, 6]  # 周六周日
                        if is_weekend:
                            # 周末可以稍微晚一点，早上8点到晚上23点
                            if (current_hour < 8 or current_hour >= 23):
                                continue
                        else:
                            # 工作日：早上7点到晚上22点
                            if (current_hour < 7 or current_hour >= 22):
                                continue
                        
                        if (silence_duration < 600 or 
                            (time.time() - data["last_active"]) > 7200):
                            continue
                    else:
                        # 私聊沉默超过一定时间才触发
                        # 增加：根据亲密度调整触发频率
                        # 高亲密度（>70）：5-120分钟
                        # 中亲密度（30-70）：15-360分钟
                        # 低亲密度（<30）：30-720分钟
                        profile = await relation_db.get_user_profile(user_qq=data["target_id"])
                        intimacy = profile.relationship.intimacy if profile else 50
                        
                        if intimacy > 70:
                            min_silence, max_silence = 300, 7200
                            # 超高亲密度可以适当放宽时间限制
                            if intimacy > 85:
                                if (current_hour < 5 or current_hour >= 23):
                                    continue
                            else:
                                if (current_hour < 6 or current_hour >= 23):
                                    continue
                        elif intimacy > 30:
                            min_silence, max_silence = 900, 21600
                            if (current_hour < 7 or current_hour >= 23):
                                continue
                        else:
                            min_silence, max_silence = 1800, 43200
                            if (current_hour < 8 or current_hour >= 22):
                                continue
                        
                        # 周末可以适当增加主动互动的频率
                        if current_weekday in [5, 6]:
                            min_silence = int(min_silence * 0.7)  # 周末触发更频繁
                        
                        if silence_duration < min_silence or silence_duration > max_silence:
                            continue

                    lock = self.get_session_lock(session_id)
                    if lock.locked(): continue  # 正在处理消息，跳过

                    async with lock:
                        logger.info(
                            f"⚡ [Proactive] Triggering check for {session_id} (Silence: {int(silence_duration)}s)")

                        # 加载状态
                        history_msgs, history_summary = await LocalHistoryManager.load_state(session_id)

                        # 对于群聊，target_id 是群号；对于私聊，是 QQ 号
                        target_id = data["target_id"]
                        msg_type = data["type"]
                        self_id = data["self_id"]

                        # 构造 Profile (主动模式下，主要交互对象设为 "Environment" 或群里的最后一个人)
                        # 这里简单处理，取最后一条消息的发送者 ID，如果没有则取 target_id
                        last_sender_id = target_id
                        last_sender_name = "User"

                        if history_msgs and isinstance(history_msgs[-1], HumanMessage):
                            # 尝试从历史消息内容里提取名字 (LocalHistory 存的是 string)
                            # 这里简化，直接使用 target_id
                            pass

                        profile = await relation_db.get_user_profile(user_qq=last_sender_id)

                        inputs = {
                            "messages": history_msgs,  # 不加新消息
                            "conversation_summary": history_summary,
                            "visual_input": None,
                            "image_urls": [],  # 这里可以对接 Monitor 的最新截图
                            "session_id": session_id,
                            "sender_qq": last_sender_id,
                            "sender_name": last_sender_name,
                            "is_group": (msg_type == "group"),
                            "is_mentioned": False,
                            "user_profile": profile.model_dump(),
                            "should_reply": False,

                            # 🚀 开启 Proactive Mode
                            "is_proactive_mode": True,

                            "global_emotion_snapshot": global_store.get_emotion_snapshot().model_dump(),
                            "psychological_context": {},
                            "current_image_artifact": None,
                            "tool_call": {},
                            "last_interaction_ts": data["last_active"]  # 传入真实的最后交互时间
                        }

                        # 传入 inputs, 触发 Proactive 流程
                        await self.handle_graph_output(inputs, self_id, msg_type, target_id, last_sender_id)

            except Exception as e:
                logger.error(f"❌ [Proactive Loop Error] {e}")
                await asyncio.sleep(60)  # 出错歇一会


bot_manager = QQBotManager()


# 定义主进程标识
import uvicorn.config
import os

# 在Uvicorn多进程模式下，只有主进程会有这个环境变量
is_main_process = os.environ.get('UVICORN_WORKER_ID') is None


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    from app.plugins.plugin_manager import plugin_manager
    from app.core.persona_manager import persona_vector_manager
    
    # 加载和初始化插件系统
    plugin_dir = os.path.join(os.path.dirname(__file__), "app", "plugins")
    loaded_count = plugin_manager.load_plugins_from_directory(plugin_dir)
    if loaded_count > 0:
        initialized_count = await plugin_manager.initialize_plugins()
        logger.info(f"✅ Plugins Initialized: {initialized_count}/{loaded_count}")
    else:
        logger.info("No plugins loaded")
    
    # 启动DreamCycle
    # DreamCycle内部有文件锁机制，确保只有一个进程能成功启动
    await dream_machine.start()
    
    # 初始化人设向量存储
    try:
        await persona_vector_manager.load_and_index_persona()
        logger.info("✅ Persona Vector Store Initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Persona Vector Store: {e}")

    # 🚀 启动主动任务循环（如果启用）
    proactive_task = None
    if enable_proactive:
        proactive_task = asyncio.create_task(bot_manager.run_proactive_check())
        logger.info("✅ Proactive Mode Enabled: Will check for conversation opportunities.")
    else:
        logger.info("ℹ️  Proactive Mode Disabled: Will only respond to user messages.")

    logger.info("✅ System Started (Reactive + Proactive + Persona Vector Store).")
    yield

    # 停止
    if proactive_task:
        proactive_task.cancel()
    
    # 关闭插件系统
    shutdown_count = await plugin_manager.shutdown_plugins()
    logger.info(f"✅ Plugins Shutdown: {shutdown_count}")
    
    if is_main_process:
        await dream_machine.stop()
    
    logger.info("🛑 System Shutdown.")

app = FastAPI(lifespan=lifespan)


# --- 全局错误处理 --- 

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器，捕获所有未处理的异常
    """
    logger.error(f"❌ [Global Error] Unhandled exception: {str(exc)} from {request.url}", exc_info=True)
    # 对于WebSocket连接，不需要返回HTTP响应
    if "websocket" in request.url.path:
        return
    # 对于HTTP请求，返回500错误
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后重试"}
    )


@app.websocket("/ws")
async def onebot_endpoint(websocket: WebSocket):
    # 1. 鉴权校验
    auth_header = websocket.headers.get("authorization", "")
    # 获取 Bearer 后面的 token
    token = auth_header.split(" ")[1] if " " in auth_header else auth_header
    
    # 从环境变量获取期望的 token
    expected_token = os.getenv("WEBSOCKET_AUTH_TOKEN", "")

    if expected_token and token != expected_token:
        logger.error(f"❌ WebSocket 鉴权失败...")
        await websocket.close(code=4003)
        return
    
    await websocket.accept()
    self_id = websocket.headers.get("X-Self-ID", "default")
    bot_manager.connections[self_id] = websocket
    logger.info(f"🚀 Linked to NapCat: {self_id}")

    try:
        while True:
            data = await websocket.receive_json()
            if "echo" in data:
                echo_id = data["echo"]
                if echo_id in bot_manager.api_futures:
                    bot_manager.api_futures[echo_id].set_result(data)
                    del bot_manager.api_futures[echo_id]
                continue

            if data.get("post_type") != "message": continue

            data["self_id"] = self_id
            msg_type = data.get("message_type")
            group_id = str(data.get("group_id", ""))
            user_id = str(data.get("user_id", ""))

            session_key = f"group_{group_id}" if msg_type == "group" else f"private_{user_id}"

            await bot_manager.msg_buffer.add(session_key, data, bot_manager.process_batch)

    except WebSocketDisconnect:
        if self_id in bot_manager.connections:
            del bot_manager.connections[self_id]
        logger.info(f"❌ Disconnected: {self_id}")


import os
import argparse

# 全局变量，控制是否启用主动回复功能
enable_proactive = True


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ProjectAlice QQ Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=6199, help="服务器端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数，默认根据CPU核心数自动调整")
    parser.add_argument("--no-proactive", action="store_true", help="关闭主动回复功能")
    args = parser.parse_args()
    
    # 设置全局变量
    enable_proactive = not args.no_proactive
    
    # 验证主机地址，确保使用有效的IP或0.0.0.0
    import socket
    valid_host = args.host
    try:
        # 尝试解析主机名或验证IP地址
        socket.getaddrinfo(valid_host, args.port)
    except socket.gaierror:
        logger.warning(f"⚠️  无效的主机地址: {valid_host}，将使用默认值 0.0.0.0")
        valid_host = "0.0.0.0"
    
    # 启用多进程模式，利用多核CPU提高性能
    if args.workers is None:
        import os
        args.workers = os.cpu_count()  # 默认使用所有CPU核心

    logger.info(f"🚀 启动ProjectAlice服务器 [多进程模式，工作进程数: {args.workers}]")
    logger.info(f"📡 监听地址: http://{valid_host}:{args.port}")
    
    # 启动Uvicorn服务器
    uvicorn.run(
        app,
        host=valid_host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
