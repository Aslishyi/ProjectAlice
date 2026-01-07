# === Python代码文件: qq_server.py ===

import uvicorn
import asyncio
import uuid
import logging
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


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
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
    # ... (保持原样，未修改) ...
    def __init__(self, wait_time=1.5):
        self.wait_time = wait_time
        self.buffers = {}
        self.lock = asyncio.Lock()

    async def add(self, session_id: str, message_data: dict, callback):
        async with self.lock:
            if session_id not in self.buffers:
                self.buffers[session_id] = {"msgs": [], "task": None}
            if self.buffers[session_id]["task"]:
                self.buffers[session_id]["task"].cancel()

            self.buffers[session_id]["msgs"].append(message_data)
            self.buffers[session_id]["task"] = asyncio.create_task(
                self._flush_timer(session_id, callback)
            )

    async def _flush_timer(self, session_id, callback):
        try:
            await asyncio.sleep(self.wait_time)
            async with self.lock:
                if session_id in self.buffers:
                    msgs = self.buffers[session_id]["msgs"]
                    del self.buffers[session_id]
                    asyncio.create_task(callback(session_id, msgs))
        except asyncio.CancelledError:
            pass


class QQBotManager:
    def __init__(self):
        self.connections: dict[str, WebSocket] = {}
        self.graph = build_graph()
        self.msg_buffer = MessageBuffer(wait_time=1.5)
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

    # --- 核心逻辑 1: 处理 Graph 输出 (复用) ---
    async def handle_graph_output(self, inputs: dict, self_id: str, msg_type: str, group_id: str, user_qq: str):
        """
        统一处理 Graph 的流式输出，无论是 Reactive 还是 Proactive
        """
        try:
            async for output in self.graph.astream(inputs):
                for node_name, node_val in output.items():
                    # 🚀 关键修改：监听 agent 和 proactive 两个节点的输出
                    if node_name in ["agent", "proactive"]:

                        # 检查 proactive 是否决定沉默
                        if node_name == "proactive" and node_val.get("next_step") == "silent":
                            continue

                        thought = node_val.get("internal_monologue")
                        if thought: logger.info(f"💭 [{node_name.upper()}] {thought}")

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
                                target = int(group_id) if msg_type == "group" else int(user_qq)
                                await self.send_msg(self_id, msg_type, target, final_send_content)

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

            # 解析消息批次
            full_text, image_urls, is_mentioned = await self._parse_message_batch(raw_messages, self_id)

            logger.info(f"📦 [Msg] {user_nickname}: {full_text[:50]}... [URLs: {len(image_urls)}]")

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

    async def _parse_message_batch(self, raw_messages: list, self_id: str):
        """解析消息批次，提取文本、图片URL和是否被提及"""
        full_text = ""
        image_urls = []
        is_mentioned = False
        processed_reply_ids = set()

        for item in raw_messages:
            # 解析单条消息
            t, imgs, reply_id = parse_onebot_array_msg(item.get("message", ""))
            full_text += t + " "
            image_urls.extend(imgs)

            # 处理引用消息
            if reply_id and reply_id not in processed_reply_ids:
                processed_reply_ids.add(reply_id)
                msg_data = await self.call_api(self_id, "get_msg", {"message_id": reply_id})
                if msg_data and "data" in msg_data:
                    ref_msg = msg_data["data"].get("message", "")
                    ref_text, ref_imgs, _ = parse_onebot_array_msg(ref_msg)
                    full_text += f"【引用: {ref_text}】\n"
                    image_urls.extend(ref_imgs)

            # 检查是否被@
            raw_arr = item.get("message", [])
            if isinstance(raw_arr, list):
                for seg in raw_arr:
                    if seg.get("type") == "at" and str(seg.get("data", {}).get("qq", "")) == self_id:
                        is_mentioned = True

        # 清理文本
        full_text = full_text.strip()
        if not full_text and image_urls:
            full_text = "[图片]"

        return full_text, image_urls, is_mentioned

    async def _build_reactive_inputs(self, session_id: str, full_text: str, image_urls: list,
                                    user_qq: str, user_nickname: str, msg_type: str, is_mentioned: bool):
        """构建响应式模式的输入参数"""
        profile = relation_db.get_user_profile(user_qq=user_qq, current_name=user_nickname)
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
                        profile = relation_db.get_user_profile(user_qq=data["target_id"])
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

                        profile = relation_db.get_user_profile(user_qq=last_sender_id)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动后台服务
    await dream_machine.start()

    # 🚀 启动主动任务循环
    proactive_task = asyncio.create_task(bot_manager.run_proactive_check())

    logger.info("✅ System Started (Reactive + Proactive).")
    yield

    # 停止
    proactive_task.cancel()
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

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ProjectAlice QQ Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=6199, help="服务器端口")
    parser.add_argument("--workers", type=int, default=None, help="工作进程数，默认根据CPU核心数自动调整")
    args = parser.parse_args()
    
    # 如果未指定工作进程数，根据CPU核心数自动调整
    if args.workers is None:
        # 获取CPU核心数
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # 根据CPU核心数设置合适的工作进程数
        args.workers = min(cpu_count * 2, 8)  # 最多8个进程
    
    print(f"🚀 启动ProjectAlice服务器 [多进程模式: {args.workers}个进程]")
    print(f"📡 监听地址: http://{args.host}:{args.port}")
    
    # 启动Uvicorn服务器，使用多进程模式
    # 需要将应用程序作为导入字符串传递才能启用多进程
    uvicorn.run(
        "qq_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
