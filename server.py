import uvicorn
import asyncio
import base64
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from langchain_core.messages import HumanMessage, AIMessage

from app.core.global_store import global_store
from app.graph.graph_builder import build_graph
from app.memory.relation_db import relation_db
from app.memory.local_history import LocalHistoryManager
from app.background.dream import dream_machine
from app.utils.qq_utils import parse_onebot_array_msg  # 使用新的解析函数

app = FastAPI()

# 对应 NapCat 配置中的 token
EXPECTED_TOKEN = "lo[-+]rSg(l?L,cK"


class QQBotManager:
    def __init__(self):
        self.connections: dict[str, WebSocket] = {}
        self.graph = build_graph()
        self.processing_lock = asyncio.Lock()

    async def send_msg(self, self_id: str, target_type: str, target_id: int, message: str):
        if self_id not in self.connections: return
        payload = {
            "action": "send_msg",
            "params": {
                "message_type": target_type,
                "user_id": target_id if target_type == 'private' else None,
                "group_id": target_id if target_type == 'group' else None,
                "message": message  # 发送时 OneBot 支持直接传字符串
            }
        }
        await self.connections[self_id].send_json(payload)


bot_manager = QQBotManager()


@app.on_event("startup")
async def startup():
    await dream_machine.start()


# 路径必须改为 /ws 以匹配 NapCat 配置
@app.websocket("/ws")
async def onebot_endpoint(websocket: WebSocket):
    # 1. 鉴权校验
    auth_header = websocket.headers.get("authorization", "")
    # 获取 Bearer 后面的 token
    token = auth_header.split(" ")[1] if " " in auth_header else auth_header

    if token != EXPECTED_TOKEN:
        print(f"❌ 鉴权失败，收到 Token: {token}")
        await websocket.close(code=4003)
        return

    await websocket.accept()
    self_id = websocket.headers.get("X-Self-ID", "default")
    bot_manager.connections[self_id] = websocket
    print(f"🚀 NapCatQQ 已连接 (Port 6199): {self_id}")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("post_type") != "message":
                continue

            # 2. 提取信息
            user_id = str(data.get("user_id"))
            group_id = data.get("group_id")
            msg_array = data.get("message")  # 此时是一个 List
            msg_type = data.get("message_type")

            # 3. 解析消息 (处理 Array 格式)
            text_content, images = parse_onebot_array_msg(msg_array)
            print(f"📩 [{msg_type}] {user_id}: {text_content} (+{len(images)}张图)")

            # 4. 运行智能体逻辑
            async with bot_manager.processing_lock:
                history_state = LocalHistoryManager.load_state()
                user_tag = f"QQ_{user_id}"

                # 视觉处理
                visual_payload = None
                if images:
                    async with httpx.AsyncClient() as client:
                        try:
                            resp = await client.get(images[0], timeout=5.0)
                            visual_payload = base64.b64encode(resp.content).decode()
                        except:
                            print("⚠️ 图片下载失败")

                inputs = {
                    "messages": history_state["messages"] + [HumanMessage(content=f"[{user_tag}]: {text_content}")],
                    "conversation_summary": history_state["conversation_summary"],
                    "visual_input": visual_payload,
                    "current_user_id": user_tag,
                    "user_profile": relation_db.get_user_profile(user_tag).model_dump(),
                    "is_proactive_mode": False,
                    "global_emotion_snapshot": global_store.get_emotion_snapshot().model_dump(),
                    "psychological_context": {},
                    "current_image_artifact": None,
                    "tool_call": {}
                }

                async for output in bot_manager.graph.astream(inputs):
                    for node_name, node_val in output.items():
                        if node_name == "agent":
                            msgs = node_val.get("messages", [])
                            if msgs and isinstance(msgs[-1], AIMessage):
                                reply = msgs[-1].content
                                target_id = group_id if msg_type == "group" else int(user_id)
                                await bot_manager.send_msg(self_id, msg_type, target_id, reply)

                                LocalHistoryManager.save_state(
                                    node_val.get("messages", []),
                                    node_val.get("conversation_summary", history_state["conversation_summary"])
                                )

    except WebSocketDisconnect:
        if self_id in bot_manager.connections:
            del bot_manager.connections[self_id]
        print(f"❌ NapCatQQ 断开连接")


if __name__ == "__main__":
    # 强制监听 6199 端口
    uvicorn.run(app, host="0.0.0.0", port=6199)