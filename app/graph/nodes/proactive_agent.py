import json
import time
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.state import AgentState
from app.core.config import config
from app.core.global_store import global_store
from app.memory.relation_db import relation_db
from app.core.prompts import SOCIAL_VOLITION_PROMPT

# 建议使用逻辑能力较强的模型 (如 GPT-4o, Qwen-72B)
llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=0.8,  # 稍微高一点的温度，让主动发言更有灵性
    api_key=config.SILICONFLOW_API_KEY,
    base_url=config.SILICONFLOW_BASE_URL
)


def _get_time_period(dt: datetime) -> str:
    h = dt.hour
    if 0 <= h < 5: return "深夜/凌晨"
    if 5 <= h < 9: return "早晨"
    if 9 <= h < 12: return "上午"
    if 12 <= h < 14: return "中午"
    if 14 <= h < 18: return "下午"
    if 18 <= h < 23: return "晚上"
    return "深夜"


async def proactive_node(state: AgentState):
    """
    主动社交引擎 (Social Volition Engine) - 视觉增强版
    综合判断图片性质(实图/表情包)、最近文本消息、沉默时长和好感度。
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] --- [Proactive] Analyzing Social Context... ---")

    # 1. 获取基础上下文
    user_id = state.get("sender_qq", "unknown")
    user_display_name = state.get("sender_name", "User")

    # 2. 计算沉默时长
    last_ts = state.get("last_interaction_ts", time.time())
    now_ts = time.time()
    silence_seconds = now_ts - last_ts

    if silence_seconds < 60:
        silence_str = "刚刚 (用户可能还在输入中或刚发完消息)"
    elif silence_seconds < 3600:
        silence_str = f"{int(silence_seconds // 60)}分钟前"
    else:
        silence_str = f"{int(silence_seconds // 3600)}小时前"

    # 3. 提取最近的一条文本消息
    msgs = state.get("messages", [])
    last_text_content = "无最近文本消息"

    # 倒序查找最近一条 HumanMessage
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            content = m.content
            # 处理多模态列表的情况
            if isinstance(content, list):
                content = next((x['text'] for x in content if x['type'] == 'text'), "[图片/无文本]")
            last_text_content = content
            break

    # 4. 视觉信息 (基于 Perception 节点的分类结果)
    # image_data 仅在 visual_type='photo' 时由 perception 节点填充，表情包时不填充以节省内存
    image_data = state.get("current_image_artifact")
    visual_type = state.get("visual_type", "none")  # 'photo', 'sticker', 'icon', 'none'

    vision_desc = "无图片"
    if visual_type == "photo":
        vision_desc = "【用户发送了一张含有具体信息的图片/截图，请结合图片内容分析】"
    elif visual_type == "sticker":
        vision_desc = "【用户发送了一个表情包/Sticker，通常用于表达情绪或玩笑】"

    # 5. 深度关系数据
    profile = relation_db.get_user_profile(user_id)
    rel = profile.relationship

    # 6. 当前情绪与环境
    emotion = global_store.get_emotion_snapshot()
    now_dt = datetime.now()

    # 7. 最近话题摘要
    summary = state.get("conversation_summary", "无最近对话记录")

    # --- 构造 System Prompt ---
    # 这里通过 Prompt 注入更多即时信息
    prompt = SOCIAL_VOLITION_PROMPT.format(
        current_time=now_dt.strftime("%H:%M"),
        time_period=_get_time_period(now_dt),
        silence_duration=silence_str,
        mood=emotion.primary_emotion,
        stamina=emotion.stamina,
        user_name=user_display_name,
        intimacy=rel.intimacy,
        relation_tags=", ".join(rel.tags) if rel.tags else "无",
        relation_notes=rel.notes or "无",
        vision_desc=vision_desc,
        conversation_summary=summary[-400:] if summary else "无"
    )

    input_msgs = [SystemMessage(content=prompt)]

    # --- 核心修改：根据 visual_type 构建不同的输入 ---

    # 场景 A: 有意义的图片 (Photo) -> 发送 Base64 给 LLM
    if visual_type == "photo" and image_data:
        print("[{ts}] 🔍 [Proactive] Injecting IMAGE payload for analysis.")
        input_msgs.append(HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text",
             "text": f"用户刚发了这张图。上一句文本是: '{last_text_content}'。请判断图片内容是否重要？我该如何评论？"}
        ]))

    # 场景 B: 表情包 (Sticker) -> 拦截 Base64，仅发送文本提示
    elif visual_type == "sticker":
        print("[{ts}] 🎭 [Proactive] Handling STICKER (Skipping visual payload).")
        # 告诉 LLM 这是个表情包，不需要深度分析，只需要社交回应
        sticker_prompt = (
            f"[系统通知] 用户发送了一个表情包 (Sticker)。\n"
            f"上一句文本是: '{last_text_content}'。\n"
            f"无需分析图片内容（未上传）。请根据当前亲密度 ({rel.intimacy}) 决定是回复一个表情、简短吐槽还是保持沉默。"
        )
        input_msgs.append(HumanMessage(content=sticker_prompt))

    # 场景 C: 无图 (纯文本或静默)
    else:
        if silence_seconds < 120:
            # C1: 刚刚聊过天 -> 判断是否追评/接话
            user_input_prompt = (
                f"User just said: '{last_text_content}'. "
                f"Context Filter might have ignored it. "
                f"Based on our intimacy ({rel.intimacy}) and the text content, "
                f"should I voluntarily add a comment or follow up? (e.g., comfort, roast, or ask detail)"
            )
        else:
            # C2: 沉默很久 -> 判断是否破冰
            user_input_prompt = (
                f"User has been silent for {silence_str}. "
                f"Last known message was: '{last_text_content}'. "
                f"Should I initiate a NEW conversation based on the time of day or our relationship?"
            )
        input_msgs.append(HumanMessage(content=user_input_prompt))

    try:
        response = await llm.ainvoke(input_msgs)
        content = response.content.strip()

        # JSON 清洗
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")

        # 有些模型可能会输出 Markdown 格式，再加一层清洗
        content = content.strip('`')

        try:
            decision = json.loads(content)
        except:
            print(f"[{ts}] ⚠️ [Proactive] JSON Parse fail: {content[:30]}...")
            # 这里的 fallback 策略可以根据需求调整，默认保持沉默比较安全
            return {"next_step": "silent"}

        intent = decision.get("intent", "silent")
        reply_content = decision.get("content", "")
        reason = decision.get("reason", "")

        print(f"[{ts}] 🤖 [Proactive Decision] {intent.upper()} | Reason: {reason}")

        if intent == "silent" or not reply_content:
            return {"next_step": "silent"}

        # 决定说话 -> 消耗体力
        global_store.update_emotion(0, 0, stamina_delta=-3.0)

        ai_msg = AIMessage(content=reply_content)

        # 返回更新后的状态
        # 注意：这里使用 msgs (从 state 获取的列表) + 新消息
        # 具体是返回完整列表还是增量列表取决于你的 Graph Reducer 定义，这里保持原逻辑风格返回完整列表
        return {
            "messages": msgs + [ai_msg],
            "next_step": "speak",
            "internal_monologue": f"[Social Volition] Intent: {intent}, Reason: {reason}"
        }

    except Exception as e:
        print(f"[{ts}] ❌ [Proactive Error] {e}")
        return {"next_step": "silent"}
