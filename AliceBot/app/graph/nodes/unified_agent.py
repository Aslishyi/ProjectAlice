import json
import re
import time
import random
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI

# 配置日志
logger = logging.getLogger("UnifiedAgent")
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.state import AgentState
from app.core.config import config
from app.memory.vector_store import vector_db
from app.memory.relation_db import relation_db
from app.core.prompts import ALICE_CORE_PERSONA, AGENT_SYSTEM_PROMPT
from app.utils.cache import cached_llm_invoke, cached_user_info_get, cached_user_info_set

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=0.7,
    api_key=config.MODEL_API_KEY,
    base_url=config.MODEL_URL
)


def robust_json_parse(text: str) -> dict:
    """
    增强型 JSON 解析器 - 专门修复 API 注入的脏数据
    """
    if not text: return None

    # 🚀 [核心修复] 移除 API 强行注入的 system hint 垃圾信息
    text = re.sub(r"\[system hint:.*?\]", "", text, flags=re.IGNORECASE)
    text = text.strip()

    # 提取 Markdown JSON
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start: end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            fixed_text = re.sub(r",\s*}", "}", text)
            return json.loads(fixed_text)
        except:
            return None


async def agent_node(state: AgentState):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{ts}]--- [Alice Core] Processing... ---")

    msgs = state.get("messages", [])
    image_data = state.get("current_image_artifact")
    visual_type = state.get("visual_type", "none")

    # 提取最近一条消息文本
    last_human_content = ""
    if msgs:
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                content = m.content
                if isinstance(content, list):
                    content = next((x['text'] for x in content if x['type'] == 'text'), "")
                last_human_content = str(content).strip()
                break

    # =========================================================================
    # 🛡️ 第一道防线：短路拦截 (Short-Circuit)
    # =========================================================================
    if visual_type == "sticker":
        # 🚀 [核心修复 1] 增强清洗逻辑
        # 1. 移除可能存在的 [用户名]: 前缀 (非贪婪匹配)
        # 2. 移除 [图片], [表情] 占位符
        # 3. 移除空格

        # 临时变量，先去掉用户名开头
        # 匹配模式：行首 + [任意字符] + 冒号 + 可选空格
        temp_text = re.sub(r"^\[.*?\]:\s*", "", last_human_content)

        clean_text = temp_text.replace("[图片]", "").replace("[表情]", "").replace(" ", "").strip()

        logger.debug(f"[{ts}]🕵️ [Debug] Sticker Check -> Raw: '{last_human_content}' | Removed Prefix: '{temp_text}' | Final Cleaned: '{clean_text}'")

        if len(clean_text) < 2:
            logger.info(f"[{ts}] 🛑 [Alice Core] Detected PURE STICKER. Skipping LLM.")

            # 50% 概率回复表情
            if random.random() < 0.6:
                replies = ["🐶", "🐱", "💖", "💕", "💝", "🤗", "👻", "👽"]
                reply = random.choice(replies)
                logger.info(f"[{ts}]🎲 [Short-Circuit] Reply: {reply}")
                return {
                    "internal_monologue": "Sticker acknowledged.",
                    "messages": msgs + [AIMessage(content=reply)],
                    "last_interaction_ts": time.time(),
                    "next_step": "save"
                }
            else:
                logger.info(f"[{ts}] 🤐 [Short-Circuit] Silent.")
                return {
                    "internal_monologue": "Sticker ignored.",
                    "messages": msgs,
                    "last_interaction_ts": time.time(),
                    "next_step": "save"
                }

    # =========================================================================
    # 🧠 LLM 处理 (Photo 或 带有文字的 Sticker)
    # =========================================================================

    psych_ctx = state.get("psychological_context", {})
    real_user_id = state.get("sender_qq", "unknown")
    user_display_name = state.get("sender_name", "User")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # RAG 检索 (保持原逻辑)
    memory_context = ""
    try:
        # 只有清洗后的文本足够长才检索，避免用 "[图片]" 检索
        query_text = re.sub(r"^\[.*?\]:\s*", "", last_human_content).replace("[图片]", "").strip()
        if len(query_text) > 4:
            docs = await vector_db.search(query_text, k=3)
            if docs:
                logger.info(f"[{ts}] 📖 [RAG] Hit: {[d[:20] + '...' for d in docs]}")
                memory_context = f"【相关回忆】\n" + "\n".join(docs)
    except Exception as e:
        logger.error(f"[{ts}] [RAG Error] {e}")
        pass

    # 视觉摘要
    vision_summary_text = "无"
    if image_data and visual_type == "photo":
        vision_summary_text = "【视觉信号活跃：用户发了具体图片，见下方多模态输入】"
    elif visual_type == "sticker":
        vision_summary_text = "【视觉信号：用户发送了一个表情包/Sticker】"

    # 构造 Prompt
    format_instruction = """
    # 强制响应格式要求
    YOU MUST OUTPUT A VALID JSON OBJECT ONLY. NO OTHER TEXT OR EXPLANATION ALLOWED.
    YOU WILL BE PUNISHED IF YOU FAIL TO FOLLOW THIS INSTRUCTION.
    
    Response Format:
    {
      "monologue": "你的内部思考过程",
      "action": "reply",
      "args": "",
      "response": "要发送给用户的回复内容"
    }
    
    Example:
    {"monologue": "用户问我喜欢什么颜色，我应该回答蓝色", "action": "reply", "args": "", "response": "我喜欢蓝色"}
    """

    # 获取情绪和关系数据
    emotion_snapshot = state.get("global_emotion_snapshot", {})
    primary_emotion = psych_ctx.get("primary_emotion", emotion_snapshot.get("primary_emotion", "平淡"))
    secondary_emotion = psych_ctx.get("secondary_emotion", emotion_snapshot.get("secondary_emotion", ""))
    valence = emotion_snapshot.get("valence", 0.0)
    arousal = emotion_snapshot.get("arousal", 0.0)
    stress = emotion_snapshot.get("stress", 0.0)
    fatigue = emotion_snapshot.get("fatigue", 0.0)
    intimacy = psych_ctx.get("current_intimacy", 30)
    familiarity = psych_ctx.get("current_familiarity", 50)
    trust = psych_ctx.get("current_trust", 50)
    interest_match = psych_ctx.get("current_interest_match", 50)
    
    # 生成关系描述
    if intimacy < 20:
        relation_desc = "陌生人"
    elif intimacy < 40:
        relation_desc = "认识的人"
    elif intimacy < 60:
        if familiarity > 70:
            relation_desc = "熟悉的朋友"
        elif trust > 70:
            relation_desc = "值得信任的朋友"
        else:
            relation_desc = "普通的朋友"
    elif intimacy < 80:
        if familiarity > 80 and trust > 80:
            relation_desc = "亲密的朋友"
        elif interest_match > 80:
            relation_desc = "志同道合的朋友"
        else:
            relation_desc = "值得信赖的朋友"
    else:
        if familiarity > 90 and trust > 90:
            relation_desc = "最亲密的朋友"
        else:
            relation_desc = "非常要好的朋友"

    # 计算次要心情显示内容
    secondary_emotion_message = f" + 次要心情: {secondary_emotion}" if secondary_emotion else ""
    
    # 构造 Prompt
    final_system_prompt = AGENT_SYSTEM_PROMPT.format(
        core_persona=ALICE_CORE_PERSONA,
        time=now_str,
        current_user=f"{user_display_name} ({real_user_id})",
        vision_summary=vision_summary_text,
        primary_emotion=primary_emotion,
        secondary_emotion_message=secondary_emotion_message,
        valence=valence,
        arousal=arousal,
        stress=stress,
        fatigue=fatigue,
        internal_thought=psych_ctx.get("internal_thought", "思考中..."),
        style_instruction=psych_ctx.get("style_instruction", "保持日常语气"),
        intimacy=intimacy,
        familiarity=familiarity,
        trust=trust,
        interest_match=interest_match,
        relation_desc=relation_desc,
        memories=memory_context
    ) + "\n\n" + format_instruction

    input_messages = [SystemMessage(content=final_system_prompt)]
    if len(msgs) > 0:
        input_messages.extend(msgs[-10:])

    # 注入图片数据 (仅限 photo)
    if visual_type == "photo" and image_data:
        input_messages.append(HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text", "text": "（系统附言：这是用户发的图片，请结合回答。）"}
        ]))

    # 🚀 [核心修复 2] Sticker 兜底指令
    # 即使短路逻辑被绕过（比如用户说了"哈哈" + 表情包），也要防止 LLM 幻视分析图片
    if visual_type == "sticker":
        logger.info(f"[{ts}] 🎭 [Alice Core] Injecting STICKER SAFEGUARD.")
        safeguard = (
            "【系统强制指令】\n"
            "用户最后发送的是一个【表情包/Sticker】（代码中可能显示为'[图片]'）。\n"
            "1. 这是一个非信息性的表情符号，**绝对不要**询问'这是什么图片'或'图片里有什么'。\n"
            "2. 请将其视为一种情绪表达，仅对用户的文字内容（若有）进行回复，或回以简单互动。\n"
        )
        input_messages.append(SystemMessage(content=safeguard))

    # 调用 LLM
    parsed = {"action": "reply", "response": "..."}
    try:
        # 自动判断对话类型
        conversation_type = "group" if "group" in str(state.get("session_id", "")) else "private"
        
        response = await cached_llm_invoke(
            llm, 
            input_messages, 
            temperature=llm.temperature,
            conversation_type=conversation_type
        )
        # 处理response可能是字符串的情况
        if isinstance(response, str):
            content = response.strip()
        else:
            content = response.content.strip()

        parsed_result = robust_json_parse(content)

        if parsed_result:
            parsed = parsed_result
        else:
            logger.warning(f"[{ts}] ⚠️ [Agent JSON Fail] Raw: {content[:50]}...")
            if "{" not in content:
                parsed = {"monologue": "Raw Text", "action": "reply", "response": content}
    except Exception as e:
        logger.error(f"[{ts}]❌ [Agent LLM Error] {e}")

    # 构造返回
    ai_msg = AIMessage(content=parsed.get("response", "..."))
    # 根据是否需要调用工具设置next_step
    action = parsed.get("action", "reply")
    next_step = "tool" if action in ["web_search", "generate_image", "run_python_analysis"] else "save"
    
    return {
        "messages": msgs + [ai_msg],
        "next_step": next_step,
        "tool_call": {} if action == "reply" else {"name": action,
                                                   "args": parsed.get("args")}
    }
