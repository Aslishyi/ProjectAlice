import json
import time
import logging
from datetime import datetime
from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.state import AgentState
from app.core.config import config
from app.core.global_store import global_store
from app.memory.relation_db import relation_db
from app.core.prompts import SOCIAL_VOLITION_PROMPT
from app.utils.cache import cached_llm_invoke

# 配置日志
logger = logging.getLogger("ProactiveAgent")

# 建议使用逻辑能力较强的模型 (如 GPT-4o, Qwen-72B)
llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=0.8,  # 稍微高一点的温度，让主动发言更有灵性
    api_key=config.MODEL_API_KEY,
    base_url=config.MODEL_URL
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
    logger.info(f"[{ts}] --- [Proactive] Analyzing Social Context... ---")

    # 获取基础上下文
    context = _get_basic_context(state, ts)
    if not context:
        return {"next_step": "silent"}
    
    user_id, user_display_name, is_group, session_id, msgs, ts = context
    
    # 计算沉默时长
    silence_info = _calculate_silence_duration(state)
    silence_seconds, silence_str = silence_info
    
    # 处理历史消息
    history_str = _process_history_messages(msgs)
    
    # 处理视觉信息
    visual_info = _process_visual_information(state)
    image_data, visual_type, vision_desc = visual_info
    
    # 获取用户关系数据
    user_relation = await _get_user_relation_data(user_id)
    profile, rel, user_tags, user_birthday, user_hobbies = user_relation
    
    # 获取环境和情绪信息
    environment_info = _get_environment_info(state)
    emotion, now_dt, summary = environment_info
    
    # 构建个性化信息
    personalized_info = _build_personalized_info(is_group, user_hobbies, user_birthday, user_tags)
    
    # 构造系统提示
    prompt = _build_system_prompt(
        now_dt, silence_str, emotion, user_display_name, rel, vision_desc, 
        summary, is_group, personalized_info
    )
    
    # 构建输入消息
    input_msgs = await _build_input_messages(
        prompt, visual_type, image_data, history_str, is_group, silence_seconds, 
        silence_str, user_display_name, rel, ts
    )
    
    # 调用LLM并处理响应
    llm_response = await _process_llm_response(input_msgs, ts)
    if not llm_response:
        return {"next_step": "silent"}
    
    intent, reply_content, reason = llm_response
    
    # 过滤不合适的回复
    if is_group and intent != "silent":
        if not _filter_group_reply(reply_content, ts):
            return {"next_step": "silent"}
    
    # 个性化回复内容
    if intent != "silent" and reply_content:
        reply_content = _personalize_reply_content(
            reply_content, is_group, rel, user_hobbies
        )
    
    # 处理沉默意图
    if intent == "silent" or not reply_content:
        return {"next_step": "silent"}
    
    # 消耗体力并构建AI消息
    return _finalize_response(
        intent, reply_content, reason, is_group, msgs, rel, ts
    )


def _get_basic_context(state: AgentState, ts: str):
    """
    获取基础上下文信息
    """
    try:
        user_id = state.get("sender_qq", "unknown")
        user_display_name = state.get("sender_name", "User")
        is_group = state.get("is_group", False)
        session_id = state.get("session_id", "unknown")
        msgs = state.get("messages", [])
        return user_id, user_display_name, is_group, session_id, msgs, ts
    except Exception as e:
        logger.error(f"[{ts}] ❌ [Proactive] Failed to get basic context: {e}")
        return None


def _calculate_silence_duration(state: AgentState):
    """
    计算沉默时长
    """
    last_ts = state.get("last_interaction_ts", time.time())
    now_ts = time.time()
    silence_seconds = now_ts - last_ts

    if silence_seconds < 60:
        silence_str = "刚刚 (用户可能还在输入中或刚发完消息)"
    elif silence_seconds < 3600:
        silence_str = f"{int(silence_seconds // 60)}分钟前"
    else:
        silence_str = f"{int(silence_seconds // 3600)}小时前"
    
    return silence_seconds, silence_str


def _process_history_messages(msgs: List[Any]):
    """
    处理历史消息，生成历史字符串
    """
    history_str = ""
    for i, m in enumerate(msgs):
        role = "AI(Alice)" if isinstance(m, (SystemMessage, dict)) or getattr(m, 'type', '') == "ai" else "User"
        content = m.content
        if isinstance(content, list):
            text_part = next((x['text'] for x in content if x.get('type') == 'text'), "")
            if not text_part: text_part = "[Image/RichMedia]"
            content = text_part

        # 简单截断过长消息防止 Prompt 爆炸
        content_str = str(content)
        if len(content_str) > 100: content_str = content_str[:100] + "..."

        prefix = ">> [LATEST MSG] " if i == len(msgs) - 1 else ""
        history_str += f"{prefix}[{role}]: {content_str}\n"
    
    return history_str


def _process_visual_information(state: AgentState):
    """
    处理视觉信息
    """
    # image_data 仅在 visual_type='photo' 时由 perception 节点填充，表情包时不填充以节省内存
    image_data = state.get("current_image_artifact")
    visual_type = state.get("visual_type", "none")  # 'photo', 'sticker', 'icon', 'none'

    vision_desc = "无图片"
    if visual_type == "photo":
        vision_desc = "【用户发送了一张含有具体信息的图片/截图，请结合图片内容分析】"
    elif visual_type == "sticker":
        vision_desc = "【用户发送了一个表情包/Sticker，通常用于表达情绪或玩笑】"
    
    return image_data, visual_type, vision_desc


async def _get_user_relation_data(user_id: str):
    """
    获取用户关系数据
    """
    profile = await relation_db.get_user_profile(user_id)
    rel = profile.relationship
    # 获取用户的个性化信息
    user_tags = rel.tags if rel.tags else []
    user_birthday = getattr(profile, 'birthday', None)
    user_hobbies = getattr(profile, 'hobbies', [])
    
    return profile, rel, user_tags, user_birthday, user_hobbies


def _get_environment_info(state: AgentState):
    """
    获取环境和情绪信息
    """
    emotion = global_store.get_emotion_snapshot()
    now_dt = datetime.now()
    summary = state.get("conversation_summary", "无最近对话记录")
    
    return emotion, now_dt, summary


def _build_personalized_info(is_group: bool, user_hobbies: List[str], 
                            user_birthday: str, user_tags: List[str]):
    """
    构建个性化信息
    """
    if not is_group:
        # 私聊场景：添加用户个性化信息
        personalized_info = ""
        if user_hobbies:
            personalized_info += f"用户的兴趣爱好包括：{', '.join(user_hobbies)}。\n"
        if user_birthday:
            try:
                birthday = datetime.strptime(user_birthday, "%Y-%m-%d")
                today = datetime.now()
                days_until_birthday = (birthday.replace(year=today.year) - today).days
                if 0 <= days_until_birthday <= 7:
                    personalized_info += f"用户的生日快到了（{user_birthday}），可以适当表示关心。\n"
            except:
                pass
        if user_tags:
            personalized_info += f"用户的标签包括：{', '.join(user_tags)}。\n"
    else:
        personalized_info = ""
    
    return personalized_info


def _build_system_prompt(now_dt: datetime, silence_str: str, emotion: Any, 
                         user_display_name: str, rel: Any, vision_desc: str, 
                         summary: str, is_group: bool, personalized_info: str):
    """
    构建系统提示
    """
    return SOCIAL_VOLITION_PROMPT.format(
        current_time=now_dt.strftime("%H:%M"),
        time_period=_get_time_period(now_dt),
        silence_duration=silence_str,
        mood=emotion.primary_emotion,
        stamina=emotion.stamina,
        user_name=user_display_name,
        intimacy=rel.intimacy,
        familiarity=rel.familiarity,
        trust=rel.trust,
        interest_match=rel.interest_match,
        relation_tags=", ".join(rel.tags) if rel.tags else "无",
        relation_notes=rel.notes or "无",
        vision_desc=vision_desc,
        conversation_summary=summary[-400:] if summary else "无",
        chat_type="群聊" if is_group else "私聊",
        personalized_info=personalized_info
    )


async def _build_input_messages(prompt: str, visual_type: str, image_data: str, 
                              history_str: str, is_group: bool, silence_seconds: float, 
                              silence_str: str, user_display_name: str, rel: Any, ts: str):
    """
    构建输入消息
    """
    input_msgs = [SystemMessage(content=prompt)]

    # 场景 A: 有意义的图片 (Photo) -> 发送 Base64 给 LLM
    if visual_type == "photo" and image_data:
        logger.info(f"[{ts}] 🔍 [Proactive] Injecting IMAGE payload for analysis.")
        input_msgs.append(HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text",
             "text": f"用户刚发了这张图。历史信息是: '{history_str}'。请判断图片内容是否重要？我该如何评论？"}
        ]))

    # 场景 B: 表情包 (Sticker) -> 拦截 Base64，仅发送文本提示
    elif visual_type == "sticker":
        logger.info(f"[{ts}] 🎭 [Proactive] Handling STICKER (Skipping visual payload).")
        # 告诉 LLM 这是个表情包，不需要深度分析，只需要社交回应
        sticker_prompt = (
            f"[系统通知] 用户发送了一个表情包 (Sticker)。\n"
            f"历史信息是: '{history_str}'。\n"
            f"无需分析图片内容（未上传）。请根据当前亲密度 ({rel.intimacy}) 决定是回复一个表情、简短吐槽还是保持沉默。"
        )
        input_msgs.append(HumanMessage(content=sticker_prompt))

    # 场景 C: 无图 (纯文本或静默)
    else:
        # 根据群聊/私聊场景设置不同的回复策略
        user_input_prompt = _build_text_prompt(
            is_group, silence_seconds, silence_str, history_str, user_display_name, rel
        )
        input_msgs.append(HumanMessage(content=user_input_prompt))
    
    return input_msgs


def _build_text_prompt(is_group: bool, silence_seconds: float, 
                       silence_str: str, history_str: str, user_display_name: str, rel: Any):
    """
    构建文本提示
    """
    if is_group:
        # 群聊场景：更加谨慎，避免打扰，主要针对最近话题进行自然延伸
        if silence_seconds < 300:
            # C1: 群聊刚刚有活动 -> 可以对最近的话题进行补充或评论，但不要过于频繁
            return (
                f"This is a group chat environment. "
                f"History Conversation Context: '{history_str}'. "
                f"Should I naturally join the conversation with a relevant comment or observation? "
                f"Keep it brief, friendly, and avoid dominating the discussion."
            )
        else:
            # C2: 群聊沉默较久 -> 可以发起轻松话题，但不要显得突兀
            return (
                f"This is a group chat environment that's been quiet for {silence_str}. "
                f"History Conversation Context was: '{history_str}'. "
                f"Would a light, friendly comment or question be appropriate to re-engage the group? "
                f"Avoid being pushy or too personal."
            )
    else:
        # 私聊场景：更加亲密和个性化
        if silence_seconds < 120:
            # C1: 刚刚聊过天 -> 判断是否追评/接话，增加亲密感
            return (
                f"This is a private chat with {user_display_name}. "
                f"History Conversation Context: '{history_str}'. "
                f"Based on our intimacy ({rel.intimacy}) and the text content, "
                f"should I voluntarily add a warm comment, follow up, or ask a personal question?"
            )
        else:
            # C2: 沉默很久 -> 破冰，更加个性化
            return (
                f"This is a private chat with {user_display_name} that's been quiet for {silence_str}. "
                f"Our relationship intimacy is {rel.intimacy}. "
                f"History Conversation Context was: '{history_str}'. "
                f"Should I initiate a new conversation with a warm, personal message? "
                f"Consider our relationship, shared topics, and the time of day."
            )


async def _process_llm_response(input_msgs: List[Any], ts: str):
    """
    调用LLM并处理响应
    """
    try:
        response = await cached_llm_invoke(llm, input_msgs, temperature=llm.temperature)
        content = response.content.strip()

        # JSON 清洗
        if "```json" in content:
            content = content.replace("```json", "").replace("```", "")

        # 有些模型可能会输出 Markdown 格式，再加一层清洗
        content = content.strip('`')

        try:
            decision = json.loads(content)
        except:
            logger.warning(f"[{ts}] ⚠️ [Proactive] JSON Parse fail: {content[:30]}...")
            # 这里的 fallback 策略可以根据需求调整，默认保持沉默比较安全
            return None

        intent = decision.get("intent", "silent")
        reply_content = decision.get("content", "")
        reason = decision.get("reason", "")

        logger.info(f"[{ts}] 🤖 [Proactive Decision] {intent.upper()} | Reason: {reason}")
        return intent, reply_content, reason
        
    except Exception as e:
        logger.error(f"[{ts}] ❌ [Proactive Error] {e}")
        return None


def _filter_group_reply(reply_content: str, ts: str):
    """
    过滤群聊中的不合适回复
    """
    # 检查群聊回复是否合适
    lower_content = reply_content.lower()
    # 避免在群聊中询问过于私人的问题
    private_questions = ["你最近怎么样", "你在干什么", "你的隐私", "你家里", "你的感情", "你工资", "你年龄", "你对象"]
    if any(q in lower_content for q in private_questions):
        logger.warning(f"[{ts}] ⚠️ [Proactive] Filtered private content in group chat: {reply_content[:30]}...")
        return False
    # 避免在群聊中使用过于亲密的称呼
    intimate_terms = ["亲爱的", "宝贝", "老公", "老婆", "哥哥", "姐姐", "弟弟", "妹妹"]
    if any(term in reply_content for term in intimate_terms):
        logger.warning(f"[{ts}] ⚠️ [Proactive] Filtered intimate term in group chat: {reply_content[:30]}...")
        return False
    # 群聊回复保持简洁
    if len(reply_content) > 100:
        reply_content = reply_content[:100] + "..."
    
    return True


def _personalize_reply_content(reply_content: str, is_group: bool, 
                               rel: Any, user_hobbies: List[str]):
    """
    个性化回复内容
    """
    import random
    
    # 为不同场景添加语气词或表情，增加自然感
    if is_group:
        # 群聊场景：低调、平淡的语气，避免太突兀
        # 使用符合Alice云淡风轻性格的开头和结尾
        group_intros = ["", "对了，", "话说，", "突然想到，", "其实吧，", "我觉得", "话说回来，", "之前看到", "哎，", "那个", "刚才", "突然发现", "哦对了，", "话说那个", "突然想问"]
        group_endings = ["", "~", "", "", "🤔", "", "", "~", "~"]
        
        # 随机选择开头和结尾
        intro = random.choice(group_intros)
        ending = random.choice(group_endings)
        
        # 群聊场景：可以在内容中间添加一些停顿或语气词，增加自然感
        if len(reply_content) > 20:
            # 在中间位置插入一个自然的停顿或语气词
            middle_pos = len(reply_content) // 2
            natural_pauses = ["", "，", "，其实", "，话说", "，对吧", "，我觉得", "，你们看"]
            pause = random.choice(natural_pauses)
            if pause:  # 避免空字符串
                reply_content = reply_content[:middle_pos] + pause + reply_content[middle_pos:]
        
        return f"{intro}{reply_content}{ending}"
    else:
        # 私聊场景：温和、礼貌的语气，符合Alice云淡风轻的性格
        # 根据亲密度调整语气，但保持自然不夸张
        private_intros = ["", "哎，", "对了，", "你知道吗？", "突然想问你，", "话说回来，", "其实吧，", "我觉得", "刚才想到", "最近", "之前", "那个", "哎对了，", "突然发现", "话说", "其实"]
        
        # 根据亲密度选择不同的语气，避免过于亲密或夸张
        if rel.intimacy > 85:
            # 超高亲密度：稍微亲密但不夸张的表达
            intimate_intros = ["那个，", "哎，", "对了，", "话说，"]
            private_endings = ["", "~", "", "~", "~"]
        elif rel.intimacy > 70:
            # 高亲密度：友好但保持距离的表达
            intimate_intros = ["那个，", "哎，", "对了，"]
            private_endings = ["", "~", "", "~"]
        elif rel.intimacy > 50:
            # 中等亲密度：普通友好的表达
            intimate_intros = ["那个，", "哎，"]
            private_endings = ["", "~", "", ""]
        else:
            # 低亲密度：礼貌、平淡的表达
            intimate_intros = []
            private_endings = ["", "~", "", ""]
        
        # 随机选择开头和结尾
        available_intros = private_intros + intimate_intros
        intro = random.choice(available_intros)
        ending = random.choice(private_endings)
        
        # 私聊场景：可以添加更多个性化的元素，比如用户的兴趣爱好
        # 如果用户有明确的兴趣爱好，可以在回复中自然提及
        if user_hobbies and len(user_hobbies) > 0 and random.random() > 0.5:
            # 随机选择一个用户的兴趣爱好
            hobby = random.choice(user_hobbies)
            # 添加一些与兴趣相关的互动内容，保持平淡自然的语气
            hobby_related_phrases = [
                f"对了，你之前说喜欢{hobby}...",
                f"说到这个，突然想到你喜欢{hobby}...",
                f"哎，我记得你喜欢{hobby}...",
                f"对了，关于{hobby}...",
                f"突然想起你喜欢{hobby}...",
            ]
            # 随机选择一个相关短语，添加到回复内容中
            if random.random() > 0.3:
                # 在开头添加
                intro += random.choice(hobby_related_phrases)
            else:
                # 在内容中间添加
                if len(reply_content) > 20:
                    middle_pos = len(reply_content) // 2
                    reply_content = reply_content[:middle_pos] + f" {random.choice(hobby_related_phrases)} " + reply_content[middle_pos:]
        
        return f"{intro}{reply_content}{ending}"


def _finalize_response(intent: str, reply_content: str, reason: str, 
                       is_group: bool, msgs: List[Any], rel: Any, ts: str):
    """
    最终处理并返回响应
    """
    # 决定说话 -> 消耗体力
    # 群聊和私聊消耗不同的体力值
    stamina_cost = -2.0 if is_group else -3.0  # 群聊消耗较少体力
    global_store.update_emotion(0, 0, stamina_delta=stamina_cost)

    ai_msg = AIMessage(content=reply_content)

    # 返回更新后的状态
    # 注意：这里使用 msgs (从 state 获取的列表) + 新消息
    # 具体是返回完整列表还是增量列表取决于你的 Graph Reducer 定义，这里保持原逻辑风格返回完整列表
    return {
        "messages": msgs + [ai_msg],
        "next_step": "speak",
        "internal_monologue": f"[Social Volition] Intent: {intent}, Reason: {reason}, ChatType: {'Group' if is_group else 'Private'}"
    }
