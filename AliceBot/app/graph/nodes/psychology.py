import json
import re
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.core.state import AgentState
from app.core.config import config
from app.core.prompts import PSYCHOLOGY_ANALYSIS_PROMPT
from app.core.global_store import global_store
from app.memory.relation_db import relation_db

llm = ChatOpenAI(
    model=config.SMALL_MODEL,
    temperature=0.3,
    api_key=config.SMALL_MODEL_API_KEY,
    base_url=config.SMALL_MODEL_URL
)

# 配置日志
logger = logging.getLogger("PsychologyNode")


async def psychology_node(state: AgentState):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{ts}]--- [Psychology] Analyzing Subconscious... ---")

    # 1. 身份锚定：只认 QQ 号作为数据库主键
    user_id = state.get("sender_qq", "unknown_user")
    # 2. 称呼适配：Prompt 中使用当前昵称
    user_display_name = state.get("sender_name", "Stranger")

    msgs = state.get("messages", [])
    if not msgs: return {}

    last_msg = msgs[-1].content
    if isinstance(last_msg, list): last_msg = "[多模态图片/文件]"

    g_emotion = global_store.get_emotion_snapshot()

    # 3. 从 DB 获取关系 (Key 必须是 Unique ID)
    profile = await relation_db.get_user_profile(user_id)
    rel = profile.relationship

    # 4. 丰富关系描述
    def get_relation_desc(intimacy, familiarity, trust, interest_match):
        if intimacy < 20:
            return "讨厌的人"
        elif intimacy < 40:
            if trust < 30:
                return "不怎么信任的人"
            else:
                return "普通路人"
        elif intimacy < 60:
            if familiarity > 70:
                return "熟悉的朋友"
            elif trust > 70:
                return "值得信任的朋友"
            else:
                return "普通的朋友"
        elif intimacy < 80:
            if familiarity > 80 and trust > 80:
                return "亲密的朋友"
            elif interest_match > 80:
                return "志同道合的朋友"
            else:
                return "值得信赖的朋友"
        else:
            if familiarity > 90 and trust > 90:
                return "最亲密的朋友"
            else:
                return "非常要好的朋友"

    rel_desc = get_relation_desc(rel.intimacy, rel.familiarity, rel.trust, rel.interest_match)

    # 5. 构造 Prompt
    prompt = PSYCHOLOGY_ANALYSIS_PROMPT.format(
        current_mood=g_emotion.primary_emotion,
        valence=g_emotion.valence,
        arousal=g_emotion.arousal,
        stress=g_emotion.stress,
        fatigue=g_emotion.fatigue,
        user_name=user_display_name,
        intimacy=rel.intimacy,
        familiarity=rel.familiarity,
        trust=rel.trust,
        interest_match=rel.interest_match,
        relation_desc=rel_desc,
        user_input=last_msg
    )

    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        raw_content = response.content.strip()

        data = {}
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except Exception as e:
                logger.error(f"[{ts}]❌ [Psychology JSON Parse Error] {str(e)}")
                logger.error(f"[{ts}]❌ Raw content: {raw_content[:100]}...")
                pass

        if not data:
            logger.error(f"[{ts}]❌ [Psychology Parse Error] Raw: {raw_content[:30]}...")
            return {}

        # 5. 执行全局情绪更新
        global_store.update_emotion(
            valence_delta=data.get("valence_delta", 0),
            arousal_delta=data.get("arousal_delta", 0),
            stress_delta=data.get("stress_delta", 0),
            fatigue_delta=data.get("fatigue_delta", 0),
            new_primary=data.get("primary_emotion"),
            new_secondary=data.get("secondary_emotion")
        )

        # 6. 执行关系维度更新 (使用唯一 ID)
        relation_deltas = data.get("relation_deltas", {})
        if relation_deltas:
            updated_dimensions = relation_db.update_relationship_dimensions(user_id, relation_deltas)
            # 记录日志
            if updated_dimensions:
                log_msg = f"[{ts}]❤️ [Relation] {user_display_name}({user_id}):"
                for dim, new_value in updated_dimensions.items():
                    old_value = getattr(rel, dim, 50)
                    delta = new_value - old_value
                    log_msg += f" {dim}: {old_value} -> {new_value} (Delta: {delta})"
                logger.info(log_msg)
        else:
            # 兼容旧格式
            i_delta = data.get("intimacy_delta", 0)
            if i_delta != 0:
                new_intimacy = relation_db.update_intimacy(user_id, i_delta)
                logger.info(f"[{ts}]❤️ [Relation] {user_display_name}({user_id}): {rel.intimacy} -> {new_intimacy} (Delta: {i_delta})")

        # 7. 获取更新后的情绪和关系数据
        updated_emotion = global_store.get_emotion_snapshot()
        updated_profile = await relation_db.get_user_profile(user_id)
        updated_rel = updated_profile.relationship

        return {
            "psychological_context": {
                "internal_thought": data.get("internal_thought", "Thinking..."),
                "style_instruction": data.get("style_instruction", "Normal"),
                "primary_emotion": updated_emotion.primary_emotion,
                "secondary_emotion": updated_emotion.secondary_emotion,
                "current_intimacy": updated_rel.intimacy,
                "current_familiarity": updated_rel.familiarity,
                "current_trust": updated_rel.trust,
                "current_interest_match": updated_rel.interest_match
            },
            "global_emotion_snapshot": updated_emotion.model_dump()
        }

    except Exception as e:
        logger.error(f"[{ts}]❌ [Psychology Error] {e}")
        return {}
