import json
import re
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.core.state import AgentState
from app.core.config import config
from app.core.prompts import PSYCHOLOGY_ANALYSIS_PROMPT
from app.core.global_store import global_store
from app.memory.relation_db import relation_db

llm = ChatOpenAI(
    model=config.SMALL_LLM_MODEL_NAME,
    temperature=0.3,
    api_key=config.SILICONFLOW_API_KEY,
    base_url=config.SILICONFLOW_BASE_URL
)


async def psychology_node(state: AgentState):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}]--- [Psychology] Analyzing Subconscious... ---")

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
    profile = relation_db.get_user_profile(user_id)
    rel = profile.relationship

    rel_desc = "普通路人"
    if rel.intimacy < 20:
        rel_desc = "讨厌的人"
    elif rel.intimacy >= 60:
        rel_desc = "值得信赖的朋友"

    # 4. 构造 Prompt
    prompt = PSYCHOLOGY_ANALYSIS_PROMPT.format(
        current_mood=g_emotion.primary_emotion,
        valence=g_emotion.valence,
        arousal=g_emotion.arousal,
        user_name=user_display_name,  # Prompt 里用名字
        intimacy=rel.intimacy,
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
            except:
                pass

        if not data:
            print(f"[{ts}]❌ [Psychology Parse Error] Raw: {raw_content[:30]}...")
            return {}

        # 5. 执行全局情绪更新
        global_store.update_emotion(
            valence_delta=data.get("valence_delta", 0),
            arousal_delta=data.get("arousal_delta", 0),
            new_primary=data.get("primary_emotion")
        )

        # 6. 执行好感度更新 (使用唯一 ID)
        i_delta = data.get("intimacy_delta", 0)
        new_intimacy = rel.intimacy
        if i_delta != 0:
            # 必须传入 sender_qq
            new_intimacy = relation_db.update_intimacy(user_id, i_delta)
            print(f"[{ts}]❤️ [Relation] {user_display_name}({user_id}): {rel.intimacy - i_delta} -> {new_intimacy} (Delta: {i_delta})")

        return {
            "psychological_context": {
                "internal_thought": data.get("internal_thought", "Thinking..."),
                "style_instruction": data.get("style_instruction", "Normal"),
                "current_intimacy": new_intimacy,
                "primary_emotion": data.get("primary_emotion", g_emotion.primary_emotion)
            },
            "global_emotion_snapshot": global_store.get_emotion_snapshot().model_dump()
        }

    except Exception as e:
        print(f"[{ts}]❌ [Psychology Error] {e}")
        return {}
