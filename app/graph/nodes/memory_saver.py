import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.state import AgentState
from app.core.config import config
from app.memory.vector_store import vector_db

# ... Prompt 保持不变，篇幅原因省略，请确保保留原文件中的 MEMORY_SYSTEM_PROMPT ...
MEMORY_SYSTEM_PROMPT = """
You are the Memory Manager for an AI Agent.
Your goal is to extract structured memory operations from the conversation.

**Current Mode:** {mode}
(Mode "INTERACTIVE": AI replied. Save normal interactions.)
(Mode "OBSERVATION": AI stayed silent. **ONLY** save critical facts about the user. IGNORE chit-chat, greetings, or fleeting comments.)

**Input Context:**
User: {user_name} (ID: {user_id})
User Input: "{user_input}"
AI Response: "{ai_output}"

**Extraction Rules:**
1. **Unify Identity**: Always associate facts with User ID {user_id}.
2. **Fact vs. Noise**: 
   - "I bought a PS5" -> SAVE (Fact).
   - "Lol" / "Weather is nice" -> IGNORE.
   - (In OBSERVATION mode): ONLY save if the user reveals permanent personal info or clear preferences.

**Output JSON Format:**
{{
  "operations": [
    {{
      "action": "add",
      "content": "User {user_name} bought a PS5.", 
      "category": "fact", 
      "importance": 4 
    }}
  ]
}}
If nothing worth saving, return {{"operations": []}}.
"""

llm = ChatOpenAI(
    model=config.SMALL_LLM_MODEL_NAME,
    temperature=0.0,
    api_key=config.SILICONFLOW_API_KEY,
    base_url=config.SILICONFLOW_BASE_URL
)


async def memory_saver_node(state: AgentState):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msgs = state.get("messages", [])
    if not msgs: return {}

    # 修改点：强制使用 sender_qq
    real_user_id = state.get("sender_qq", "unknown")
    user_nickname = state.get("sender_name", "User")

    last_msg = msgs[-1]
    ai_output = "N/A (AI remained silent)"
    mode = "OBSERVATION"
    user_text = ""

    if last_msg.type == 'ai':
        mode = "INTERACTIVE"
        ai_output = last_msg.content
        if len(msgs) >= 2:
            user_text = msgs[-2].content
        else:
            return {}
    else:
        mode = "OBSERVATION"
        user_text = last_msg.content

    if isinstance(user_text, list):
        user_text = next((x['text'] for x in user_text if x['type'] == 'text'), "[Image]")

    try:
        prompt = ChatPromptTemplate.from_template(MEMORY_SYSTEM_PROMPT)
        chain = prompt | llm

        resp = await chain.ainvoke({
            "mode": mode,
            "user_id": real_user_id, # 告诉 LLM ID
            "user_name": user_nickname, # 告诉 LLM 名字
            "user_input": user_text,
            "ai_output": ai_output
        })

        raw_content = resp.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_content)

        operations = data.get("operations", [])
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        facts_to_add = []
        metadatas_to_add = []

        for op in operations:
            action = op.get("action")
            content = op.get("content", "")
            category = op.get("category", "event")
            importance = op.get("importance", 1)

            if mode == "OBSERVATION" and importance < 3:
                continue

            if action == "add":
                final_content = f"User {user_nickname} (ID:{real_user_id}): {content}"
                full_text = f"[{current_time}] ({category.upper()}) {final_content}"

                facts_to_add.append(full_text)
                metadatas_to_add.append({
                    "source": "chat" if mode == "INTERACTIVE" else "observation",
                    "user_id": real_user_id, # 修改点：Metadata Key
                    "created_at": current_time,
                    "importance": importance,
                    "category": category
                })

                print(f"[{ts}] 🧠 [Memory] Saved ({mode}): {content} (ID: {real_user_id})")

        if facts_to_add:
            vector_db.add_texts(facts_to_add, metadatas_to_add)

    except Exception as e:
        print(f"[{ts}] ❌ [Memory Error] {e}")

    return {}
