import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from app.core.state import AgentState
from app.core.config import config
from app.memory.vector_store import vector_db
from app.memory.combined_memory import combined_memory
from app.utils.cache import cached_llm_invoke

# ... Prompt 保持不变，篇幅原因省略，请确保保留原文件中的 MEMORY_SYSTEM_PROMPT ...
MEMORY_SYSTEM_PROMPT = """
You are the Memory Manager for an AI Agent.
Your goal is to extract structured memory operations from the conversation.

**Current Mode:** {mode}
(Mode "INTERACTIVE": AI replied. Extract facts **only from the user's input**.)
(Mode "OBSERVATION": AI stayed silent. Extract facts **only from the user's input**.)

**Input Context:**
User: {user_name} (ID: {user_id})
User Input: "{user_input}"
AI Response: "{ai_output}"  (DO NOT extract facts from the AI's response!)

**Extraction Rules:**
1. **Unify Identity**: Always associate facts with User ID {user_id}.
2. **Fact vs. Noise**: 
   - "I bought a PS5" -> SAVE (Fact about the user).
   - "Lol" / "Weather is nice" -> IGNORE (trivial).
   - **NEVER** extract facts from the AI's response.
   - Focus only on what the user has said about themselves, their life, preferences, etc.
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
    model=config.SMALL_MODEL,
    temperature=0.0,
    api_key=config.SMALL_MODEL_API_KEY,
    base_url=config.SMALL_MODEL_URL
)

# 配置日志
logger = logging.getLogger("MemorySaver")


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
        
        # 将链式调用转换为直接调用，以便使用缓存
        formatted_prompt = prompt.format(
            mode=mode,
            user_id=real_user_id,  # 告诉 LLM ID
            user_name=user_nickname,  # 告诉 LLM 名字
            user_input=user_text,
            ai_output=ai_output
        )
        
        # 使用缓存的LLM调用
        resp = await cached_llm_invoke(
            llm, 
            [SystemMessage(content=formatted_prompt)], 
            temperature=llm.temperature,
            query_type="memory_extraction"
        )

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

            # 增强的重要性判断逻辑
            # 1. 检查是否包含明确的指令性词汇
            instruction_keywords = ["需要记住", "请记住", "重要", "关键", "一定要", "务必", "牢记"]
            has_instruction = any(keyword in content for keyword in instruction_keywords)
            has_instruction_in_input = any(keyword in user_text for keyword in instruction_keywords)
            
            # 2. 如果有明确指令，强制提高重要性
            if has_instruction or has_instruction_in_input:
                importance = max(importance, 5)  # 最高重要性
            
            # 3. 过滤不重要的信息
            # 在任何模式下，重要性低于2的信息都不存储
            if importance < 2:
                continue
                
            # 4. 在OBSERVATION模式下，需要更高的重要性
            if mode == "OBSERVATION" and importance < 4:
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

                logger.info(f"[{ts}] 🧠 [Memory] Saved ({mode}): {content} (ID: {real_user_id}, Importance: {importance})")

        if facts_to_add:
            await vector_db.add_texts(facts_to_add, metadatas_to_add)
            
            # 同时更新组合内存管理器
            try:
                await combined_memory.update_memory(user_text, ai_output, real_user_id, user_nickname)
                logger.info(f"[{ts}] 🧠 [CombinedMemory] Updated memories for user {real_user_id}")
            except Exception as e:
                logger.error(f"[{ts}] ❌ [CombinedMemory] Failed to update: {e}")

    except Exception as e:
        logger.error(f"[{ts}] ❌ [Memory Error] {e}")

    return {}
