from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from app.core.state import AgentState
from app.core.config import config
from app.memory.local_history import LocalHistoryManager

MAX_HISTORY_LEN = 15
PRUNE_COUNT = 10

SUMMARY_PROMPT = """
You are a Conversation Summarizer.
Update the running summary with new lines.

【Current Summary】
{current_summary}

【New Lines】
{new_lines}

Output ONLY the updated summary text.
"""

llm = ChatOpenAI(
    model=config.SMALL_MODEL,
    temperature=0.1,
    api_key=config.SMALL_MODEL_API_KEY,
    base_url=config.SMALL_MODEL_URL
)


async def summarizer_node(state: AgentState):
    messages = state.get("messages", [])
    current_summary = state.get("conversation_summary", "")

    # 获取 Session ID (用于隔离不同群/私聊的历史文件)
    # 如果上游未传 session_id，则回退到 sender_qq (仅兼容旧逻辑，建议上游必传)
    session_key = state.get("session_id") or state.get("sender_qq")

    # 1. 剪枝逻辑
    if len(messages) > MAX_HISTORY_LEN:
        to_prune = messages[:PRUNE_COUNT]
        remaining = messages[PRUNE_COUNT:]

        text_lines = []
        for m in to_prune:
            role = "User" if isinstance(m, HumanMessage) else "AI"
            content = m.content
            if isinstance(content, list): content = "[MultiModal/Image]"
            text_lines.append(f"{role}: {content}")

        input_text = "\n".join(text_lines)

        try:
            prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
            chain = prompt | llm
            response = await chain.ainvoke({
                "current_summary": current_summary if current_summary else "Start of log.",
                "new_lines": input_text
            })
            current_summary = response.content.strip()
            messages = remaining

        except Exception as e:
            print(f"❌ [Summarizer Error] {e}")

    # 2. 核心修复：调用异步保存方法，传入 session_key
    # 假设 LocalHistoryManager.save_state 签名支持 session_id 参数
    # 如果您的 LocalHistoryManager 是基于全局单例的，请务必修改它以接受 session_id 作为文件路径的一部分
    if session_key:
        await LocalHistoryManager.save_state(messages, current_summary, session_id=session_key)
    else:
        print("⚠️ [Summarizer] No session_id found, history might not persist correctly.")

    return {
        "messages": messages,
        "conversation_summary": current_summary
    }
