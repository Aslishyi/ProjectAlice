# === Python代码文件: context_filter.py ===

import json
import re
import time
import logging
from datetime import datetime

# 配置日志
logger = logging.getLogger("ContextFilter")

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.core.state import AgentState
from app.core.config import config


# ==============================================================================
# 改进点 1: 鲁棒的 JSON 清洗与解析函数 (防御 API 脏数据注入)
# ==============================================================================
def _clean_and_parse_json(text: str) -> dict:
    """
    专门清洗 [system hint] 脏数据并解析 JSON
    """
    if not text: return None

    # 1. 暴力清洗 API 注入的系统提示
    # 移除类似 [system hint: ...] 的内容
    text = re.sub(r"\[system hint:.*?\]", "", text, flags=re.IGNORECASE)
    text = text.strip()

    # 2. 提取 Markdown 代码块中的 JSON
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 尝试直接寻找大括号
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start: end + 1]
        else:
            json_str = text

    # 3. 尝试解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 尝试修复常见的尾随逗号错误
            return json.loads(re.sub(r",\s*}", "}", json_str))
        except:
            return None


# ==============================================================================
# 改进点 2: Prompt 优化 (防止图片/表情包在 Filter 阶段被错误拦截)
# ==============================================================================
FILTER_PROMPT = """
You are the "Attention Filter" for an AI Assistant named Alice.
Your task is to analyze the latest message and decide if Alice should reply.

【Current Mode】
**{chat_mode}**

【Conversation Context (Last 3 messages)】
{context_history}

【Message Details】
- Sender: {user_name} (QQ: {user_qq})
- Is Mentioned (@Alice): {is_mentioned}
- Has Image Attached: {has_image}

【Decision Logic】

### STEP 1: UNIVERSAL BLOCKERS (Check these FIRST for BOTH Private & Group)
**Return FALSE (Do NOT Reply)** if ANY of these apply:
1.  **Conversation Closure**: The user explicitly ends the topic (e.g., "Ok", "Thanks", "Got it", "Good night", "Bye", "好的", "收到", "谢了", "睡了").
2.  **Meaningless Phatic**: The user sends ONLY generic emojis or simple reactions (e.g., "Haha", "666") with NO new info. 
    *   **EXCEPTION:** If `Has Image Attached` is TRUE, do **NOT** block it here. Pass it through for visual analysis.
3.  **Sentence Fragmentation**: The user is sending a split sentence. Wait for the full thought.
4.  **Double Sending**: Multiple messages in <1s. Only process the final one.
5.  **Topic Exhaustion**: Alice gave a final answer, and the user's reply adds nothing new.

### STEP 2: MODE-SPECIFIC RULES (If NO Blockers found)

#### SCENARIO A: PRIVATE CHAT (1-on-1)
**DEFAULT DECISION: TRUE (Reply)**.
If the message is NOT blocked by Step 1, Alice should reply to maintain the conversation flow.

#### SCENARIO B: GROUP CHAT
**DEFAULT DECISION: FALSE (Do NOT reply)**.
Alice should stay quiet to avoid spamming. **Return TRUE** ONLY if:
1.  **Explicit Mention**: `Is Mentioned` is true.
2.  **Name Reference**: The message content explicitly mentions "Alice" (e.g., "Alice，你今天下午做了什么呀？").
3.  **Explicit Question**: The user asks a clear question Alice is uniquely qualified to answer.
4.  **Active Engagement**: The user is replying *directly* to Alice's previous statement.

### Output Format
Return a JSON object with a "reasoning" field and a "should_reply" boolean.
{{"reasoning": "Private chat. Message is 'Ok', which hits Universal Blocker #1 (Closure).", "should_reply": false}}
"""

llm = ChatOpenAI(
    model=config.SMALL_MODEL,  # 建议统一使用 config.MODEL_NAME 或确认 config.MIMO_MODEL 存在
    temperature=0.0,
    api_key=config.SMALL_MODEL_API_KEY,  # 建议统一配置
    base_url=config.SMALL_MODEL_URL
)


def _extract_last_message_content(msgs: list) -> str:
    """
    从消息列表中提取最后一条消息的文本内容
    """
    if not msgs:
        return ""
    
    last_msg = msgs[-1]
    if isinstance(last_msg.content, list):
        return next((x['text'] for x in last_msg.content if x.get('type') == 'text'), "")
    else:
        return str(last_msg.content).strip()


def _check_has_image(state: AgentState, last_content: str) -> bool:
    """
    检查消息是否包含图片
    """
    image_urls = state.get("image_urls", [])
    return bool(image_urls or "[图片]" in last_content)


def _build_context_history(msgs: list) -> str:
    """
    构建上下文历史字符串
    """
    recent_msgs = msgs[-3:]
    history_str = ""
    
    for i, m in enumerate(recent_msgs):
        role = "AI(Alice)" if isinstance(m, (SystemMessage, dict)) or m.type == "ai" else "User"
        content = m.content
        
        if isinstance(content, list):
            text_part = next((x['text'] for x in content if x.get('type') == 'text'), "")
            if not text_part:
                text_part = "[Image/RichMedia]"
            content = text_part

        # 截断过长消息防止 Prompt 爆炸
        content_str = str(content)
        if len(content_str) > 100:
            content_str = content_str[:100] + "..."

        prefix = ">> [LATEST MSG] " if i == len(recent_msgs) - 1 else ""
        history_str += f"{prefix}[{role}]: {content_str}\n"
    
    return history_str


def _apply_heuristic_pre_filter(state: AgentState, last_content: str, has_img: bool) -> dict or None:
    """
    应用启发式预过滤规则
    """
    is_group = state.get("is_group", False)
    current_ts = time.time()
    
    # 如果没有图片，且文本长度极短且非问句
    if not has_img and len(last_content) < 2 and last_content not in ["?", "？", "hi", "Hi"]:
        # 私聊时，如果太短可能也需要回（比如"?"），这里主要针对群聊噪音
        if is_group:
            return {
                "should_reply": False,
                "filter_reason": "Message too short/Noise (Heuristic)",
                "last_interaction_ts": current_ts
            }
    
    return None


async def context_filter_node(state: AgentState):
    current_ts = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_group = state.get("is_group", False)
    is_mentioned = state.get("is_mentioned", False)

    # 1. 强规则：无论群聊私聊，被艾特必须回 (最高优先级)
    if is_mentioned:
        return {
            "should_reply": True,
            "filter_reason": "Directly mentioned (Hard Rule)",
            "last_interaction_ts": current_ts
        }

    msgs = state.get("messages", [])
    if not msgs:
        return {"should_reply": False, "filter_reason": "No messages"}

    # 提取最后一条消息的内容
    last_content = _extract_last_message_content(msgs)
    
    # 检查是否有图片
    has_img = _check_has_image(state, last_content)
    
    # 2. 应用启发式预过滤
    pre_filter_result = _apply_heuristic_pre_filter(state, last_content, has_img)
    if pre_filter_result:
        return pre_filter_result

    # 3. 构建上下文历史
    history_str = _build_context_history(msgs)
    chat_mode = "GROUP CHAT" if is_group else "PRIVATE CHAT (1-on-1)"

    try:
        # 4. 填充并调用LLM
        prompt = FILTER_PROMPT.format(
            chat_mode=chat_mode,
            context_history=history_str,
            user_name=state.get("sender_name", "User"),
            user_qq=state.get("sender_qq", "Unknown"),
            is_mentioned=str(is_mentioned),
            has_image=str(has_img)
        )

        resp = await llm.ainvoke([SystemMessage(content=prompt)])
        # 处理resp可能是字符串的情况
        if isinstance(resp, str):
            raw_content = resp.strip()
        else:
            raw_content = resp.content.strip()

        # 5. 使用增强的解析器解析结果
        data = _clean_and_parse_json(raw_content)

        if data:
            should = data.get("should_reply", False)
            reason = data.get("reasoning", data.get("reason", "No reason"))

            log_icon = "✅" if should else "🛑"
            mode_icon = "👥" if is_group else "👤"
            logger.info(f"[{ts}]{log_icon} [Filter] [{mode_icon}] Reply? {should} | Reason: {reason[:100]}")

            return {
                "should_reply": should,
                "filter_reason": reason,
                "last_interaction_ts": current_ts
            }
        else:
            logger.warning(f"[{ts}]⚠️ [Filter Warning] JSON Parse Failed. Raw: {raw_content[:50]}...")
            # 兜底：私聊回，群聊不回
            return {
                "should_reply": not is_group,
                "filter_reason": "Parse fail (Fallback)",
                "last_interaction_ts": current_ts
            }

    except Exception as e:
        logger.error(f"[{ts}]❌ [Filter Error] {e}. Fallback used.")
        return {
            "should_reply": not is_group,
            "filter_reason": f"Error fallback: {str(e)}",
            "last_interaction_ts": current_ts
        }
