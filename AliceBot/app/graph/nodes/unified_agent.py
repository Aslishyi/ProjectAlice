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
from app.plugins.emoji_plugin.emoji_service import get_emoji_service

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

    # 检查是否有短路回复信息
    short_circuit_emoji = state.get("short_circuit_emoji")
    short_circuit_text = state.get("short_circuit_text")
    
    # 处理短路回复表情包
    if short_circuit_emoji:
        logger.info(f"[{ts}]🚀 [Alice Core] 收到短路回复指令，直接回复表情包")
        return {
            "internal_monologue": "Short circuit: reply with emoji",
            "messages": state.get("messages", []) + [AIMessage(content=""), AIMessage(content=f"[CQ:image,file=file:///{short_circuit_emoji}]")],
            "last_interaction_ts": time.time(),
            "next_step": "save",
            "emoji_reply": short_circuit_emoji
        }
    
    # 处理短路回复默认表情符号
    if short_circuit_text:
        logger.info(f"[{ts}]🚀 [Alice Core] 收到短路回复指令，直接回复表情符号")
        return {
            "internal_monologue": "Short circuit: reply with emoji",
            "messages": state.get("messages", []) + [AIMessage(content=short_circuit_text)],
            "last_interaction_ts": time.time(),
            "next_step": "save"
        }

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
    # 如果已经有短路回复指令，则跳过内部短路拦截逻辑
    if visual_type == "sticker" and not (short_circuit_emoji or short_circuit_text):
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

            # 使用用户存储的表情包回复
            if random.random() < 0.6:
                try:
                    emoji_service = get_emoji_service()
                    if emoji_service:
                        # 使用emoji_service选择匹配的表情包
                        context = {
                            "last_message": last_human_content,
                            "message_history": msgs[-5:]
                        }
                        selected_emojis = emoji_service.get_emoji_for_context(context, count=1)
                        if selected_emojis:
                            selected_emoji = selected_emojis[0]
                            logger.info(f"[{ts}]🎲 [Short-Circuit] Reply with saved emoji: {selected_emoji.emoji_hash}")
                            return {
                                "internal_monologue": "Sticker acknowledged with saved emoji.",
                                "messages": msgs + [AIMessage(content=""), AIMessage(content=f"[CQ:image,file=file:///{selected_emoji.file_path}]")],
                                "last_interaction_ts": time.time(),
                                "next_step": "save",
                                "emoji_reply": selected_emoji.file_path
                            }
                except Exception as e:
                    logger.error(f"[{ts}]❌ [Emoji Reply Error] {e}")
                
                # 如果没有可用的表情包，使用默认表情符号
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

    # 智能记忆检索 (替换传统RAG)
    memory_context = ""
    try:
        # 清洗文本，移除表情包描述和其他无关信息
        query_text = re.sub(r"^\[.*?\]:\s*", "", last_human_content)
        query_text = re.sub(r"【表情包:.*?】", "", query_text)
        query_text = query_text.replace("[图片]", "").strip()
        if len(query_text) > 4:
            from app.memory.combined_memory import CombinedMemoryManager
            
            # 初始化记忆管理器
            memory_manager = CombinedMemoryManager()
            
            # 构建聊天历史字符串
            chat_history_str = ""
            for msg in msgs[-5:]:  # 使用最近5条消息作为上下文
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, str):
                        role = "AI" if hasattr(msg, 'type') and msg.type == "ai" else "User"
                        chat_history_str += f"[{role}]: {msg.content}\n"
            
            # 执行智能记忆检索
            retrieval_result = await memory_manager.smart_retrieve(
                query=query_text,
                chat_history=chat_history_str,
                sender=state.get("sender_name", "User"),
                user_id=state.get("sender_qq", "unknown")
            )
            
            if retrieval_result["has_relevant_memory"]:
                logger.info(f"[{ts}] 📖 [Smart RAG] Found relevant memories")
                memory_context = f"【相关回忆】\n" + retrieval_result["memory_content"]
            else:
                # 如果智能记忆检索失败，回退到传统RAG检索
                logger.info(f"[{ts}] 📖 [Fallback RAG] Using traditional retrieval")
                docs = await vector_db.search(query_text, k=3)
                if docs:
                    # 过滤检索结果中的表情包信息
                    filtered_docs = []
                    for doc in docs:
                        # 移除检索结果中的表情包描述
                        filtered_doc = re.sub(r"【表情包:.*?】", "", doc)
                        if filtered_doc.strip():
                            filtered_docs.append(filtered_doc.strip())
                    if filtered_docs:
                        memory_context = f"【相关回忆】\n" + "\n".join(filtered_docs)
    except Exception as e:
        logger.error(f"[{ts}] [Smart RAG Error] {e}")
        # 异常情况下回退到传统RAG检索
        try:
            docs = await vector_db.search(query_text, k=3)
            if docs:
                filtered_docs = []
                for doc in docs:
                    filtered_doc = re.sub(r"【表情包:.*?】", "", doc)
                    if filtered_doc.strip():
                        filtered_docs.append(filtered_doc.strip())
                if filtered_docs:
                    memory_context = f"【相关回忆】\n" + "\n".join(filtered_docs)
        except Exception as fallback_e:
            logger.error(f"[{ts}] [Fallback RAG Error] {fallback_e}")
        pass
    
    # 获取用户记忆点和表达习惯
    user_memory_points = ""
    user_expression_habits = ""
    try:
        # 获取用户随机记忆点
        random_memory_points = relation_db.get_random_memory_points(real_user_id, num=3)
        if random_memory_points:
            memory_content = []
            for mp in random_memory_points:
                parts = mp.split(":")
                if len(parts) >= 3:
                    category = parts[0]
                    content = parts[1]
                    memory_content.append(f"{category}: {content}")
            if memory_content:
                user_memory_points = f"【用户记忆点】\n" + "\n".join(memory_content)
        
        # 获取用户表达习惯
        db_profile = await relation_db.get_user_profile(real_user_id)
        if db_profile and db_profile.relationship.expression_habits:
            expression_habits = db_profile.relationship.expression_habits[:5]  # 最多取5个习惯
            if expression_habits:
                user_expression_habits = f"【用户表达习惯】\n" + "\n".join(expression_habits)
    except Exception as e:
        logger.error(f"[{ts}] [User Memory Error] {e}")
        pass
    
    # 合并记忆上下文
    if user_memory_points:
        memory_context = user_memory_points + "\n" + memory_context
    if user_expression_habits:
        memory_context = memory_context + "\n" + user_expression_habits

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

    # 增强表达习惯指令
    expression_habits_instruction = ""
    if user_expression_habits:
        expression_habits_instruction = """
### 表达习惯模仿要求 (CRITICAL)
仔细分析用户的表达习惯，在回复中自然地融入这些习惯：
1. **用词模仿**: 使用用户常用的词汇、短语和表达方式
2. **句式模仿**: 模仿用户的句子结构和长度
3. **语气模仿**: 匹配用户的语气（比如用户喜欢用感叹号，你也可以适当使用）
4. **习惯表达**: 自然地使用用户的习惯用语和口头禅
5. **避免冲突**: 如果用户的表达习惯与Alice的核心性格有冲突，优先保持Alice的核心性格，但可以适当调整表达风格
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
    # 提取用户表达习惯，用于单独传递
    expression_habits_text = ""
    if "【用户表达习惯】" in memory_context:
        # 分离记忆上下文中的表达习惯部分
        parts = memory_context.split("【用户表达习惯】")
        if len(parts) == 2:
            memory_context = parts[0].strip()
            expression_habits_text = "【用户表达习惯】" + parts[1].strip()
    
    # 将增强的表达习惯指令插入到系统提示中
    modified_agent_prompt = AGENT_SYSTEM_PROMPT
    if expression_habits_instruction:
        # 找到插入位置（在用户表达习惯部分之后）
        insert_pos = modified_agent_prompt.find("{user_expression_habits}") + len("{user_expression_habits}")
        modified_agent_prompt = modified_agent_prompt[:insert_pos] + "\n" + expression_habits_instruction + modified_agent_prompt[insert_pos:]
    
    final_system_prompt = modified_agent_prompt.format(
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
        memories=memory_context,
        user_expression_habits=expression_habits_text
    ) + "\n\n" + format_instruction

    input_messages = [SystemMessage(content=final_system_prompt)]
    if len(msgs) > 0:
        # 过滤并清理历史消息，忽略表情包信息的影响
        cleaned_msgs = []
        for msg in msgs[-10:]:
            if isinstance(msg, HumanMessage):
                # 清理用户消息中的表情包描述
                content = msg.content
                if isinstance(content, str):
                    # 移除表情包描述
                    content = re.sub(r"【表情包:.*?】", "", content)
                    # 如果清理后内容为空，跳过这条消息
                    if content.strip():
                        cleaned_msg = HumanMessage(content=content.strip())
                        cleaned_msg.additional_kwargs = msg.additional_kwargs.copy()
                        cleaned_msgs.append(cleaned_msg)
                else:
                    cleaned_msgs.append(msg)
            else:
                cleaned_msgs.append(msg)
        input_messages.extend(cleaned_msgs)

    # 注入图片数据 (仅限 photo)
    all_image_artifacts = state.get("all_image_artifacts", [])
    if visual_type == "photo":
        if all_image_artifacts:
            # 处理多张图片
            image_content = []
            for i, image_artifact in enumerate(all_image_artifacts):
                if image_artifact["type"] == "photo" and image_artifact["data"]:
                    image_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_artifact['data']}"}})
            
            if image_content:
                # 添加图片附言
                image_content.append({"type": "text", "text": "（系统附言：这是用户发的图片，请结合回答。）"})
                input_messages.append(HumanMessage(content=image_content))
        elif image_data:
            # 兼容旧的单张图片逻辑
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
            parsed = {"monologue": "Raw Text", "action": "reply", "response": content}
        
        # 智能添加表情包到回复中
        if parsed.get("action") == "reply":
            response_content = parsed.get("response", "")
            if response_content and len(response_content.strip()) > 0:
                try:
                    emoji_service = get_emoji_service()
                    if emoji_service:
                        # 基于心理分析和对话情感来选择表情包
                        # 使用主要情感作为搜索关键词
                        text_emotion = primary_emotion
                        if secondary_emotion:
                            text_emotion += " " + secondary_emotion
                        
                        # 根据对话类型和亲密程度调整表情包使用策略
                        conversation_type = "group" if "group" in str(state.get("session_id", "")) else "private"
                        emoji_count = 0
                        
                        # 根据不同场景决定是否使用表情包
                        # 结合对话类型、情感强度、亲密程度和回复长度综合判断
                        emotion_intensity = abs(valence) + abs(arousal)
                        response_length = len(response_content)
                        
                        if conversation_type == "group":
                            # 群聊中更谨慎地使用表情包，但比之前更灵活
                            if intimacy > 50:  # 与用户有一定关系
                                # 情感强烈或回复较短时更可能发送表情
                                if ((emotion_intensity > 0.5 and random.random() < 0.5) or \
                                   (emotion_intensity > 0.8 and random.random() < 0.8)):
                                    emoji_count = 1
                        else:
                            # 私聊中更自然地使用表情包
                            if intimacy > 40:  # 与用户有一定关系
                                # 情感适中以上且随机概率
                                if ((emotion_intensity > 0.3 and random.random() < 0.6) or \
                                   (emotion_intensity > 0.7 and random.random() < 0.9)):
                                    emoji_count = 1
                        
                        # 回复内容过短或过长时调整概率
                        if response_length < 10:
                            # 短回复时更谨慎发送表情
                            emoji_count = 0 if random.random() < 0.3 else emoji_count
                        elif response_length > 100:
                            # 长回复时更可能发送表情来缓解阅读压力
                            emoji_count = 1 if random.random() < 0.4 else emoji_count
                        
                        if emoji_count > 0:
                            # 从对话历史中提取上下文信息
                            context = {
                                "last_message": last_human_content,
                                "message_history": msgs[-5:]
                            }
                            # 使用emoji_service根据上下文选择表情包
                            matching_emojis = emoji_service.get_emoji_for_context(context, count=1)
                            if matching_emojis:
                                logger.info(f"[{ts}] 😊 [Emoji] 为回复添加匹配表情包: {text_emotion} -> {matching_emojis[0].emotions}")
                                # 在回复内容末尾添加表情包引用
                                # 同时保存表情包信息，供后续分开发送使用
                                parsed["response"] = f"{response_content} [表情: {matching_emojis[0].emoji_hash}]"
                                parsed["emoji_info"] = matching_emojis[0]
                except Exception as e:
                    logger.error(f"[{ts}] ❌ [Emoji] 添加表情包失败: {e}")
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
