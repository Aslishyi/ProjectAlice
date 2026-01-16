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
from app.core.prompts import ALICE_CORE_PERSONA, AGENT_SYSTEM_PROMPT, build_prompt_with_persona
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
    增强型 JSON 解析器 - 专门修复 API 注入的脏数据和处理纯文本响应
    """
    if not text: return None

    # 🚀 [核心修复] 移除 API 强行注入的 system hint 垃圾信息
    text = re.sub(r"\[system hint:.*?\]", "", text, flags=re.IGNORECASE)
    text = text.strip()

    # 检查是否可能包含JSON
    if "{" in text and "}" in text:
        # 提取 Markdown JSON
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # 找到所有可能的JSON片段
            all_starts = [m.start() for m in re.finditer(r"\{\s*\"", text)]
            all_ends = [m.start() for m in re.finditer(r"\}\s*", text)]
            
            if all_starts and all_ends:
                # 找到最外层的JSON
                start = all_starts[0]
                # 找到与最外层start匹配的end
                depth = 0
                end = start
                for i, c in enumerate(text[start:]):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = start + i + 1
                            break
                
                if end > start:
                    text = text[start:end]
            else:
                # 简单提取第一个{到最后一个}之间的内容
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    text = text[start: end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # 修复JSON格式问题
                fixed_text = re.sub(r",\s*}", "}", text)  # 移除末尾的逗号
                fixed_text = re.sub(r",\s*]", "]", fixed_text)  # 移除数组末尾的逗号
                return json.loads(fixed_text)
            except:
                # 尝试更激进的修复
                try:
                    # 移除所有非JSON字符
                    clean_text = re.sub(r"[^\x00-\x7F]+|", "", text)  # 移除非ASCII字符
                    return json.loads(clean_text)
                except:
                    # 如果仍然无法解析，将其视为纯文本响应
                    # 这种情况通常发生在LLM没有遵循格式要求时
                    return {
                        "monologue": "LLM返回了纯文本响应，自动包装为JSON格式",
                        "action": "reply",
                        "args": "",
                        "response": text
                    }
    
    # 如果不包含JSON，将纯文本包装为预期的JSON格式
    # 这可以减少不必要的JSON解析失败警告
    return {
        "monologue": "LLM返回了纯文本响应，自动包装为JSON格式",
        "action": "reply",
        "args": "",
        "response": text
    }


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
                logger.info(f"[{ts}] 📖 [Smart RAG] Retrieved memory content: {retrieval_result['memory_content']}")
                memory_context = f"【相关回忆】\n" + retrieval_result["memory_content"]
            else:
                # 如果智能记忆检索失败，回退到传统RAG检索
                logger.info(f"[{ts}] 📖 [Fallback RAG] Using traditional retrieval")
                docs = await vector_db.search(query_text, k=3)
                logger.info(f"[{ts}] 📖 [Fallback RAG] Retrieved {len(docs) if docs else 0} documents")
                if docs:
                    logger.info(f"[{ts}] 📖 [Fallback RAG] Raw documents: {docs}")
                    # 过滤检索结果中的表情包信息
                    filtered_docs = []
                    for doc in docs:
                        # 移除检索结果中的表情包描述
                        filtered_doc = re.sub(r"【表情包:.*?】", "", doc)
                        if filtered_doc.strip():
                            filtered_docs.append(filtered_doc.strip())
                    logger.info(f"[{ts}] 📖 [Fallback RAG] Filtered to {len(filtered_docs)} documents")
                    if filtered_docs:
                        logger.info(f"[{ts}] 📖 [Fallback RAG] Final filtered documents: {filtered_docs}")
                        memory_context = f"【相关回忆】\n" + "\n".join(filtered_docs)
    except Exception as e:
        logger.error(f"[{ts}] [Smart RAG Error] {e}")
        # 异常情况下回退到传统RAG检索
        try:
            logger.info(f"[{ts}] 📖 [Exception RAG] Falling back to traditional retrieval due to Smart RAG error")
            docs = await vector_db.search(query_text, k=3)
            logger.info(f"[{ts}] 📖 [Exception RAG] Retrieved {len(docs) if docs else 0} documents")
            if docs:
                logger.info(f"[{ts}] 📖 [Exception RAG] Raw documents: {docs}")
                filtered_docs = []
                for doc in docs:
                    filtered_doc = re.sub(r"【表情包:.*?】", "", doc)
                    if filtered_doc.strip():
                        filtered_docs.append(filtered_doc.strip())
                logger.info(f"[{ts}] 📖 [Exception RAG] Filtered to {len(filtered_docs)} documents")
                if filtered_docs:
                    logger.info(f"[{ts}] 📖 [Exception RAG] Final filtered documents: {filtered_docs}")
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
    # 强制响应格式要求 - 必须严格遵守
    YOU MUST OUTPUT A VALID JSON OBJECT ONLY. NO OTHER TEXT OR EXPLANATION ALLOWED.
    YOU WILL BE PUNISHED SEVERELY IF YOU FAIL TO FOLLOW THIS INSTRUCTION.
    
    Response Format:
    {
      "monologue": "你的内部思考过程",
      "action": "reply",
      "args": "",
      "response": "要发送给用户的回复内容"
    }
    
    # 重要说明：
    1. 必须包含所有四个字段：monologue, action, args, response
    2. action字段只能是"reply"
    3. response字段不能为空
    4. 所有字段值必须用双引号包围
    5. 不能有任何多余的文本，包括Markdown格式、注释等
    6. 必须是有效的JSON格式
    
    # 错误示例（会被惩罚）：
    - 哦，这是一个很好的问题！{"response": "好的，我会帮助你"}
    - ```json {"response": "你好"} ```
    - {"response": "你好"} （缺少必要字段）
    - {'response': '你好'} （使用单引号）
    
    # 正确示例（必须严格按照此格式）：
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
    
    # 获取情绪和关系信息
    primary_emotion = state.get("primary_emotion", "平静")
    intimacy = state.get("intimacy", 0)
    familiarity = state.get("familiarity", 0)
    
    # 根据亲密度和熟悉度确定关系类型
    if intimacy > 70 and familiarity > 70:
        relation = "好朋友"
    elif intimacy > 40 and familiarity > 40:
        relation = "普通朋友"
    elif intimacy > 10 and familiarity > 10:
        relation = "熟人"
    else:
        relation = "陌生人"
    
    # 构建包含扩展人设和说话风格的完整core_persona
    scene = "private" if "private" in str(state.get("session_id", "")) else "group"
    complete_core_persona = await build_prompt_with_persona(ALICE_CORE_PERSONA, last_human_content, scene, primary_emotion, relation)
    
    final_system_prompt = modified_agent_prompt.format(
        core_persona=complete_core_persona,
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
            # 这种情况理论上不应该发生，因为robust_json_parse现在总是返回一个有效的JSON对象
            logger.error(f"[{ts}] ❌ [Agent JSON Parse Fatal Error] Raw: {content[:50]}...")
            parsed = {"monologue": "JSON Parse Fatal Error", "action": "reply", "response": "Someone tells Aslishyi there is a problem with his Alice."}
        
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
                        
                        # 检查是否连续使用了表情包
                        consecutive_emoji = False
                        if msgs and len(msgs) > 1:
                            # 检查上一条机器人回复是否包含表情包
                            last_bot_msg = next((msg for msg in reversed(msgs[:-1]) if hasattr(msg, "role") and msg.role == "assistant"), None)
                            if last_bot_msg and hasattr(last_bot_msg, "content") and "[表情:" in last_bot_msg.content:
                                consecutive_emoji = True
                        
                        if conversation_type == "group":
                            # 群聊中更谨慎地使用表情包
                            if intimacy > 50:  # 与用户有一定关系
                                # 情感强烈或回复较短时更可能发送表情
                                if ((emotion_intensity > 0.6 and random.random() < 0.3) or \
                                   (emotion_intensity > 0.9 and random.random() < 0.5)):
                                    emoji_count = 1
                        else:
                            # 私聊中更自然地使用表情包，但降低频率
                            if intimacy > 50:  # 提高亲密程度阈值
                                # 情感适中以上且随机概率，降低概率值
                                if ((emotion_intensity > 0.4 and random.random() < 0.3) or \
                                   (emotion_intensity > 0.8 and random.random() < 0.6)):
                                    emoji_count = 1
                        
                        # 回复内容过短或过长时调整概率
                        if response_length < 10:
                            # 短回复时更谨慎发送表情
                            emoji_count = 0 if random.random() < 0.7 else emoji_count
                        elif response_length > 100:
                            # 长回复时更可能发送表情来缓解阅读压力，但降低概率
                            emoji_count = 1 if random.random() < 0.2 else emoji_count
                        
                        # 避免连续使用表情包
                        if consecutive_emoji and emoji_count > 0:
                            emoji_count = 0 if random.random() < 0.8 else emoji_count
                        
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

    # 记录用户表达习惯和重要信息
    try:
        # 分析用户的表达习惯
        if last_human_content:
            # 计算消息长度特征
            msg_length = len(last_human_content)
            
            # 表情符号分析
            emoji_pattern = re.compile(r'[\u2600-\u27BF]|\[表情\]')
            emojis = emoji_pattern.findall(last_human_content)
            emoji_count = len(emojis)
            
            # 标点符号分析
            punctuation_pattern = re.compile(r'[!！?？。，、；：…]')
            punctuations = punctuation_pattern.findall(last_human_content)
            punctuation_count = len(punctuations)
            
            # 问句分析
            question_pattern = re.compile(r'[?？]')
            questions = question_pattern.findall(last_human_content)
            question_count = len(questions)
            
            # 感叹句分析
            exclamation_pattern = re.compile(r'[!！]')
            exclamations = exclamation_pattern.findall(last_human_content)
            exclamation_count = len(exclamations)
            
            # 重复字符分析
            repeat_pattern = re.compile(r'(.)\1{2,}')
            repeats = repeat_pattern.findall(last_human_content)
            repeat_count = len(repeats)
            
            # 记录表达习惯（基于使用频率和上下文）
            # 表情符号使用习惯
            if emoji_count > 0:
                if emoji_count >= 3:
                    relation_db.add_expression_habit(real_user_id, "喜欢频繁使用表情符号", confidence=0.9)
                elif emoji_count >= 1:
                    relation_db.add_expression_habit(real_user_id, "偶尔使用表情符号", confidence=0.7)
            
            # 标点符号使用习惯
            if punctuation_count > 0:
                punctuation_ratio = punctuation_count / msg_length
                if punctuation_ratio > 0.1:
                    relation_db.add_expression_habit(real_user_id, "使用丰富的标点符号", confidence=0.8)
                elif punctuation_ratio > 0.05:
                    relation_db.add_expression_habit(real_user_id, "使用规范的标点符号", confidence=0.6)
            
            # 问句使用习惯
            if question_count > 0:
                if question_count >= 3:
                    relation_db.add_expression_habit(real_user_id, "经常提出多个问题", confidence=0.9)
                elif question_count >= 1:
                    relation_db.add_expression_habit(real_user_id, "偶尔使用问句", confidence=0.7)
            
            # 感叹句使用习惯
            if exclamation_count > 0:
                if exclamation_count >= 2:
                    relation_db.add_expression_habit(real_user_id, "经常使用感叹句表达情绪", confidence=0.8)
                else:
                    relation_db.add_expression_habit(real_user_id, "偶尔使用感叹句", confidence=0.6)
            
            # 重复字符使用习惯
            if repeat_count > 0:
                relation_db.add_expression_habit(real_user_id, "偶尔使用重复字符强调", confidence=0.7)
            
            # 消息长度习惯
            if msg_length > 100:
                relation_db.add_expression_habit(real_user_id, "喜欢发送长消息", confidence=0.8)
            elif msg_length < 20:
                relation_db.add_expression_habit(real_user_id, "喜欢发送短消息", confidence=0.8)
        
        # 记录重要记忆点（更智能的判断逻辑）
        if query_text and len(query_text) > 5:
            # 定义重要信息的模式
            important_patterns = [
                # 个人信息（年龄、性别、职业等）
                r'(?:我(?:今年|现在)?(?:是|有)?(?:\d+|多少)?岁)|(?:我的(?:名字|年龄|性别|职业|生日|爱好|喜欢)是?.*)',
                # 事件信息（时间、地点、人物等）
                r'(?:(?:今天|明天|后天|昨天|上周|下周|去年|今年)(?:\w+)?)|(?:在(?:哪里|哪个地方|什么位置))|(?:和(?:谁|什么人))',
                # 情绪表达
                r'(?:(?:我觉得|我感到|我认为)(?:很|非常|有点)(?:开心|高兴|难过|伤心|生气|愤怒|失望|期待|紧张))',
                # 需求和请求
                r'(?:请(?:帮我|给我|告诉我|教我)|(?:我想|我要|我需要)(?:\w+))'
            ]
            
            # 检查是否包含重要信息
            has_important_info = False
            for pattern in important_patterns:
                if re.search(pattern, query_text, re.IGNORECASE):
                    has_important_info = True
                    break
            
            # 记录重要记忆点
            if has_important_info:
                # 尝试提取记忆点的类型
                memory_type = "普通对话"
                if re.search(r'我的(?:名字|年龄|性别|职业|生日|爱好|喜欢)', query_text, re.IGNORECASE):
                    memory_type = "个人信息"
                elif re.search(r'(?:今天|明天|后天|昨天|上周|下周|去年|今年)', query_text):
                    memory_type = "事件信息"
                elif re.search(r'(?:我觉得|我感到|我认为)(?:很|非常|有点)(?:开心|高兴|难过|伤心|生气|愤怒|失望|期待|紧张)', query_text):
                    memory_type = "情绪表达"
                elif re.search(r'(?:请(?:帮我|给我|告诉我|教我)|(?:我想|我要|我需要))', query_text):
                    memory_type = "需求请求"
                
                # 根据信息重要性设置权重
                weight = 1.0
                if memory_type in ["个人信息", "重要事件"]:
                    weight = 2.0
                elif memory_type in ["情绪表达", "需求请求"]:
                    weight = 1.5
                
                relation_db.add_memory_point(real_user_id, memory_type, query_text[:150], weight=weight)
            elif len(query_text) > 50:
                # 长消息即使没有明显的重要信息也可能包含有价值的内容
                relation_db.add_memory_point(real_user_id, "长文本对话", query_text[:150], weight=0.8)
    except Exception as e:
        logger.error(f"[{ts}] ❌ [Memory Record Error] {e}")
    
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
