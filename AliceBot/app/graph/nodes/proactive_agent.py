import json
import time
import logging
import random
from datetime import datetime
from typing import List, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.state import AgentState
from app.core.config import config
from app.core.global_store import global_store
from app.memory.relation_db import relation_db
from app.core.prompts import ALICE_CORE_PERSONA, SOCIAL_VOLITION_PROMPT
from app.utils.cache import cached_llm_invoke
from app.memory.vector_store import vector_db as vector_store

# 配置日志
logger = logging.getLogger("ProactiveAgent")

# 主动交互配置
PROACTIVE_CONFIG = {
    # 活跃时间窗口（小时）
    "active_time_windows": [
        (9, 12),    # 上午
        (14, 17),   # 下午
        (19, 22)    # 晚上
    ],
    # 最小沉默时长（小时）
    "min_silence_hours": 1,
    # 最大沉默时长（小时）
    "max_silence_hours": 24,
    # 基础触发概率
    "base_chance": 0.3,
    # 用户反馈影响因子
    "feedback_factor": 0.2,
    # 个性化话题权重
    "topic_relevance_weight": 0.7,
    # 人设一致性过滤阈值
    "persona_consistency_threshold": 0.8
}

# 建议使用逻辑能力较强的模型
llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=0.6,  # 降低温度，让主动发言更符合人设，避免过于活泼
    api_key=config.MODEL_API_KEY,
    base_url=config.MODEL_URL
)

class ProactiveInteractionManager:
    def __init__(self):
        self.logger = logger
        self.feedback_store = {}
        
    def is_in_active_time_window(self) -> bool:
        """检查当前时间是否在活跃窗口内"""
        current_hour = datetime.now().hour
        for start, end in PROACTIVE_CONFIG["active_time_windows"]:
            if start <= current_hour < end:
                return True
        return False
    
    def should_initiate_interaction(self, user_id: str, last_interaction_time: float, user_feedback_score: float, intimacy: int, familiarity: int, trust: int, interest_match: int, stamina: float, interaction_patterns: Dict[str, Any]) -> bool:
        """判断是否应该发起主动交互"""
        # 1. 检查当前时间是否在活跃窗口内
        if not self.is_in_active_time_window():
            self.logger.debug("不在活跃时间窗口内，跳过主动交互")
            return False
        
        # 2. 检查体力值
        if stamina < 20:
            self.logger.debug(f"体力值过低 ({stamina:.1f})，跳过主动交互")
            return False
        
        # 3. 计算沉默时长
        silence_hours = (time.time() - last_interaction_time) / 3600
        
        # 4. 检查沉默时长是否在合理范围内
        if silence_hours < PROACTIVE_CONFIG["min_silence_hours"] or silence_hours > PROACTIVE_CONFIG["max_silence_hours"]:
            self.logger.debug(f"沉默时长 ({silence_hours:.1f}小时) 不在合理范围，跳过主动交互")
            return False
        
        # 5. 获取用户交互模式偏好
        preferred_response_time = interaction_patterns.get("preferred_response_time", None)
        current_hour = datetime.now().hour
        
        # 检查是否在用户偏好的回复时间段内
        if preferred_response_time:
            # 假设preferred_response_time格式为 [start_hour, end_hour]
            if not (preferred_response_time[0] <= current_hour < preferred_response_time[1]):
                self.logger.debug(f"当前时间不在用户偏好的回复时间段内，跳过主动交互")
                return False
        
        # 6. 计算触发概率
        base_probability = PROACTIVE_CONFIG["base_chance"]
        
        # 基于关系亲密度的调整
        intimacy_factor = 0.5 + (intimacy / 100)  # 0.5-1.5
        
        # 基于熟悉度的调整
        familiarity_factor = 0.8 + (familiarity / 500)  # 0.8-1.0
        
        # 基于信任度的调整
        trust_factor = 0.8 + (trust / 500)  # 0.8-1.0
        
        # 基于兴趣匹配度的调整
        interest_factor = 0.5 + (interest_match / 100)  # 0.5-1.5
        
        # 基于沉默时长的概率调整（更智能的曲线）
        if silence_hours < 6:
            # 短时间沉默：概率随时间线性增加，但受亲密度影响
            silence_factor = min(1.5, (silence_hours / PROACTIVE_CONFIG["min_silence_hours"]) * intimacy_factor)
        elif silence_hours < 12:
            # 中等时间沉默：保持较高概率
            silence_factor = 1.2 * intimacy_factor
        else:
            # 长时间沉默：概率逐渐降低，但受熟悉度和信任度影响
            silence_factor = max(0.5, (1 - (silence_hours - 12) / 24) * (familiarity_factor + trust_factor) / 2)
        
        # 用户反馈调整，权重更高
        feedback_factor = 1 + (user_feedback_score * PROACTIVE_CONFIG["feedback_factor"] * 1.5)
        
        # 综合所有因子
        final_probability = base_probability * silence_factor * feedback_factor * interest_factor
        
        # 根据关系阶段调整最终概率
        if intimacy < 30:
            # 低亲密度用户：降低触发概率
            final_probability *= 0.7
        elif intimacy > 70:
            # 高亲密度用户：适当提高触发概率
            final_probability *= 1.2
        
        # 限制概率范围
        final_probability = max(0.03, min(0.85, final_probability))
        
        # 7. 随机判断是否触发
        if random.random() < final_probability:
            self.logger.debug(f"触发主动交互，概率: {final_probability:.2f}")
            return True
        
        return False
    
    def _check_topic_relevance(self, topic: str, favorite_topics: List[str], avoid_topics: List[str]) -> bool:
        """检查话题相关性和是否为避免话题"""
        # 1. 检查是否为避免话题
        if avoid_topics:
            for avoid_topic in avoid_topics:
                if avoid_topic in topic:
                    return False
        
        # 2. 检查是否与兴趣话题相关
        if favorite_topics:
            for fav_topic in favorite_topics:
                if fav_topic in topic:
                    return True
        
        # 3. 默认返回True（允许使用）
        return True
    
    def _score_topic_relevance(self, topic: str, favorite_topics: List[str], memory_topics: List[str]) -> float:
        """为话题打分，评估其相关性"""
        score = 0.5  # 基础分数
        
        # 与兴趣话题匹配度
        if favorite_topics:
            for fav_topic in favorite_topics:
                if fav_topic in topic:
                    score += 0.3
        
        # 与记忆点匹配度
        if memory_topics:
            for mem_topic in memory_topics:
                if mem_topic in topic:
                    score += 0.2
        
        return min(1.0, score)
    
    async def get_personalized_topics(self, user_id: str, limit: int = 5) -> List[str]:
        """获取个性化话题列表"""
        try:
            # 获取用户关系数据
            profile = await relation_db.get_user_profile(user_id)
            rel = profile.relationship
            
            # 优先使用用户感兴趣的话题
            favorite_topics = rel.favorite_topics.copy()
            avoid_topics = rel.avoid_topics.copy()
            
            # 获取用户记忆点
            memory_points = rel.memory_points
            memory_topics = []
            
            # 解析记忆点，过滤出兴趣爱好和日常话题相关的内容
            for mp in memory_points:
                if isinstance(mp, str):
                    parts = mp.split(":")
                    if len(parts) >= 3:
                        category, content, weight = parts[0], ":".join(parts[1:-1]), float(parts[-1])
                        # 只保留高权重的记忆点
                        if weight > 0.5 and category in ["兴趣爱好", "共同经历", "日常话题"]:
                            memory_topics.append((content.strip(), weight))
            
            # 对记忆点按权重排序
            memory_topics.sort(key=lambda x: x[1], reverse=True)
            memory_topic_texts = [topic for topic, weight in memory_topics]
            
            # 从向量存储中获取相关记忆点，使用用户的兴趣作为查询
            vector_query = " ".join(favorite_topics[:3]) if favorite_topics else "日常话题"
            vector_memories = await vector_store.search(
                query=vector_query,
                k=10,
                categories=["兴趣爱好", "共同经历", "日常话题"]
            )
            
            vector_topics = []
            if vector_memories:
                for memory in vector_memories:
                    if memory.content and len(memory.content) > 5:
                        vector_topics.append(memory.content)
            
            # 合并所有话题源并打分
            all_topic_candidates = []
            
            # 1. 优先添加用户感兴趣的话题
            for topic in favorite_topics:
                if topic and self._check_topic_relevance(topic, favorite_topics, avoid_topics):
                    relevance_score = self._score_topic_relevance(topic, favorite_topics, memory_topic_texts)
                    all_topic_candidates.append((topic, relevance_score, "favorite"))
            
            # 2. 添加记忆点话题
            for topic in memory_topic_texts:
                if topic and self._check_topic_relevance(topic, favorite_topics, avoid_topics):
                    relevance_score = self._score_topic_relevance(topic, favorite_topics, memory_topic_texts)
                    all_topic_candidates.append((topic, relevance_score, "memory"))
            
            # 3. 添加向量存储话题
            for topic in vector_topics:
                if topic and self._check_topic_relevance(topic, favorite_topics, avoid_topics):
                    relevance_score = self._score_topic_relevance(topic, favorite_topics, memory_topic_texts)
                    all_topic_candidates.append((topic, relevance_score, "vector"))
            
            # 4. 如果话题不够，使用默认话题（经过避免话题过滤）
            default_topics = [
                "最近有没有读到什么有意思的书？",
                "附近新开了家咖啡馆，环境挺安静的...",
                "旧书店打折，你有想去看看吗？",
                "今天天气不错，适合出门散步呢",
                "最近总是睡不够，你也这样吗？",
                "听说有部老电影重映了，好像还不错",
                "昨天在咖啡馆看到一只很可爱的猫",
                "最近在听一些老歌，突然觉得以前的歌更有味道",
                "你平时喜欢去哪些安静的地方？",
                "今天尝试做了手冲咖啡，虽然味道一般..."
            ]
            
            # 过滤默认话题，避免使用用户不喜欢的话题
            filtered_defaults = []
            for topic in default_topics:
                if self._check_topic_relevance(topic, favorite_topics, avoid_topics):
                    relevance_score = self._score_topic_relevance(topic, favorite_topics, memory_topic_texts)
                    filtered_defaults.append((topic, relevance_score, "default"))
            
            # 添加默认话题候选
            all_topic_candidates.extend(filtered_defaults)
            
            # 去重
            seen_topics = set()
            unique_candidates = []
            for topic, score, source in all_topic_candidates:
                if topic not in seen_topics:
                    seen_topics.add(topic)
                    unique_candidates.append((topic, score, source))
            
            # 按相关性分数排序
            unique_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 选择前limit个话题
            result_topics = [topic for topic, score, source in unique_candidates[:limit]]
            
            # 确保多样性：最多允许2个相同来源的话题
            source_count = {}
            diverse_topics = []
            for topic, score, source in unique_candidates:
                if source_count.get(source, 0) < 2:
                    diverse_topics.append(topic)
                    source_count[source] = source_count.get(source, 0) + 1
                    if len(diverse_topics) >= limit:
                        break
            
            # 如果多样性话题不足，使用原始排序的话题补充
            if len(diverse_topics) < limit:
                for topic in result_topics:
                    if topic not in diverse_topics:
                        diverse_topics.append(topic)
                        if len(diverse_topics) >= limit:
                            break
            
            return diverse_topics[:limit]
        except Exception as e:
            self.logger.error(f"获取个性化话题失败: {e}")
            return []
    
    def update_user_feedback(self, user_id: str, feedback_type: str):
        """更新用户反馈（positive/negative）"""
        if user_id not in self.feedback_store:
            self.feedback_store[user_id] = {"positive": 0, "negative": 0, "last_updated": time.time()}
        
        # 更新反馈计数
        if feedback_type == "positive":
            self.feedback_store[user_id]["positive"] += 1
        elif feedback_type == "negative":
            self.feedback_store[user_id]["negative"] += 1
        
        self.feedback_store[user_id]["last_updated"] = time.time()
    
    def get_user_feedback_score(self, user_id: str) -> float:
        """获取用户反馈分数（-1到1之间）"""
        if user_id not in self.feedback_store:
            return 0.0
        
        feedback = self.feedback_store[user_id]
        total = feedback["positive"] + feedback["negative"]
        if total == 0:
            return 0.0
        
        # 计算反馈分数（positive - negative）/ total
        score = (feedback["positive"] - feedback["negative"]) / total
        return score

# 初始化主动交互管理器
interaction_manager = ProactiveInteractionManager()

def _filter_unnatural_responses(content: str) -> str:
    """过滤不符合Alice人设的不自然回应"""
    # Alice的核心性格：云淡风轻、波澜不惊、不刻意、不讨好
    
    # 1. 移除过于正式的表达
    formal_phrases = [
        "很高兴认识你", "乐意效劳", "根据我的知识", "我认为", "我觉得",
        "你好", "在吗", "请问", "感谢", "谢谢", "对不起", "抱歉",
        "请问有什么可以帮助你的", "随时为你服务", "我来帮你",
        "我明白了", "我理解", "你说得对", "确实如此"
    ]
    
    filtered_content = content
    for phrase in formal_phrases:
        if phrase in filtered_content:
            filtered_content = filtered_content.replace(phrase, "")
    
    # 2. 移除过于亲密的表达
    intimate_phrases = [
        "亲爱的", "宝贝", "老公", "老婆", "哥哥", "姐姐", "弟弟", "妹妹",
        "我爱你", "我想你", "思念你", "喜欢你", "抱抱", "亲亲",
        "想你啦", "爱你", "我的", "专属", "唯一"
    ]
    
    for phrase in intimate_phrases:
        if phrase in filtered_content:
            filtered_content = filtered_content.replace(phrase, "")
    
    # 3. 移除刻意引导对话的表达
    guiding_phrases = [
        "那你呢", "你觉得呢", "有什么想法", "分享给我听听",
        "有什么感受", "觉得怎么样", "随时来找我聊聊哦",
        "对吧", "是不是", "对吗", "好不好", "可以吗",
        "怎么样", "呢",  # 注意：只移除作为疑问词的"呢"，保留作为语气词的"呢"
    ]
    
    for phrase in guiding_phrases:
        if filtered_content.endswith(phrase):
            filtered_content = filtered_content[:-len(phrase)]
        elif f"{phrase}" in filtered_content:
            filtered_content = filtered_content.replace(f" {phrase}", "")
    
    # 4. 移除感叹号和问号（Alice很少用强烈的标点）
    filtered_content = filtered_content.replace("!", "...")
    filtered_content = filtered_content.replace("?", "...")
    
    # 5. 移除过长的句子（保持简短）
    sentences = filtered_content.split("。")
    short_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) < 25:  # 更严格的长度限制
            short_sentences.append(sentence)
    
    filtered_content = "。".join(short_sentences)
    
    # 6. 确保内容符合Alice的说话风格
    filtered_content = filtered_content.strip()
    if not filtered_content:
        return ""
    
    # 7. 添加适当的语气词（符合云淡风轻的风格）
    valid_endings = ["...", "呢", "呀", "哦", "嗯", ""]
    if not any(filtered_content.endswith(ending) for ending in valid_endings):
        filtered_content += random.choice(["...", "哦", ""])
    
    # 8. 确保句子简短（最多25字）
    if len(filtered_content) > 25:
        filtered_content = filtered_content[:25] + "..."
    
    return filtered_content

def _ensure_alice_persona(content: str, intimacy: int) -> str:
    """确保内容符合Alice的人设"""
    # Alice的核心特点：简短、云淡风轻、避免麻烦、不刻意
    
    # 1. 初步过滤
    filtered_content = _filter_unnatural_responses(content)
    if not filtered_content:
        return ""
    
    # 2. 保持句子数量限制（最多两句话）
    lines = filtered_content.split("。")
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            filtered_lines.append(line)
    
    if not filtered_lines:
        return ""
    
    # 最多两句话
    result = "。".join(filtered_lines[:2])
    
    # 3. 根据亲密度调整语气（更细腻的调整）
    if intimacy > 85:
        # 极高亲密度：可以稍微随意一点，但仍然保持云淡风轻
        result = result.replace("...", "~").replace("哦", "哦~")
        # 可以添加一些轻微的亲昵语气词
        if not any(result.endswith(ending) for ending in ["~", "哦~", "呢"]):
            result += "~"
    elif intimacy > 70:
        # 高亲密度：保持自然，略微随意
        result = result.replace("...", "~").replace("哦", "哦~")
    elif intimacy < 35:
        # 低亲密度：保持距离感，更冷淡
        result = result.replace("~", "...").replace("呀", "哦").replace("哦~", "哦")
        # 移除过于活泼的语气词
        result = result.replace("哈", "").replace("嘿", "")
    elif intimacy < 20:
        # 极低亲密度：非常冷淡，尽量简短
        result = result.replace("~", "...").replace("呀", "").replace("哦~", "")
        # 只保留最核心的内容
        if len(result) > 15:
            result = result[:15] + "..."
    
    # 4. 最终过滤，确保符合Alice的核心风格
    final_content = _filter_unnatural_responses(result)
    
    # 5. 确保内容不是刻意的提问或引导
    if any(final_content.endswith(ending) for ending in ["...?", "?", "呢", "吗", "吧"]):
        # 转换为陈述句
        final_content = final_content[:-1] + "..."
    
    # 6. 添加一些Alice特有的说话习惯（偶尔的错别字或省略）
    alice_mannerisms = [
        lambda x: x.replace("什么", "啥"),
        lambda x: x.replace("怎么", "咋"),
        lambda x: x.replace("没有", "没"),
        lambda x: x.replace("是不是", "是不"),
        lambda x: x  # 不修改
    ]
    
    # 随机应用一个说话习惯（30%概率）
    if random.random() < 0.3:
        final_content = random.choice(alice_mannerisms)(final_content)
    
    return final_content

async def _generate_proactive_content(user_id: str, topics: List[str], intimacy: int, current_time: str, silence_duration: str, stamina: float, chat_type: str, user_name: str, familiarity: int, trust: int, interest_match: int, communication_style: str) -> str:
    """生成符合人设的主动交互内容"""
    if not topics:
        return ""
    
    try:
        # 随机选择一个话题，但基于话题的相关性和多样性
        selected_topic = random.choice(topics)
        
        # 获取用户关系数据，用于构建更个性化的prompt
        profile = await relation_db.get_user_profile(user_id)
        rel = profile.relationship
        
        # 生成当前情绪状态，基于用户的情感趋势
        sentiment_trends = rel.sentiment_trends
        current_mood = "平静"
        
        # 分析最近的情感趋势
        if sentiment_trends:
            # 获取最近5条情感记录
            recent_sentiments = sentiment_trends[-5:]
            # 计算积极/消极情感的比例
            positive_count = sum(1 for trend in recent_sentiments if trend.get("sentiment") in ["开心", "愉快", "兴高采烈"])
            negative_count = sum(1 for trend in recent_sentiments if trend.get("sentiment") in ["低落", "沮丧", "烦躁"])
            
            if positive_count > negative_count:
                current_mood = "愉快"
            elif negative_count > positive_count:
                current_mood = "低落"
        
        # 填充SOCIAL_VOLITION_PROMPT所需的参数
        prompt = SOCIAL_VOLITION_PROMPT.format(
            alice_core_persona=ALICE_CORE_PERSONA,
            current_time=current_time,
            time_period="上午" if 9 <= int(current_time.split(":")[0]) < 12 else "下午" if 12 <= int(current_time.split(":")[0]) < 18 else "晚上",
            silence_duration=silence_duration,
            mood=current_mood,
            stamina=stamina,
            chat_type=chat_type,
            user_name=user_name,
            intimacy=intimacy,
            familiarity=familiarity,
            trust=trust,
            interest_match=interest_match,
            relation_tags=", ".join(rel.tags) if rel.tags else "",
            relation_notes=rel.notes if rel.notes else "",
            vision_desc="无",  # 主动发起时通常没有图片
            personalized_info=f"感兴趣的话题: {selected_topic}，用户沟通风格: {communication_style}",
            conversation_summary=f"最近的话题: {selected_topic}"
        )
        
        # 使用动态人设管理系统构建更丰富的prompt
        from app.core.prompts import build_prompt_with_persona
        contextual_prompt = await build_prompt_with_persona(
            core_persona=prompt,
            context=f"当前正在与{user_name}进行主动交互，话题是{selected_topic}",
            scene=chat_type,
            emotion=current_mood,
            relation=f"好感度{intimacy}/100",
            max_extended_items=3,
            max_contextual_items=2
        )
        
        # 根据沟通风格调整temperature
        temperature = 0.5
        if communication_style == "playful":
            temperature = 0.7
        elif communication_style == "formal":
            temperature = 0.3
        
        response = await cached_llm_invoke(
            llm, 
            [SystemMessage(content=contextual_prompt)],
            temperature=temperature,
            query_type="proactive_content",
            conversation_type=chat_type
        )
        
        content = response.content.strip()
        if content:
            try:
                # 解析JSON响应
                import json
                result = json.loads(content)
                proactive_content = result.get("content", "")
                if proactive_content:
                    # 确保内容符合Alice人设
                    return _ensure_alice_persona(proactive_content, intimacy)
            except json.JSONDecodeError:
                # 如果不是JSON格式，直接使用内容
                return _ensure_alice_persona(content, intimacy)
        
        return ""
    except Exception as e:
        logger.error(f"生成主动内容失败: {e}")
        return ""

async def proactive_node(state: AgentState):
    """主动交互节点 - 自然触发版本"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{ts}] --- [Proactive] Checking interaction opportunity... ---")
    
    # 1. 获取基本上下文
    try:
        user_id = state.get("sender_qq", "unknown")
        user_display_name = state.get("sender_name", "User")
        is_group = state.get("is_group", False)
        session_id = state.get("session_id", "unknown")
        msgs = state.get("messages", [])
        
        if not user_id or user_id == "unknown":
            logger.warning(f"[{ts}] 缺少用户ID，跳过主动交互")
            return {"next_step": "silent"}
    except Exception as e:
        logger.error(f"[{ts}] 获取上下文失败: {e}")
        return {"next_step": "silent"}
    
    # 2. 获取用户关系数据
    try:
        profile = await relation_db.get_user_profile(user_id)
        rel = profile.relationship
        intimacy = rel.intimacy
        familiarity = rel.familiarity
        trust = rel.trust
        interest_match = rel.interest_match
        
        # 3. 检查关系阶段 - 低亲密度用户减少主动交互
        if intimacy < 20 and random.random() > 0.3:
            logger.debug(f"[{ts}] 用户亲密度较低 ({intimacy})，减少主动交互")
            return {"next_step": "silent"}
        
        # 4. 获取上次交互时间
        last_interaction_time = getattr(rel, "last_interaction_time", time.time() - 3600 * 2)
        
        # 5. 获取用户反馈分数
        feedback_score = interaction_manager.get_user_feedback_score(user_id)
        
        # 6. 获取体力值
        stamina = getattr(rel, "stamina", 80.0)  # 使用默认值80.0如果没有
        
        # 7. 判断是否应该发起主动交互
        if not interaction_manager.should_initiate_interaction(
            user_id, 
            last_interaction_time, 
            feedback_score, 
            intimacy, 
            familiarity, 
            trust, 
            interest_match, 
            stamina, 
            rel.interaction_patterns
        ):
            return {"next_step": "silent"}
            
        # 8. 获取个性化话题
        topics = await interaction_manager.get_personalized_topics(user_id)
        
        # 9. 生成主动内容
        # 准备SOCIAL_VOLITION_PROMPT所需的参数
        current_time = datetime.now().strftime("%H:%M")
        # 计算沉默时长（小时）
        silence_hours = (time.time() - last_interaction_time) / 3600
        silence_duration = f"{silence_hours:.1f}小时"
        chat_type = "group" if is_group else "private"
        user_name = user_display_name
        
        content = await _generate_proactive_content(
            user_id, topics, intimacy, current_time, silence_duration, stamina, 
            chat_type, user_name, familiarity, trust, interest_match, rel.communication_style
        )
        
        if not content or len(content.strip()) < 5:
            logger.debug(f"[{ts}] 生成的内容不符合要求，跳过主动交互")
            return {"next_step": "silent"}
            
        # 9. 构建AI消息
        ai_msg = AIMessage(content=content)
        
        # 10. 更新最后交互时间
        rel.last_interaction_time = time.time()
        relation_db.update_relationship(user_id, user_id, rel)
        
        # 11. 消耗体力
        stamina_cost = -1.5 if is_group else -2.0  # 减少体力消耗，避免频繁触发
        global_store.update_emotion(0, 0, stamina_delta=stamina_cost)
        
        logger.info(f"[{ts}] 🤖 [Proactive] INITIATE_TOPIC | Content: {content}")
        
        return {
            "messages": msgs + [ai_msg],
            "next_step": "speak",
            "internal_monologue": f"[Social Volition] Intent: initiate_topic, Reason: 基于用户沉默时长和关系亲密度的自然触发, ChatType: {'Group' if is_group else 'Private'}"
        }
        
    except Exception as e:
        logger.error(f"[{ts}] 主动交互失败: {e}")
        return {"next_step": "silent"}

# 额外的工具函数，用于外部模块调用
def update_proactive_feedback(user_id: str, is_positive: bool):
    """更新主动交互的用户反馈"""
    feedback_type = "positive" if is_positive else "negative"
    interaction_manager.update_user_feedback(user_id, feedback_type)
    logger.info(f"Updated proactive feedback for {user_id}: {feedback_type}")
