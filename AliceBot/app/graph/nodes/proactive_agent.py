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
from app.core.prompts import ALICE_CORE_PERSONA
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
    
    def should_initiate_interaction(self, user_id: str, last_interaction_time: float, user_feedback_score: float) -> bool:
        """判断是否应该发起主动交互"""
        # 1. 检查当前时间是否在活跃窗口内
        if not self.is_in_active_time_window():
            self.logger.debug("不在活跃时间窗口内，跳过主动交互")
            return False
        
        # 2. 计算沉默时长
        silence_hours = (time.time() - last_interaction_time) / 3600
        
        # 3. 检查沉默时长是否在合理范围内
        if silence_hours < PROACTIVE_CONFIG["min_silence_hours"] or silence_hours > PROACTIVE_CONFIG["max_silence_hours"]:
            self.logger.debug(f"沉默时长 ({silence_hours:.1f}小时) 不在合理范围，跳过主动交互")
            return False
        
        # 4. 计算触发概率
        base_probability = PROACTIVE_CONFIG["base_chance"]
        
        # 基于沉默时长的概率调整（钟形曲线，避免过于机械）
        if silence_hours < 6:
            # 短时间沉默：概率随时间线性增加
            silence_factor = min(1.5, silence_hours / PROACTIVE_CONFIG["min_silence_hours"])
        else:
            # 长时间沉默：概率逐渐降低（避免过度打扰）
            silence_factor = max(0.5, 1 - (silence_hours - 6) / 18)
        
        # 用户反馈调整
        feedback_factor = 1 + (user_feedback_score * PROACTIVE_CONFIG["feedback_factor"])
        
        final_probability = base_probability * silence_factor * feedback_factor
        final_probability = max(0.05, min(0.8, final_probability))
        
        # 5. 随机判断是否触发
        if random.random() < final_probability:
            self.logger.debug(f"触发主动交互，概率: {final_probability:.2f}")
            return True
        
        return False
    
    async def get_personalized_topics(self, user_id: str, limit: int = 5) -> List[str]:
        """获取个性化话题列表"""
        try:
            # 从向量存储中获取相关记忆点
            memories = await vector_store.search(
                query="",  # 空查询表示获取所有相关记忆
                k=10,
                categories=["兴趣爱好", "共同经历", "日常话题"]
            )
            
            topics = []
            if memories:
                for memory in memories:
                    if memory.content and len(memory.content) > 5:
                        topics.append(memory.content)
            
            # 如果没有足够的记忆点，使用默认话题
            if len(topics) < limit:
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
                # 随机选择一些默认话题补充
                while len(topics) < limit and default_topics:
                    topic = random.choice(default_topics)
                    if topic not in topics:
                        topics.append(topic)
                    default_topics.remove(topic)
            
            return topics[:limit]
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
    # 移除过于正式的表达
    formal_phrases = [
        "很高兴认识你", "乐意效劳", "根据我的知识", "我认为", "我觉得",
        "你好", "在吗", "请问", "感谢", "谢谢", "对不起", "抱歉"
    ]
    
    filtered_content = content
    for phrase in formal_phrases:
        if phrase in filtered_content:
            filtered_content = filtered_content.replace(phrase, "")
    
    # 移除过于亲密的表达
    intimate_phrases = [
        "亲爱的", "宝贝", "老公", "老婆", "哥哥", "姐姐", "弟弟", "妹妹",
        "我爱你", "我想你", "思念你", "喜欢你"
    ]
    
    for phrase in intimate_phrases:
        if phrase in filtered_content:
            filtered_content = filtered_content.replace(phrase, "")
    
    # 移除感叹号（Alice很少用）
    filtered_content = filtered_content.replace("!", "...")
    
    # 确保内容符合Alice的说话风格
    filtered_content = filtered_content.strip()
    if not filtered_content:
        return ""
    
    # 添加适当的语气词
    if not filtered_content.endswith(("...", "呢", "呀", "哦", "嗯", "吧")):
        endings = ["...", "呢", "呀", "哦", "", "嗯"]
        filtered_content += random.choice(endings)
    
    return filtered_content

def _ensure_alice_persona(content: str, intimacy: int) -> str:
    """确保内容符合Alice的人设"""
    # Alice的核心特点：简短、云淡风轻、避免麻烦
    lines = content.split("。")
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if line and len(line) < 30:  # 保持句子简短
            filtered_lines.append(line)
    
    if not filtered_lines:
        return ""
    
    result = "。".join(filtered_lines[:2])  # 最多两句话
    
    # 根据亲密度调整语气
    if intimacy > 80:
        # 高亲密度：可以稍微随意一点
        result = result.replace("...", "~").replace("哦", "哦~")
    elif intimacy < 30:
        # 低亲密度：保持距离感
        result = result.replace("~", "...").replace("呀", "哦")
    
    return _filter_unnatural_responses(result)

async def _generate_proactive_content(user_id: str, topics: List[str], intimacy: int) -> str:
    """生成符合人设的主动交互内容"""
    if not topics:
        return ""
    
    try:
        # 随机选择一个话题
        selected_topic = random.choice(topics)
        
        # 构建生成prompt
        prompt = f"""
        你是18岁女大学生Alice，性格云淡风轻，喜欢安静的咖啡馆和旧书店。
        现在你想主动和朋友聊聊天，基于以下话题点，用你平时的语气随便说点什么：
        话题点：{selected_topic}
        
        要求：
        1. 句子要短，10-20字左右
        2. 语气自然，不要太刻意
        3. 符合Alice云淡风轻的性格
        4. 不要用感叹号，多用省略号
        5. 不要太正式，就像平时说话一样
        6. 不要问太多问题，随便聊聊就行
        
        例子：
        话题：最近有没有读到什么有意思的书？
        Alice：最近在看一本老书... 挺有意思的呢
        
        话题：附近新开了家咖啡馆
        Alice：楼下新开的咖啡馆... 环境还不错
        """
        
        response = await cached_llm_invoke(
            llm, 
            [SystemMessage(content=prompt)],
            temperature=0.5,
            query_type="proactive_content",
            conversation_type="private"
        )
        
        content = response.content.strip()
        if content:
            # 确保内容符合Alice人设
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
        
        # 3. 检查关系阶段 - 低亲密度用户减少主动交互
        if intimacy < 20 and random.random() > 0.3:
            logger.debug(f"[{ts}] 用户亲密度较低 ({intimacy})，减少主动交互")
            return {"next_step": "silent"}
        
        # 4. 获取上次交互时间
        last_interaction_time = getattr(rel, "last_interaction_time", time.time() - 3600 * 2)
        
        # 5. 获取用户反馈分数
        feedback_score = interaction_manager.get_user_feedback_score(user_id)
        
        # 6. 判断是否应该发起主动交互
        if not interaction_manager.should_initiate_interaction(user_id, last_interaction_time, feedback_score):
            return {"next_step": "silent"}
            
        # 7. 获取个性化话题
        topics = await interaction_manager.get_personalized_topics(user_id)
        
        # 8. 生成主动内容
        content = await _generate_proactive_content(user_id, topics, intimacy)
        
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
