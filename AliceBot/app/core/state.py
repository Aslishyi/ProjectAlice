from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # --- 基础消息流 ---
    messages: List[BaseMessage]
    conversation_summary: str

    # --- 核心身份与环境 ---
    session_id: str
    sender_qq: str
    sender_name: str
    is_group: bool
    is_mentioned: bool

    # --- 流程控制 ---
    should_reply: bool
    filter_reason: str
    is_proactive_mode: bool

    # --- 视觉优化 ---
    image_urls: List[str]

    # --- 上下文与状态 ---
    psychological_context: Dict[str, Any]
    global_emotion_snapshot: Dict[str, Any]
    internal_monologue: str
    emotion: Any

    # [修改点] 视觉相关字段
    current_image_artifact: Optional[str]  # 只有"有意义的图片"才存这里(Base64)
    visual_input: Optional[str]
    visual_type: Optional[str]  # 新增: 'photo', 'sticker', 'icon', 'none'

    current_activity: str
    last_interaction_ts: float
    next_step: str
    user_profile: Dict
    tool_call: Dict[str, Any]
