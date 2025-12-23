# app/core/global_store.py

import os
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel


# 定义情绪数据模型
class EmotionSnapshot(BaseModel):
    primary_emotion: str
    valence: float
    arousal: float
    stamina: float
    last_updated: str


class GlobalStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalStore, cls).__new__(cls)
            cls._instance._init_store()
        return cls._instance

    def _init_store(self):
        # --- 初始状态 ---
        self.valence = 0.1  # 愉悦度 (-1.0 ~ 1.0)
        self.arousal = 0.4  # 激活度 (0.0 ~ 1.0)
        self.stamina = 100.0  # 体力值
        self.primary_emotion = "平静"

        # --- 核心参数：情绪惯性 (0.0 ~ 1.0) ---
        # 0.8 表示新情绪只占 20% 权重，旧情绪占 80%
        # 这会让 Alice 很难被瞬间激怒，也很难瞬间哄好，显得更有性格
        self.mood_inertia = 0.75

        self.last_updated = datetime.now()

    def get_emotion_snapshot(self) -> EmotionSnapshot:
        return EmotionSnapshot(
            primary_emotion=self.primary_emotion,
            valence=self.valence,
            arousal=self.arousal,
            stamina=self.stamina,
            last_updated=self.last_updated.strftime("%Y-%m-%d %H:%M:%S")
        )

    def update_emotion(self, valence_delta: float, arousal_delta: float, stamina_delta: float = 0.0,
                       new_primary: str = None):
        """
        使用指数移动平均 (EMA) 模拟情绪惯性。
        """
        # 1. 限制单次输入的冲击力，防止极端跳变
        v_input = max(-0.4, min(0.4, valence_delta))
        a_input = max(-0.4, min(0.4, arousal_delta))

        # 2. 计算目标值
        target_valence = max(-1.0, min(1.0, self.valence + v_input))
        target_arousal = max(0.0, min(1.0, self.arousal + a_input))

        # 3. 应用惯性公式
        # New = Old * Inertia + Target * (1 - Inertia)
        self.valence = (self.valence * self.mood_inertia) + (target_valence * (1 - self.mood_inertia))
        self.arousal = (self.arousal * self.mood_inertia) + (target_arousal * (1 - self.mood_inertia))

        # 4. 体力更新 (直接扣减，无惯性)
        self.stamina = max(0.0, min(100.0, self.stamina + stamina_delta))
        self.last_updated = datetime.now()

        # 5. 推导或更新情绪标签
        if new_primary and new_primary != "N/A":
            # 如果 LLM 给了明确标签，我们采纳，但要检查是否与数值严重冲突
            self.primary_emotion = new_primary
        else:
            self.primary_emotion = self._derive_emotion_label()

    def _derive_emotion_label(self) -> str:
        """根据数值坐标反推情绪词 (PAD模型简化版)"""
        v, a = self.valence, self.arousal

        if v > 0.6 and a > 0.6: return "兴高采烈"
        if v > 0.3 and a > 0.3: return "开心"
        if v > 0.2 and a <= 0.3: return "惬意"

        if v < -0.6 and a > 0.6: return "愤怒"
        if v < -0.3 and a > 0.3: return "烦躁"
        if v < -0.3 and a <= 0.3: return "沮丧"

        if abs(v) < 0.2 and a < 0.2: return "困倦/发呆"
        return "平静"


global_store = GlobalStore()
