# app/core/global_store.py

import os
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel


# 定义情绪数据模型
class EmotionSnapshot(BaseModel):
    primary_emotion: str
    secondary_emotion: Optional[str] = None
    valence: float
    arousal: float
    stress: float
    fatigue: float
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
        self.stress = 0.05  # 压力 (0.0 ~ 1.0)
        self.fatigue = 0.05  # 疲惫度 (0.0 ~ 1.0)
        self.stamina = 100.0  # 体力值
        self.primary_emotion = "平静"
        self.secondary_emotion = None

        # --- 核心参数：情绪惯性 (0.0 ~ 1.0) ---
        # 0.8 表示新情绪只占 20% 权重，旧情绪占 80%
        # 这会让 Alice 很难被瞬间激怒，也很难瞬间哄好，显得更有性格
        self.mood_inertia = 0.75

        # --- 新增参数：情绪自然衰减速率 ---
        self.emotion_decay_rate = 0.05  # 情绪自然衰减的速率
        self.fatigue_recovery_rate = 0.02  # 疲劳恢复的速率
        self.stress_decay_rate = 0.03  # 压力衰减的速率
        
        # --- 新增参数：时间相关的情绪波动 ---
        self.time_based_mood_modifiers = {
            "morning": {"arousal": 0.1, "valence": 0.1},  # 早上更有活力
            "afternoon": {"arousal": -0.05, "valence": -0.05},  # 下午有点疲惫
            "evening": {"arousal": -0.1, "valence": 0.05}  # 晚上更放松
        }

        self.last_updated = datetime.now()

    def get_emotion_snapshot(self) -> EmotionSnapshot:
        # 先应用情绪自然衰减和时间影响
        self._apply_emotion_decay()
        self._apply_time_based_mood()
        
        return EmotionSnapshot(
            primary_emotion=self.primary_emotion,
            secondary_emotion=self.secondary_emotion,
            valence=self.valence,
            arousal=self.arousal,
            stress=self.stress,
            fatigue=self.fatigue,
            stamina=self.stamina,
            last_updated=self.last_updated.strftime("%Y-%m-%d %H:%M:%S")
        )

    def _apply_emotion_decay(self):
        """应用情绪的自然衰减，让情绪逐渐回归平静"""
        current_time = datetime.now()
        time_diff = (current_time - self.last_updated).total_seconds() / 3600  # 小时
        
        if time_diff < 0.1:  # 不足6分钟，不衰减
            return
        
        # 情绪自然回归中性
        self.valence += (0 - self.valence) * self.emotion_decay_rate * time_diff
        self.arousal += (0.4 - self.arousal) * self.emotion_decay_rate * time_diff  # 回归到中等激活度
        
        # 压力自然衰减
        self.stress = max(0.0, self.stress - self.stress_decay_rate * time_diff)
        
        # 疲劳自然恢复
        self.fatigue = max(0.0, self.fatigue - self.fatigue_recovery_rate * time_diff)
        
        # 体力自然恢复
        self.stamina = min(100.0, self.stamina + 5.0 * time_diff)  # 每小时恢复5点体力
        
        # 限制数值范围
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        
        self.last_updated = current_time
        
        # 更新情绪标签
        self.primary_emotion = self._derive_emotion_label()

    def _apply_time_based_mood(self):
        """应用时间相关的情绪波动"""
        current_hour = datetime.now().hour
        
        if 6 <= current_hour < 12:
            time_period = "morning"
        elif 12 <= current_hour < 18:
            time_period = "afternoon"
        else:
            time_period = "evening"
        
        modifiers = self.time_based_mood_modifiers.get(time_period, {})
        
        if modifiers:
            self.arousal = max(0.0, min(1.0, self.arousal + modifiers["arousal"]))
            self.valence = max(-1.0, min(1.0, self.valence + modifiers["valence"]))

    def update_emotion(self, valence_delta: float, arousal_delta: float, stress_delta: float = 0.0, fatigue_delta: float = 0.0, stamina_delta: float = 0.0,
                       new_primary: str = None, new_secondary: str = None):
        """
        使用指数移动平均 (EMA) 模拟情绪惯性，并添加情绪积累效应。
        """
        # 先应用自然衰减
        self._apply_emotion_decay()
        
        # 1. 情绪积累效应：连续的相似情绪会有更强的影响
        # 计算情绪方向一致性
        valence_direction = 1 if valence_delta > 0 else -1 if valence_delta < 0 else 0
        current_valence_direction = 1 if self.valence > 0.2 else -1 if self.valence < -0.2 else 0
        
        # 如果情绪方向一致，增强影响
        consistency_factor = 1.5 if valence_direction == current_valence_direction and valence_direction != 0 else 1.0
        
        # 2. 限制单次输入的冲击力，防止极端跳变，但考虑积累效应
        v_input = max(-0.5, min(0.5, valence_delta * consistency_factor))
        a_input = max(-0.5, min(0.5, arousal_delta * consistency_factor))
        s_input = max(-0.4, min(0.4, stress_delta * consistency_factor))
        f_input = max(-0.4, min(0.4, fatigue_delta * consistency_factor))

        # 3. 计算目标值
        target_valence = max(-1.0, min(1.0, self.valence + v_input))
        target_arousal = max(0.0, min(1.0, self.arousal + a_input))
        target_stress = max(0.0, min(1.0, self.stress + s_input))
        target_fatigue = max(0.0, min(1.0, self.fatigue + f_input))

        # 4. 应用惯性公式
        # New = Old * Inertia + Target * (1 - Inertia)
        self.valence = (self.valence * self.mood_inertia) + (target_valence * (1 - self.mood_inertia))
        self.arousal = (self.arousal * self.mood_inertia) + (target_arousal * (1 - self.mood_inertia))
        self.stress = (self.stress * self.mood_inertia) + (target_stress * (1 - self.mood_inertia))
        self.fatigue = (self.fatigue * self.mood_inertia) + (target_fatigue * (1 - self.mood_inertia))

        # 5. 体力更新 (直接扣减，无惯性)
        self.stamina = max(0.0, min(100.0, self.stamina + stamina_delta))
        
        # 6. 疲劳影响体力，体力影响疲劳
        self.stamina = max(0.0, self.stamina - self.fatigue * 2.0)  # 疲劳会缓慢消耗体力
        if self.stamina < 30.0:
            self.fatigue += 0.1  # 体力不足时，疲劳增加

        self.last_updated = datetime.now()

        # 7. 更新情绪标签
        if new_primary and new_primary != "N/A":
            self.primary_emotion = new_primary
        else:
            self.primary_emotion = self._derive_emotion_label()
        
        # 更新次要情绪标签
        self.secondary_emotion = new_secondary if new_secondary and new_secondary != "N/A" else None

    def _derive_emotion_label(self) -> str:
        """根据数值坐标反推情绪词 (PAD模型增强版)"""
        v, a, s, f = self.valence, self.arousal, self.stress, self.fatigue

        # 考虑压力和疲劳的影响
        if f > 0.7:  # 非常疲惫
            return "疲惫不堪"
        elif f > 0.4:  # 比较疲惫
            if v > 0.2: return "疲惫但愉快"
            elif v < -0.2: return "疲惫又沮丧"
            else: return "疲惫"
        
        if s > 0.7:  # 非常压力大
            if v < -0.3: return "焦虑不安"
            else: return "压力山大"
        elif s > 0.4:  # 有压力
            if v > 0.2: return "紧张但兴奋"
            elif v < -0.2: return "烦躁"
            else: return "紧张"

        # 核心情绪判断
        if v > 0.7 and a > 0.7: return "兴高采烈"
        if v > 0.5 and a > 0.5: return "开心"
        if v > 0.3 and a > 0.3: return "愉快"
        if v > 0.2 and a <= 0.3: return "惬意"
        if v > 0.1 and a < 0.2: return "放松"

        if v < -0.7 and a > 0.7: return "暴怒"
        if v < -0.5 and a > 0.5: return "愤怒"
        if v < -0.3 and a > 0.3: return "烦躁"
        if v < -0.4 and a <= 0.3: return "沮丧"
        if v < -0.2 and a < 0.2: return "低落"

        if abs(v) < 0.2 and a < 0.2: return "困倦/发呆"
        if abs(v) < 0.1 and a < 0.1: return "恍惚"
        return "平静"


global_store = GlobalStore()
