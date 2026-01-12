# === 修改文件: app/memory/relation_db.py ===

import json
import os
import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Union, Optional
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text, JSON
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

# 导入数据库配置
from app.core.database import Base, engine, SessionLocal, init_db

# 配置日志
logger = logging.getLogger(__name__)

# JSON文件路径（用于数据迁移）
# 获取当前文件所在目录的父目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD_JSON_DB = os.path.join(BASE_DIR, "data", "user_profiles.json")
# 迁移完成标记文件
MIGRATION_COMPLETE_FILE = os.path.join(BASE_DIR, "data", "migration_complete.txt")


class Relationship(BaseModel):
    target_id: str
    relation_type: str = "acquaintance"
    intimacy: int = Field(default=60, ge=0, le=100)  # 好感度
    familiarity: int = Field(default=50, ge=0, le=100)  # 熟悉度
    trust: int = Field(default=50, ge=0, le=100)  # 信任度
    interest_match: int = Field(default=50, ge=0, le=100)  # 兴趣匹配度
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    nickname_for_user: str = ""
    memory_points: List[str] = Field(default_factory=list)  # 记忆点列表，格式：category:content:weight:timestamp
    expression_habits: List[str] = Field(default_factory=list)  # 表达习惯列表
    group_nicknames: List[Dict[str, str]] = Field(default_factory=list)  # 群昵称列表，每个元素包含group_id和nickname
    
    # 新增字段
    communication_style: str = "casual"  # 沟通风格: casual, formal, playful
    favorite_topics: List[str] = Field(default_factory=list)  # 感兴趣的话题
    avoid_topics: List[str] = Field(default_factory=list)  # 避免的话题
    interaction_patterns: Dict[str, Any] = Field(default_factory=dict)  # 交互模式（如回复时间偏好）
    sentiment_trends: List[Dict[str, Any]] = Field(default_factory=list)  # 情感变化趋势


class UserProfile(BaseModel):
    name: str
    qq_id: str = ""
    relationship: Relationship


# 数据库模型
class UserProfileModel(Base):
    __tablename__ = "user_profiles"
    
    qq_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    relationship_data = Column(JSON, nullable=False)  # 存储Relationship对象的JSON数据
    updated_at = Column(String(50), default=lambda: str(time.time()))


class GlobalRelationDB:
    def __init__(self):
        # 初始化数据库
        init_db()
        
        # 检查是否需要从JSON迁移数据
        self._migrate_from_json()
    
    def calculate_memory_point_weight(self, memory_content: str, interaction_count: int = 1, recency: int = 1) -> float:
        """
        计算记忆点权重
        
        Args:
            memory_content: 记忆内容
            interaction_count: 互动次数
            recency: 时间衰减因子（1表示最新，值越大越旧）
            
        Returns:
            计算后的权重
        """
        # 基础权重
        base_weight = 1.0
        
        # 内容长度权重（越长的内容权重可能越高）
        content_weight = min(2.0, 1.0 + len(memory_content) / 100)
        
        # 互动次数权重
        interaction_weight = min(3.0, 1.0 + interaction_count * 0.5)
        
        # 时间衰减权重
        recency_weight = max(0.1, 1.0 - (recency - 1) * 0.1)
        
        # 综合权重
        total_weight = base_weight * content_weight * interaction_weight * recency_weight
        return round(total_weight, 2)
    
    def analyze_communication_style(self, message_content: str) -> str:
        """
        分析用户的沟通风格
        
        Args:
            message_content: 用户消息内容
            
        Returns:
            沟通风格（casual, formal, playful）
        """
        # 简单的沟通风格分析
        casual_words = ["哈哈", "嘿嘿", "嗯嗯", "哦哦", "呀", "呢", "啦", "哒", "哦", "啊"]
        formal_words = ["您好", "请问", "感谢", "谢谢", "请", "贵", "令"]
        playful_words = ["^_^", "😄", "😁", "😃", "😂", "😆", "😊", "😉", "😋", "😎"]
        
        # 计算各种风格的得分
        casual_score = sum(1 for word in casual_words if word in message_content)
        formal_score = sum(1 for word in formal_words if word in message_content)
        playful_score = sum(1 for word in playful_words if word in message_content)
        
        # 根据得分确定风格
        scores = {
            "casual": casual_score,
            "formal": formal_score,
            "playful": playful_score
        }
        
        # 返回得分最高的风格
        return max(scores, key=scores.get)
    
    def update_communication_style(self, user_qq: str, style: str) -> bool:
        """
        更新用户的沟通风格
        
        Args:
            user_qq: 用户QQ号
            style: 沟通风格（casual, formal, playful）
            
        Returns:
            bool: 是否更新成功
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                relationship_data = profile.relationship_data
                if not relationship_data:
                    relationship_data = {"target_id": user_qq}
                
                relationship_data["communication_style"] = style
                profile.relationship_data = relationship_data
                profile.updated_at = str(time.time())
                db.commit()
                return True
            else:
                # 用户不存在，创建新用户
                relationship = Relationship(target_id=user_qq, communication_style=style)
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=relationship.model_dump()
                )
                
                db.add(new_profile)
                db.commit()
                return True
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 更新沟通风格失败: {str(e)}")
            return False
        finally:
            db.close()
    
    def add_favorite_topic(self, user_qq: str, topic: str) -> bool:
        """
        添加用户感兴趣的话题
        
        Args:
            user_qq: 用户QQ号
            topic: 感兴趣的话题
            
        Returns:
            bool: 是否添加成功
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                relationship_data = profile.relationship_data
                if not relationship_data:
                    relationship_data = {"target_id": user_qq, "favorite_topics": []}
                
                if "favorite_topics" not in relationship_data:
                    relationship_data["favorite_topics"] = []
                
                if topic not in relationship_data["favorite_topics"]:
                    relationship_data["favorite_topics"].append(topic)
                    profile.relationship_data = relationship_data
                    profile.updated_at = str(time.time())
                    db.commit()
                
                return True
            else:
                # 用户不存在，创建新用户
                relationship = Relationship(target_id=user_qq, favorite_topics=[topic])
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=relationship.model_dump()
                )
                
                db.add(new_profile)
                db.commit()
                return True
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 添加感兴趣话题失败: {str(e)}")
            return False
        finally:
            db.close()
    
    def add_avoid_topic(self, user_qq: str, topic: str) -> bool:
        """
        添加用户避免的话题
        
        Args:
            user_qq: 用户QQ号
            topic: 避免的话题
            
        Returns:
            bool: 是否添加成功
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                relationship_data = profile.relationship_data
                if not relationship_data:
                    relationship_data = {"target_id": user_qq, "avoid_topics": []}
                
                if "avoid_topics" not in relationship_data:
                    relationship_data["avoid_topics"] = []
                
                if topic not in relationship_data["avoid_topics"]:
                    relationship_data["avoid_topics"].append(topic)
                    profile.relationship_data = relationship_data
                    profile.updated_at = str(time.time())
                    db.commit()
                
                return True
            else:
                # 用户不存在，创建新用户
                relationship = Relationship(target_id=user_qq, avoid_topics=[topic])
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=relationship.model_dump()
                )
                
                db.add(new_profile)
                db.commit()
                return True
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 添加避免话题失败: {str(e)}")
            return False
        finally:
            db.close()
    
    def update_interaction_pattern(self, user_qq: str, pattern_type: str, value: Any) -> bool:
        """
        更新用户的交互模式
        
        Args:
            user_qq: 用户QQ号
            pattern_type: 交互模式类型
            value: 交互模式值
            
        Returns:
            bool: 是否更新成功
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                relationship_data = profile.relationship_data
                if not relationship_data:
                    relationship_data = {"target_id": user_qq, "interaction_patterns": {}}
                
                if "interaction_patterns" not in relationship_data:
                    relationship_data["interaction_patterns"] = {}
                
                relationship_data["interaction_patterns"][pattern_type] = value
                profile.relationship_data = relationship_data
                profile.updated_at = str(time.time())
                db.commit()
                return True
            else:
                # 用户不存在，创建新用户
                relationship = Relationship(target_id=user_qq, interaction_patterns={pattern_type: value})
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=relationship.model_dump()
                )
                
                db.add(new_profile)
                db.commit()
                return True
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 更新交互模式失败: {str(e)}")
            return False
        finally:
            db.close()
    
    def add_sentiment_trend(self, user_qq: str, sentiment: str, intensity: float) -> bool:
        """
        添加用户的情感趋势
        
        Args:
            user_qq: 用户QQ号
            sentiment: 情感类型
            intensity: 情感强度
            
        Returns:
            bool: 是否添加成功
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                relationship_data = profile.relationship_data
                if not relationship_data:
                    relationship_data = {"target_id": user_qq, "sentiment_trends": []}
                
                if "sentiment_trends" not in relationship_data:
                    relationship_data["sentiment_trends"] = []
                
                # 添加情感趋势记录
                sentiment_record = {
                    "timestamp": str(time.time()),
                    "sentiment": sentiment,
                    "intensity": intensity
                }
                relationship_data["sentiment_trends"].append(sentiment_record)
                
                # 只保留最近100条情感记录
                if len(relationship_data["sentiment_trends"]) > 100:
                    relationship_data["sentiment_trends"] = relationship_data["sentiment_trends"][-100:]
                
                profile.relationship_data = relationship_data
                profile.updated_at = str(time.time())
                db.commit()
                return True
            else:
                # 用户不存在，创建新用户
                sentiment_record = {
                    "timestamp": str(time.time()),
                    "sentiment": sentiment,
                    "intensity": intensity
                }
                relationship = Relationship(target_id=user_qq, sentiment_trends=[sentiment_record])
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=relationship.model_dump()
                )
                
                db.add(new_profile)
                db.commit()
                return True
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 添加情感趋势失败: {str(e)}")
            return False
        finally:
            db.close()

    def _migrate_from_json(self):
        """从旧的JSON文件迁移数据到数据库"""
        # 检查迁移是否已经完成
        if os.path.exists(MIGRATION_COMPLETE_FILE):
            logger.info("[RelationDB] 数据迁移已经完成，跳过")
            return
            
        if not os.path.exists(OLD_JSON_DB):
            logger.info("[RelationDB] 没有发现旧的JSON数据库文件，跳过迁移")
            # 创建迁移完成标记，避免下次检查
            try:
                with open(MIGRATION_COMPLETE_FILE, "w") as f:
                    f.write("Migration completed at " + time.strftime("%Y-%m-%d %H:%M:%S"))
            except Exception as e:
                logger.error(f"[RelationDB] 创建迁移标记文件失败: {str(e)}")
            return
            
        try:
            with open(OLD_JSON_DB, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                
            if not old_data:
                logger.info("[RelationDB] 旧的JSON数据库文件为空，跳过迁移")
                return
                
            db = SessionLocal()
            migrated_count = 0
            
            try:
                for user_qq, profile_data in old_data.items():
                    # 检查用户是否已经存在
                    existing = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
                    if existing:
                        continue
                        
                    # 构建新的数据库记录
                    user_profile = UserProfileModel(
                        qq_id=str(user_qq),
                        name=profile_data.get("name", f"User_{user_qq}"),
                        relationship_data=profile_data.get("relationship", {})
                    )
                    db.add(user_profile)
                    migrated_count += 1
                    
                db.commit()
                logger.info(f"[RelationDB] 成功从JSON迁移了 {migrated_count} 条用户数据到数据库")
                
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"[RelationDB] 数据迁移失败: {str(e)}")
            finally:
                db.close()
                
                # 无论是否迁移数据，都创建迁移完成标记
                try:
                    with open(MIGRATION_COMPLETE_FILE, "w") as f:
                        f.write("Migration completed at " + time.strftime("%Y-%m-%d %H:%M:%S"))
                except Exception as e:
                    logger.error(f"[RelationDB] 创建迁移标记文件失败: {str(e)}")
                
        except Exception as e:
            logger.error(f"[RelationDB] 读取旧JSON文件失败: {str(e)}")
            
            # 即使读取失败，也创建迁移标记避免重复尝试
            try:
                with open(MIGRATION_COMPLETE_FILE, "w") as f:
                    f.write("Migration completed at " + time.strftime("%Y-%m-%d %H:%M:%S") + " (with errors)")
            except Exception as create_e:
                logger.error(f"[RelationDB] 创建迁移标记文件失败: {str(create_e)}")

    async def get_user_profile(self, user_qq: str, current_name: str = None) -> UserProfile:
        from app.utils.cache import cached_user_info_get, cached_user_info_set
        
        user_qq = str(user_qq)
        
        # 先检查缓存
        cached_profile = await cached_user_info_get(user_qq)
        if cached_profile:
            # 检查cached_profile是否为字典，如果是则转换为UserProfile对象
            if isinstance(cached_profile, dict):
                # 从字典重建UserProfile对象
                try:
                    # 先提取relationship数据
                    relationship_data = cached_profile.get("relationship", {})
                    if isinstance(relationship_data, dict) and "target_id" not in relationship_data:
                        relationship_data["target_id"] = user_qq
                    
                    cached_profile = UserProfile(
                        name=cached_profile.get("name", f"User_{user_qq}"),
                        qq_id=cached_profile.get("qq_id", user_qq),
                        relationship=Relationship(**relationship_data)
                    )
                except Exception as e:
                    logger.error(f"[RelationDB] 从字典转换UserProfile失败: {str(e)}")
                    # 转换失败时，清除缓存并重新获取
                    await cached_user_info_set(user_qq, None)
                    cached_profile = None
                    # 继续执行后续逻辑，从数据库获取
            
            if cached_profile:
                # 如果用户名有更新，需要同步到数据库和缓存
                if current_name and cached_profile.name != current_name:
                    cached_profile.name = current_name
                db = SessionLocal()
                try:
                    db_profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
                    if db_profile:
                        # 只有当current_name不为None且不为空字符串时才更新用户名
                        if current_name is not None and current_name.strip():
                            db_profile.name = current_name
                            db_profile.updated_at = str(time.time())
                            db.commit()
                            await cached_user_info_set(user_qq, cached_profile)
                except SQLAlchemyError as e:
                    db.rollback()
                    logger.error(f"[RelationDB] 更新用户名失败: {str(e)}")
                finally:
                    db.close()
            return cached_profile
        
        db = SessionLocal()
        
        try:
            # 查询用户
            db_profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if db_profile:
                # 从数据库记录构建UserProfile对象
                relationship_data = db_profile.relationship_data
                if not relationship_data:
                    relationship_data = {"target_id": user_qq}
                
                profile = UserProfile(
                    name=db_profile.name,
                    qq_id=db_profile.qq_id,
                    relationship=Relationship(**relationship_data)
                )
                
                # 更新用户名
                if current_name is not None and current_name.strip() and profile.name != current_name:
                    db_profile.name = current_name
                    db_profile.updated_at = str(time.time())
                    db.commit()
                    profile.name = current_name
                
                # 存入缓存
                await cached_user_info_set(user_qq, profile)
                return profile
            else:
                # 创建新用户
                display_name = current_name if current_name else f"User_{user_qq}"
                relationship = Relationship(target_id=user_qq)
                
                new_db_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=display_name,
                    relationship_data=relationship.model_dump()
                )
                
                db.add(new_db_profile)
                db.commit()
                
                profile = UserProfile(
                    name=display_name,
                    qq_id=user_qq,
                    relationship=relationship
                )
                
                # 存入缓存
                await cached_user_info_set(user_qq, profile)
                return profile
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 获取用户资料失败: {str(e)}")
            # 出错时返回默认值
            display_name = current_name if current_name else f"User_{user_qq}"
            profile = UserProfile(
                name=display_name,
                qq_id=user_qq,
                relationship=Relationship(target_id=user_qq)
            )
            # 存入缓存
            await cached_user_info_set(user_qq, profile)
            return profile
        finally:
            db.close()

    def update_intimacy(self, user_qq: str, delta: int):
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                relationship_data = profile.relationship_data
                if not relationship_data:
                    relationship_data = {"target_id": user_qq, "intimacy": 60}
                
                # 更新亲密度
                current_intimacy = relationship_data.get("intimacy", 60)
                new_intimacy = max(0, min(100, current_intimacy + delta))
                relationship_data["intimacy"] = new_intimacy
                
                profile.relationship_data = relationship_data
                profile.updated_at = str(time.time())
                db.commit()
                
                return new_intimacy
            else:
                # 用户不存在，创建新用户
                relationship = Relationship(target_id=user_qq, intimacy=60 + delta)
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=relationship.model_dump()
                )
                
                db.add(new_profile)
                db.commit()
                
                return relationship.intimacy
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 更新亲密度失败: {str(e)}")
            return 60  # 出错时返回默认值
        finally:
            db.close()

    def update_relationship_dimensions(self, user_qq: str, deltas: Dict[str, int]):
        """
        更新关系的多个维度（好感度、熟悉度、信任度、兴趣匹配等）
        :param user_qq: 用户QQ号
        :param deltas: 包含各个维度变化值的字典，例如：{"intimacy": 2, "familiarity": 1}
        :return: 更新后的关系维度字典
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                relationship_data = profile.relationship_data
                if not relationship_data:
                    relationship_data = {
                        "target_id": user_qq,
                        "intimacy": 60,
                        "familiarity": 50,
                        "trust": 50,
                        "interest_match": 50
                    }
                
                # 确保所有维度都有默认值
                for dimension in ["intimacy", "familiarity", "trust", "interest_match"]:
                    if dimension not in relationship_data:
                        if dimension == "intimacy":
                            relationship_data[dimension] = 60
                        else:
                            relationship_data[dimension] = 50
                
                # 更新各个维度
                updated_dimensions = {}
                for dimension, delta in deltas.items():
                    if dimension in ["intimacy", "familiarity", "trust", "interest_match"]:
                        current_value = relationship_data.get(dimension, 50)
                        new_value = max(0, min(100, current_value + delta))
                        relationship_data[dimension] = new_value
                        updated_dimensions[dimension] = new_value
                
                profile.relationship_data = relationship_data
                profile.updated_at = str(time.time())
                db.commit()
                
                return updated_dimensions
            else:
                # 用户不存在，创建新用户
                relationship_data = {
                    "target_id": user_qq,
                    "intimacy": 60,
                    "familiarity": 50,
                    "trust": 50,
                    "interest_match": 50
                }
                
                # 应用变化值
                updated_dimensions = {}
                for dimension, delta in deltas.items():
                    if dimension in ["intimacy", "familiarity", "trust", "interest_match"]:
                        new_value = max(0, min(100, relationship_data[dimension] + delta))
                        relationship_data[dimension] = new_value
                        updated_dimensions[dimension] = new_value
                
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=relationship_data
                )
                
                db.add(new_profile)
                db.commit()
                
                return updated_dimensions
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 更新关系维度失败: {str(e)}")
            return {}
        finally:
            db.close()

    def update_relationship(self, user_qq: str, target_id: str, new_data: Relationship):
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                profile.relationship_data = new_data.model_dump()
                profile.updated_at = str(time.time())
                db.commit()
                return True
            else:
                # 用户不存在，创建新用户
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=new_data.model_dump()
                )
                
                db.add(new_profile)
                db.commit()
                return True
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 更新关系失败: {str(e)}")
            return False
        finally:
            db.close()

    def add_memory_point(self, user_qq: str, category: str, content: str, weight: float = 1.0) -> bool:
        """
        添加记忆点到用户关系中
        
        Args:
            user_qq: 用户QQ号
            category: 记忆分类
            content: 记忆内容
            weight: 记忆权重
            
        Returns:
            bool: 是否添加成功
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                relationship_data = profile.relationship_data
                if not relationship_data:
                    relationship_data = {
                        "target_id": user_qq,
                        "intimacy": 60,
                        "familiarity": 50,
                        "trust": 50,
                        "interest_match": 50,
                        "memory_points": [],
                        "expression_habits": []
                    }
                
                # 确保memory_points存在
                if "memory_points" not in relationship_data:
                    relationship_data["memory_points"] = []
                
                # 创建记忆点
                memory_point = f"{category}:{content}:{weight}"
                relationship_data["memory_points"].append(memory_point)
                
                profile.relationship_data = relationship_data
                profile.updated_at = str(time.time())
                db.commit()
                return True
            else:
                # 用户不存在，创建新用户
                relationship_data = {
                    "target_id": user_qq,
                    "intimacy": 60,
                    "familiarity": 50,
                    "trust": 50,
                    "interest_match": 50,
                    "memory_points": [f"{category}:{content}:{weight}"],
                    "expression_habits": []
                }
                
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=relationship_data
                )
                
                db.add(new_profile)
                db.commit()
                return True
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 添加记忆点失败: {str(e)}")
            return False
        finally:
            db.close()

    def add_expression_habit(self, user_qq: str, habit: str) -> bool:
        """
        添加表达习惯到用户关系中
        
        Args:
            user_qq: 用户QQ号
            habit: 表达习惯内容
            
        Returns:
            bool: 是否添加成功
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                relationship_data = profile.relationship_data
                if not relationship_data:
                    relationship_data = {
                        "target_id": user_qq,
                        "intimacy": 60,
                        "familiarity": 50,
                        "trust": 50,
                        "interest_match": 50,
                        "memory_points": [],
                        "expression_habits": []
                    }
                
                # 确保expression_habits存在
                if "expression_habits" not in relationship_data:
                    relationship_data["expression_habits"] = []
                
                # 添加表达习惯
                relationship_data["expression_habits"].append(habit)
                
                profile.relationship_data = relationship_data
                profile.updated_at = str(time.time())
                db.commit()
                return True
            else:
                # 用户不存在，创建新用户
                relationship_data = {
                    "target_id": user_qq,
                    "intimacy": 60,
                    "familiarity": 50,
                    "trust": 50,
                    "interest_match": 50,
                    "memory_points": [],
                    "expression_habits": [habit]
                }
                
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=relationship_data
                )
                
                db.add(new_profile)
                db.commit()
                return True
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 添加表达习惯失败: {str(e)}")
            return False
        finally:
            db.close()

    def get_memory_points_by_category(self, user_qq: str, category: str) -> List[str]:
        """
        获取用户指定分类的记忆点
        
        Args:
            user_qq: 用户QQ号
            category: 记忆分类
            
        Returns:
            List[str]: 记忆点列表
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile and profile.relationship_data:
                memory_points = profile.relationship_data.get("memory_points", [])
                return [mp for mp in memory_points if mp.startswith(f"{category}:")]
            return []
            
        except SQLAlchemyError as e:
            logger.error(f"[RelationDB] 获取记忆点失败: {str(e)}")
            return []
        finally:
            db.close()

    def get_random_memory_points(self, user_qq: str, category: str = None, num: int = 3) -> List[str]:
        """
        获取用户随机的记忆点
        
        Args:
            user_qq: 用户QQ号
            category: 记忆分类（可选）
            num: 获取数量
            
        Returns:
            List[str]: 随机记忆点列表
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile and profile.relationship_data:
                memory_points = profile.relationship_data.get("memory_points", [])
                
                if category:
                    memory_points = [mp for mp in memory_points if mp.startswith(f"{category}:")]
                
                if not memory_points:
                    return []
                
                # 随机选择记忆点
                return random.sample(memory_points, min(num, len(memory_points)))
            return []
            
        except SQLAlchemyError as e:
            logger.error(f"[RelationDB] 获取随机记忆点失败: {str(e)}")
            return []
        finally:
            db.close()
    
    def get_all_memory_categories(self, user_qq: str) -> List[str]:
        """
        获取用户所有记忆点分类
        
        Args:
            user_qq: 用户QQ号
            
        Returns:
            List[str]: 记忆分类列表
        """
        user_qq = str(user_qq)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile and profile.relationship_data:
                memory_points = profile.relationship_data.get("memory_points", [])
                categories = set()
                for mp in memory_points:
                    parts = mp.split(":", 1)
                    if len(parts) > 1:
                        categories.add(parts[0].strip())
                return list(categories)
            return []
            
        except SQLAlchemyError as e:
            logger.error(f"[RelationDB] 获取记忆分类失败: {str(e)}")
            return []
        finally:
            db.close()
    
    def get_memory_content(self, memory_point: str) -> str:
        """
        从记忆点中提取记忆内容
        
        Args:
            memory_point: 记忆点字符串，格式：category:content:weight
            
        Returns:
            str: 记忆内容
        """
        if not isinstance(memory_point, str):
            return ""
        parts = memory_point.split(":")
        return ":".join(parts[1:-1]).strip() if len(parts) > 2 else ""
    
    def get_memory_weight(self, memory_point: str) -> float:
        """
        从记忆点中提取记忆权重
        
        Args:
            memory_point: 记忆点字符串，格式：category:content:weight
            
        Returns:
            float: 记忆权重
        """
        if not isinstance(memory_point, str):
            return 1.0
        parts = memory_point.rsplit(":", 1)
        if len(parts) <= 1:
            return 1.0
        try:
            return float(parts[-1].strip())
        except Exception:
            return 1.0
    
    def add_group_nickname(self, user_qq: str, group_id: str, nickname: str) -> bool:
        """
        添加或更新用户在指定群的昵称
        
        Args:
            user_qq: 用户QQ号
            group_id: 群号
            nickname: 群昵称
            
        Returns:
            bool: 是否添加成功
        """
        user_qq = str(user_qq)
        group_id = str(group_id)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile:
                relationship_data = profile.relationship_data
                if not relationship_data:
                    relationship_data = {
                        "target_id": user_qq,
                        "intimacy": 60,
                        "familiarity": 50,
                        "trust": 50,
                        "interest_match": 50,
                        "memory_points": [],
                        "expression_habits": [],
                        "group_nicknames": []
                    }
                
                # 确保group_nicknames存在
                if "group_nicknames" not in relationship_data:
                    relationship_data["group_nicknames"] = []
                
                # 查找群昵称记录
                group_nicknames = relationship_data["group_nicknames"]
                updated = False
                for item in group_nicknames:
                    if item.get("group_id") == group_id:
                        item["nickname"] = nickname
                        updated = True
                        break
                
                # 如果不存在则添加新记录
                if not updated:
                    group_nicknames.append({
                        "group_id": group_id,
                        "nickname": nickname,
                        "updated_at": str(time.time())
                    })
                
                profile.relationship_data = relationship_data
                profile.updated_at = str(time.time())
                db.commit()
                return True
            else:
                # 用户不存在，创建新用户
                relationship = Relationship(
                    target_id=user_qq,
                    intimacy=60,
                    familiarity=50,
                    trust=50,
                    interest_match=50,
                    memory_points=[],
                    expression_habits=[],
                    group_nicknames=[{
                        "group_id": group_id,
                        "nickname": nickname,
                        "updated_at": str(time.time())
                    }]
                )
                new_profile = UserProfileModel(
                    qq_id=user_qq,
                    name=f"User_{user_qq}",
                    relationship_data=relationship.model_dump()
                )
                
                db.add(new_profile)
                db.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"[RelationDB] 添加群昵称失败: {str(e)}")
            return False
        finally:
            db.close()
    
    def get_group_nickname(self, user_qq: str, group_id: str) -> str:
        """
        获取用户在指定群的昵称
        
        Args:
            user_qq: 用户QQ号
            group_id: 群号
            
        Returns:
            str: 群昵称，如果不存在则返回空字符串
        """
        user_qq = str(user_qq)
        group_id = str(group_id)
        db = SessionLocal()
        
        try:
            profile = db.query(UserProfileModel).filter(UserProfileModel.qq_id == user_qq).first()
            
            if profile and profile.relationship_data:
                relationship_data = profile.relationship_data
                group_nicknames = relationship_data.get("group_nicknames", [])
                for item in group_nicknames:
                    if item.get("group_id") == group_id:
                        return item.get("nickname", "")
            return ""
            
        except SQLAlchemyError as e:
            logger.error(f"[RelationDB] 获取群昵称失败: {str(e)}")
            return ""
        finally:
            db.close()


# 创建全局实例
relation_db = GlobalRelationDB()
