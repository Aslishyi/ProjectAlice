# === 修改文件: app/memory/relation_db.py ===

import json
import os
import asyncio
import logging
import time
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


# 创建全局实例
relation_db = GlobalRelationDB()
