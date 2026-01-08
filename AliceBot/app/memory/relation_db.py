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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JSON文件路径（用于数据迁移）
OLD_JSON_DB = "data/user_profiles.json"
# 迁移完成标记文件
MIGRATION_COMPLETE_FILE = "data/migration_complete.txt"


class Relationship(BaseModel):
    target_id: str
    relation_type: str = "acquaintance"
    intimacy: int = Field(default=60, ge=0, le=100)
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

    def get_user_profile(self, user_qq: str, current_name: str = None) -> UserProfile:
        user_qq = str(user_qq)
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
                if current_name and profile.name != current_name:
                    db_profile.name = current_name
                    db_profile.updated_at = str(time.time())
                    db.commit()
                    profile.name = current_name
                
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
                
                return UserProfile(
                    name=display_name,
                    qq_id=user_qq,
                    relationship=relationship
                )
                
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"[RelationDB] 获取用户资料失败: {str(e)}")
            # 出错时返回默认值
            display_name = current_name if current_name else f"User_{user_qq}"
            return UserProfile(
                name=display_name,
                qq_id=user_qq,
                relationship=Relationship(target_id=user_qq)
            )
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
