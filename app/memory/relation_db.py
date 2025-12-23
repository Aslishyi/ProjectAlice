# === 修改文件: app/memory/relation_db.py ===

import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Union
from pydantic import BaseModel, Field

DB_FILE = "data/user_profiles.json"
_executor = ThreadPoolExecutor(max_workers=1)  # 专门用于文件写入的单线程池


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


class GlobalRelationDB:
    def __init__(self):
        self.db_path = DB_FILE
        self._ensure_db_exists()
        self.data: Dict[str, Union[Dict, UserProfile]] = self._load_db()

    def _ensure_db_exists(self):
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

    def _load_db(self) -> Dict[str, Any]:
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    # 🚀 优化点：将保存操作改为非阻塞 (Fire-and-forget)
    # 实际上为了数据安全，我们可以放到 executor 里跑
    def _save_db_sync(self):
        """同步保存逻辑，供 Executor 调用"""
        try:
            saveable = {}
            for uid, profile in self.data.items():
                if hasattr(profile, "model_dump"):
                    saveable[uid] = profile.model_dump()
                else:
                    saveable[uid] = profile

            # 使用原子写入防止损坏：写临时文件 -> 重命名
            temp_path = self.db_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(saveable, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.db_path)
        except Exception as e:
            print(f"❌ [RelationDB] Save error: {e}")

    def _trigger_save(self):
        """触发异步保存"""
        # 获取当前的 event loop，如果在 loop 中则 await run_in_executor
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(_executor, self._save_db_sync)
        except RuntimeError:
            # 如果没有 loop (比如初始化时)，同步跑
            self._save_db_sync()

    def get_user_profile(self, user_qq: str, current_name: str = None) -> UserProfile:
        user_qq = str(user_qq)
        if user_qq in self.data:
            entry = self.data[user_qq]
            profile = None
            if isinstance(entry, dict):
                if "qq_id" not in entry: entry["qq_id"] = user_qq
                profile = UserProfile(**entry)
                self.data[user_qq] = profile
            elif isinstance(entry, UserProfile):
                profile = entry
                if not profile.qq_id: profile.qq_id = user_qq
            else:
                profile = UserProfile(name=current_name or f"User_{user_qq}", qq_id=user_qq,
                                      relationship=Relationship(target_id=user_qq))

            if current_name and profile.name != current_name:
                profile.name = current_name
                self.data[user_qq] = profile
                self._trigger_save()  # 异步保存
            return profile

        display_name = current_name if current_name else f"User_{user_qq}"
        new_profile = UserProfile(name=display_name, qq_id=user_qq,
                                  relationship=Relationship(target_id=user_qq, intimacy=60))
        self.data[user_qq] = new_profile
        self._trigger_save()  # 异步保存
        return new_profile

    def update_intimacy(self, user_qq: str, delta: int):
        profile = self.get_user_profile(user_qq)
        current = profile.relationship.intimacy
        new_val = max(0, min(100, current + delta))
        profile.relationship.intimacy = new_val
        self.data[user_qq] = profile
        self._trigger_save()  # 异步保存
        return new_val

    # ... (其他 update 方法同理，替换 _save_db 为 _trigger_save) ...
    def update_relationship(self, user_qq: str, target_id: str, new_data: Relationship):
        profile = self.get_user_profile(user_qq)
        # ... (逻辑保持不变) ...
        # ...
        self.data[user_qq] = profile
        self._trigger_save()


relation_db = GlobalRelationDB()
