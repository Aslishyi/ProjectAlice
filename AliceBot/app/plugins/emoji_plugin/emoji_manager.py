import os
import json
import base64
import hashlib
import logging
import random
import httpx
import io
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

logger = logging.getLogger("EmojiManager")


class EmojiInfo:
    """表情包信息类"""
    def __init__(self, emoji_hash: str, base64_data: str, file_path: str = "", description: str = "", emotions: List[str] = None, tags: List[str] = None, category: str = "general"):
        self.emoji_hash = emoji_hash  # 表情包的哈希值，用于唯一标识
        self.base64_data = base64_data  # 表情包的base64编码数据
        self.file_path = file_path  # 表情包的本地文件路径
        self.description = description  # 表情包的描述
        self.emotions = emotions or []  # 与表情包相关的情绪标签
        self.tags = tags or []  # 表情包的自定义标签
        self.category = category  # 表情包的分类
        self.created_at = datetime.now().isoformat()  # 创建时间
        self.usage_count = 0  # 使用次数
        self.last_used_at = None  # 最后使用时间

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "emoji_hash": self.emoji_hash,
            "base64_data": self.base64_data,
            "file_path": self.file_path,
            "description": self.description,
            "emotions": self.emotions,
            "tags": self.tags,
            "category": self.category,
            "created_at": self.created_at,
            "usage_count": self.usage_count,
            "last_used_at": self.last_used_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmojiInfo":
        """从字典创建EmojiInfo对象"""
        emoji = cls(
            data["emoji_hash"],
            data["base64_data"],
            data.get("file_path", ""),
            data.get("description", ""),
            data.get("emotions", []),
            data.get("tags", []),
            data.get("category", "general")
        )
        emoji.created_at = data.get("created_at", datetime.now().isoformat())
        emoji.usage_count = data.get("usage_count", 0)
        emoji.last_used_at = data.get("last_used_at", None)
        return emoji



    def increment_usage(self) -> None:
        """增加使用次数并更新最后使用时间"""
        self.usage_count += 1
        self.last_used_at = datetime.now().isoformat()


class EmojiManager:
    """表情包管理器"""
    
    def __init__(self, data_dir: str = "emoji_data"):
        """初始化表情包管理器
        
        Args:
            data_dir: 表情包数据存储目录
        """
        # 确保data_dir是绝对路径
        if not os.path.isabs(data_dir):
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建相对于项目根目录的绝对路径
            self.data_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", data_dir))
        else:
            self.data_dir = data_dir
            
        self.emoji_db_path = os.path.join(self.data_dir, "emoji_db.json")
        self.emoji_images_dir = os.path.join(self.data_dir, "images")  # 表情包图片存储目录
        self.emojis: Dict[str, EmojiInfo] = {}  # emoji_hash -> EmojiInfo
        
        # 确保数据目录和图片目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.emoji_images_dir, exist_ok=True)
        
        # 加载已有的表情包
        self._load_emojis()
        
        # 压缩现有表情包（仅在初始化时执行一次）
        self.compress_existing_emojis()
        
        logger.info(f"表情包管理器初始化完成，加载了 {len(self.emojis)} 个表情包")
    
    def _load_emojis(self) -> None:
        """加载表情包数据"""
        if os.path.exists(self.emoji_db_path):
            try:
                with open(self.emoji_db_path, "r", encoding="utf-8") as f:
                    emoji_data = json.load(f)
                    
                for emoji_hash, data in emoji_data.items():
                    self.emojis[emoji_hash] = EmojiInfo.from_dict(data)
                    
                logger.info(f"从 {self.emoji_db_path} 加载了 {len(self.emojis)} 个表情包")
                
                # 检查是否需要为旧的表情包数据生成file_path字段
                self._update_old_emojis()
                
            except Exception as e:
                logger.error(f"加载表情包数据失败: {e}")
                self.emojis = {}
    
    def _update_old_emojis(self) -> None:
        """为旧的表情包数据生成file_path字段"""
        try:
            updated = False
            for emoji_hash, emoji in self.emojis.items():
                # 如果file_path字段为空，并且有base64_data，则生成file_path
                if not emoji.file_path and emoji.base64_data:
                    file_path = self._save_image_to_file(emoji.base64_data, emoji_hash)
                    if file_path:
                        emoji.file_path = file_path
                        updated = True
                        logger.info(f"为旧表情包生成文件路径: {emoji_hash} -> {file_path}")
                # 如果file_path字段是相对路径，将其转换为绝对路径
                elif emoji.file_path and not os.path.isabs(emoji.file_path):
                    # 将相对路径转换为绝对路径
                    absolute_path = os.path.join(self.emoji_images_dir, os.path.basename(emoji.file_path))
                    # 检查文件是否存在
                    if os.path.exists(absolute_path):
                        emoji.file_path = absolute_path
                        updated = True
                        logger.info(f"将旧表情包的相对路径转换为绝对路径: {emoji_hash} -> {absolute_path}")
                    else:
                        # 如果文件不存在，重新生成
                        file_path = self._save_image_to_file(emoji.base64_data, emoji_hash)
                        if file_path:
                            emoji.file_path = file_path
                            updated = True
                            logger.info(f"重新生成旧表情包的绝对路径: {emoji_hash} -> {file_path}")
            
            # 如果有更新，保存表情包数据
            if updated:
                self._save_emojis()
                logger.info("已更新旧表情包的文件路径信息")
                
        except Exception as e:
            logger.error(f"更新旧表情包数据失败: {e}")
    
    def compress_existing_emojis(self) -> None:
        """批量压缩现有的表情包图片"""
        try:
            logger.info("开始批量压缩现有的表情包图片...")
            updated_count = 0
            
            for emoji_hash, emoji in self.emojis.items():
                if emoji.file_path and os.path.exists(emoji.file_path):
                    file_ext = os.path.splitext(emoji.file_path)[1].lower()
                    
                    # 跳过GIF图片和已经处理过的图片
                    if file_ext != '.gif':
                        # 读取现有图片
                        with open(emoji.file_path, 'rb') as f:
                            image_bytes = f.read()
                        
                        # 转换为base64
                        base64_data = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # 删除旧文件
                        os.remove(emoji.file_path)
                        
                        # 重新保存（会自动压缩）
                        new_file_path = self._save_image_to_file(base64_data, emoji_hash)
                        if new_file_path:
                            emoji.file_path = new_file_path
                            updated_count += 1
                            logger.info(f"已压缩表情包: {emoji_hash}")
            
            # 如果有更新，保存表情包数据
            if updated_count > 0:
                self._save_emojis()
                logger.info(f"批量压缩完成，共处理了 {updated_count} 个表情包")
            else:
                logger.info("没有需要压缩的表情包")
                
        except Exception as e:
            logger.error(f"批量压缩表情包失败: {e}")
    
    def _save_emojis(self) -> bool:
        """保存表情包数据"""
        try:
            emoji_data = {emoji_hash: emoji.to_dict() for emoji_hash, emoji in self.emojis.items()}
            
            with open(self.emoji_db_path, "w", encoding="utf-8") as f:
                json.dump(emoji_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"表情包数据已保存到 {self.emoji_db_path}")
            return True
        except Exception as e:
            logger.error(f"保存表情包数据失败: {e}")
            return False
    
    def download_image_to_base64(self, image_url: str) -> Optional[str]:
        """从URL下载图片并转换为base64编码
        
        Args:
            image_url: 图片的URL地址
            
        Returns:
            Optional[str]: base64编码的图片数据，失败返回None
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(image_url)
                if response.status_code != 200:
                    logger.error(f"下载图片失败: HTTP {response.status_code}, URL: {image_url}")
                    return None
                
                image_bytes = response.content
                base64_data = base64.b64encode(image_bytes).decode('utf-8')
                logger.info(f"成功下载并转换图片为base64, URL: {image_url}")
                return base64_data
        except Exception as e:
            logger.error(f"下载图片失败: {e}, URL: {image_url}")
            return None
    
    def analyze_image_emotions(self, base64_data: str, description: str = "") -> List[str]:
        """分析图片的情绪标签，现在仅作为兼容接口，实际使用大模型分析
        
        Args:
            base64_data: 图片的base64编码数据
            description: 图片的描述文本
            
        Returns:
            List[str]: 情绪标签列表
        """
        logger.warning("analyze_image_emotions: 此方法已废弃，建议使用perception.py中的大模型分析功能")
        return ["未知"]
    
    # 旧的情绪分析辅助方法已删除，建议使用perception.py中的大模型分析功能
    
    def calculate_emoji_hash(self, base64_data: str) -> str:
        """计算表情包的哈希值
        
        Args:
            base64_data: 表情包的base64编码数据
            
        Returns:
            str: 表情包的哈希值
        """
        try:
            # 清理base64数据
            if isinstance(base64_data, str):
                base64_clean = base64_data.encode("ascii", errors="ignore").decode("ascii")
            else:
                base64_clean = str(base64_data)
            
            # 解码并计算哈希
            image_bytes = base64.b64decode(base64_clean)
            emoji_hash = hashlib.md5(image_bytes).hexdigest()
            return emoji_hash
        except Exception as e:
            logger.error(f"计算表情包哈希值失败: {e}")
            return ""
    
    def _save_image_to_file(self, base64_data: str, emoji_hash: str) -> str:
        """将图片保存到本地文件并进行压缩和调整大小
        
        Args:
            base64_data: 图片的base64编码数据
            emoji_hash: 表情包的哈希值
            
        Returns:
            str: 保存后的文件路径，失败返回空字符串
        """
        try:
            # 清理base64数据
            if isinstance(base64_data, str):
                base64_clean = base64_data.encode("ascii", errors="ignore").decode("ascii")
            else:
                base64_clean = str(base64_data)
            
            # 解码图片
            image_bytes = base64.b64decode(base64_clean)
            
            # 确定文件格式
            # 尝试从文件头识别格式
            if image_bytes.startswith(b'\xff\xd8\xff'):
                ext = '.jpg'
            elif image_bytes.startswith(b'\x89PNG'):
                ext = '.png'
            elif image_bytes.startswith(b'GIF8'):
                ext = '.gif'
            else:
                # 默认使用jpg格式
                ext = '.jpg'
            
            # 生成文件名
            filename = f"{emoji_hash}{ext}"
            file_path = os.path.join(self.emoji_images_dir, filename)
            
            # 处理GIF图片（调整大小并压缩）
            if ext == '.gif':
                # 保持GIF动画特性的同时调整大小
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # 调整大小，最大边长为512像素
                    max_size = 512
                    if image.width > max_size or image.height > max_size:
                        width, height = image.size
                        if width > height:
                            new_width = max_size
                            new_height = int((height / width) * max_size)
                        else:
                            new_height = max_size
                            new_width = int((width / height) * max_size)
                        
                        # 创建调整大小后的GIF
                        frames = []
                        durations = []
                        
                        try:
                            # 遍历GIF的所有帧
                            while True:
                                # 调整当前帧的大小
                                resized_frame = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                frames.append(resized_frame.convert('RGBA'))
                                
                                # 记录帧持续时间
                                durations.append(image.info.get('duration', 100))
                                
                                # 移动到下一帧
                                image.seek(image.tell() + 1)
                        except EOFError:
                            pass  # 所有帧处理完毕
                        
                        # 保存调整大小后的GIF
                        if frames:
                            frames[0].save(
                                file_path,
                                format='GIF',
                                save_all=True,
                                append_images=frames[1:],
                                duration=durations,
                                loop=0,  # 无限循环
                                optimize=True,  # 启用优化
                                disposal=2  # 每帧显示后恢复背景
                            )
                            logger.info(f"GIF图片已调整大小并保存: {file_path}")
                        else:
                            # 如果无法处理帧，保存原始图片
                            with open(file_path, 'wb') as f:
                                f.write(image_bytes)
                    else:
                        # 尺寸合适，直接保存但启用优化
                        image = Image.open(io.BytesIO(image_bytes))
                        frames = []
                        durations = []
                        
                        try:
                            while True:
                                frames.append(image.convert('RGBA'))
                                durations.append(image.info.get('duration', 100))
                                image.seek(image.tell() + 1)
                        except EOFError:
                            pass
                        
                        if frames:
                            frames[0].save(
                                file_path,
                                format='GIF',
                                save_all=True,
                                append_images=frames[1:],
                                duration=durations,
                                loop=0,
                                optimize=True,
                                disposal=2
                            )
                            logger.info(f"GIF图片已优化并保存: {file_path}")
                        else:
                            with open(file_path, 'wb') as f:
                                f.write(image_bytes)
                except Exception as gif_err:
                    logger.error(f"处理GIF图片时发生错误: {gif_err}")
                    # 出错时保存原始图片
                    with open(file_path, 'wb') as f:
                        f.write(image_bytes)
            else:
                # 对JPG和PNG进行压缩和调整大小
                image_io = io.BytesIO(image_bytes)
                image = Image.open(image_io)
                
                # 调整大小，最大边长为512像素
                max_size = 512
                if image.width > max_size or image.height > max_size:
                    width, height = image.size
                    if width > height:
                        new_width = max_size
                        new_height = int((height / width) * max_size)
                    else:
                        new_height = max_size
                        new_width = int((width / height) * max_size)
                    
                    # 使用高质量的图像缩放算法
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 保存图片并压缩
                if ext == '.jpg':
                    # JPG压缩质量设置为65，进一步减小文件大小
                    image.save(file_path, format='JPEG', quality=65, optimize=True)
                else:
                    # PNG压缩，启用更高的压缩级别
                    image.save(file_path, format='PNG', optimize=True, compress_level=9)  # 最高压缩级别
            
            logger.info(f"图片已保存到本地并压缩: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"保存图片到本地失败: {e}")
            return ""
    
    def add_emoji(self, base64_data: str, description: str = "", emotions: List[str] = None, tags: List[str] = None, category: str = "general") -> Tuple[bool, str, Optional[EmojiInfo]]:
        """添加表情包
        
        Args:
            base64_data: 表情包的base64编码数据
            description: 表情包的描述
            emotions: 与表情包相关的情绪标签
            tags: 表情包的自定义标签
            category: 表情包的分类
            
        Returns:
            Tuple[bool, str, Optional[EmojiInfo]]: (是否成功, 消息, 表情包信息)
        """
        try:
            # 计算哈希值
            emoji_hash = self.calculate_emoji_hash(base64_data)
            if not emoji_hash:
                return False, "无法计算表情包哈希值", None
            
            # 检查是否已存在
            if emoji_hash in self.emojis:
                # 更新现有表情包
                existing_emoji = self.emojis[emoji_hash]
                existing_emoji.description = description or existing_emoji.description
                # 限制情绪标签最多1个
                existing_emoji.emotions = emotions[:1] if emotions else existing_emoji.emotions
                existing_emoji.tags = tags or existing_emoji.tags
                
                if self._save_emojis():
                    return True, "表情包已更新", existing_emoji
                else:
                    return False, "更新表情包失败", None
            
            # 将图片保存到本地文件
            file_path = self._save_image_to_file(base64_data, emoji_hash)
            if not file_path:
                return False, "保存图片文件失败", None
            
            # 创建新的表情包信息，限制情绪标签最多1个
            emoji = EmojiInfo(emoji_hash, base64_data, file_path, description, emotions[:1] if emotions else [], tags, category)
            self.emojis[emoji_hash] = emoji
            
            # 保存数据
            if self._save_emojis():
                logger.info(f"添加新表情包: {description or '无描述'}, 哈希值: {emoji_hash}, 文件路径: {file_path}")
                return True, "表情包添加成功", emoji
            else:
                # 保存失败，回滚
                del self.emojis[emoji_hash]
                # 删除已保存的图片文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                return False, "添加表情包失败", None
                
        except Exception as e:
            logger.error(f"添加表情包失败: {e}")
            return False, f"添加表情包时发生错误: {str(e)}", None
    
    def add_emoji_from_url(self, image_url: str, description: str = "", emotions: List[str] = None, tags: List[str] = None, category: str = "general") -> Tuple[bool, str, Optional[EmojiInfo]]:
        """从图片URL添加表情包
        
        Args:
            image_url: 图片的URL地址
            description: 表情包的描述
            emotions: 与表情包相关的情绪标签，不提供则自动分析
            tags: 表情包的自定义标签
            category: 表情包的分类
            
        Returns:
            Tuple[bool, str, Optional[EmojiInfo]]: (是否成功, 消息, 表情包信息)
        """
        try:
            # 下载图片并转换为base64
            base64_data = self.download_image_to_base64(image_url)
            if not base64_data:
                return False, "无法下载图片", None
            
            # 如果没有提供情绪标签、描述或分类，使用默认分析
            # 注意：这里仍然使用本地分析，因为LLM分析需要异步调用
            # 在perception.py中会使用LLM分析
            if not emotions:
                emotions = self.analyze_image_emotions(base64_data, description)
            
            # 添加表情包
            return self.add_emoji(base64_data, description, emotions, tags, category)
            
        except Exception as e:
            logger.error(f"从URL添加表情包失败: {e}")
            return False, f"从URL添加表情包时发生错误: {str(e)}", None
    
    def delete_emoji(self, emoji_hash: str) -> Tuple[bool, str, Optional[EmojiInfo]]:
        """删除表情包
        
        Args:
            emoji_hash: 表情包的哈希值
            
        Returns:
            Tuple[bool, str, Optional[EmojiInfo]]: (是否成功, 消息, 被删除的表情包信息)
        """
        try:
            if emoji_hash not in self.emojis:
                return False, "表情包不存在", None
            
            # 获取要删除的表情包信息
            deleted_emoji = self.emojis[emoji_hash]
            
            # 记录文件路径，用于后续删除
            file_path = deleted_emoji.file_path
            
            # 删除表情包
            del self.emojis[emoji_hash]
            
            # 保存数据
            if self._save_emojis():
                # 删除本地图片文件
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"删除表情包图片文件: {file_path}")
                    except Exception as e:
                        logger.error(f"删除表情包图片文件失败: {e}")
                        # 文件删除失败不影响表情包数据删除的结果
                
                logger.info(f"删除表情包: {deleted_emoji.description or '无描述'}, 哈希值: {emoji_hash}")
                return True, "表情包删除成功", deleted_emoji
            else:
                # 保存失败，回滚
                self.emojis[emoji_hash] = deleted_emoji
                return False, "删除表情包失败", None
                
        except Exception as e:
            logger.error(f"删除表情包失败: {e}")
            return False, f"删除表情包时发生错误: {str(e)}", None
    
    def get_emoji(self, emoji_hash: str) -> Optional[EmojiInfo]:
        """根据哈希值获取表情包
        
        Args:
            emoji_hash: 表情包的哈希值
            
        Returns:
            Optional[EmojiInfo]: 表情包信息或None
        """
        return self.emojis.get(emoji_hash)
    
    def get_all_emojis(self) -> List[EmojiInfo]:
        """获取所有表情包
        
        Returns:
            List[EmojiInfo]: 所有表情包的列表
        """
        return list(self.emojis.values())
    
    def get_emojis_by_emotion(self, emotion: str) -> List[EmojiInfo]:
        """根据情绪标签获取表情包
        
        Args:
            emotion: 情绪标签
            
        Returns:
            List[EmojiInfo]: 符合条件的表情包列表
        """
        return [emoji for emoji in self.emojis.values() if emotion.lower() in [e.lower() for e in emoji.emotions]]
    
    def get_emojis_by_tag(self, tag: str) -> List[EmojiInfo]:
        """根据自定义标签获取表情包
        
        Args:
            tag: 自定义标签
            
        Returns:
            List[EmojiInfo]: 符合条件的表情包列表
        """
        return [emoji for emoji in self.emojis.values() if tag.lower() in [t.lower() for t in emoji.tags]]
    
    def get_emojis_by_category(self, category: str) -> List[EmojiInfo]:
        """根据分类获取表情包
        
        Args:
            category: 表情包分类
            
        Returns:
            List[EmojiInfo]: 符合条件的表情包列表
        """
        return [emoji for emoji in self.emojis.values() if emoji.category.lower() == category.lower()]
    
    def search_emojis(self, keyword: str) -> List[EmojiInfo]:
        """根据关键词搜索表情包
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            List[EmojiInfo]: 搜索结果列表
        """
        results = []
        keyword_lower = keyword.lower()
        
        for emoji in self.emojis.values():
            # 搜索描述
            if keyword_lower in emoji.description.lower():
                results.append(emoji)
                continue
            
            # 搜索情绪标签
            if any(keyword_lower in emotion.lower() for emotion in emoji.emotions):
                results.append(emoji)
                continue
            
            # 搜索自定义标签
            if any(keyword_lower in tag.lower() for tag in emoji.tags):
                results.append(emoji)
                continue
            
            # 搜索分类
            if keyword_lower in emoji.category.lower():
                results.append(emoji)
                continue
        
        return results
    
    def update_emoji_category(self, emoji_hash: str, category: str) -> Tuple[bool, str, Optional[EmojiInfo]]:
        """更新表情包的分类
        
        Args:
            emoji_hash: 表情包的哈希值
            category: 新的分类
            
        Returns:
            Tuple[bool, str, Optional[EmojiInfo]]: (是否成功, 消息, 更新后的表情包信息)
        """
        try:
            if emoji_hash not in self.emojis:
                return False, "表情包不存在", None
            
            emoji = self.emojis[emoji_hash]
            emoji.category = category
            
            if self._save_emojis():
                logger.info(f"更新表情包分类成功，哈希值: {emoji_hash}, 新分类: {category}")
                return True, "表情包分类更新成功", emoji
            else:
                return False, "更新表情包分类失败", None
                
        except Exception as e:
            logger.error(f"更新表情包分类失败: {e}")
            return False, f"更新表情包分类时发生错误: {str(e)}", None
    
    def get_all_categories(self) -> List[str]:
        """获取所有表情包分类
        
        Returns:
            List[str]: 分类列表
        """
        categories = set()
        for emoji in self.emojis.values():
            categories.add(emoji.category)
        return list(categories)
    
    def get_random_emoji(self, count: int = 1, emotion: str = None, tag: str = None, category: str = None) -> List[EmojiInfo]:
        """获取随机表情包
        
        Args:
            count: 获取的表情包数量
            emotion: 可选，情绪标签过滤
            tag: 可选，自定义标签过滤
            category: 可选，分类过滤
            
        Returns:
            List[EmojiInfo]: 随机表情包列表
        """
        # 过滤表情包
        filtered_emojis = list(self.emojis.values())
        
        if emotion:
            filtered_emojis = self.get_emojis_by_emotion(emotion)
        
        if tag:
            filtered_emojis = self.get_emojis_by_tag(tag)
        
        if category:
            filtered_emojis = self.get_emojis_by_category(category)
        
        if not filtered_emojis:
            return []
        
        # 随机选择
        count = min(count, len(filtered_emojis))
        return random.sample(filtered_emojis, count)
    
    def get_most_used_emojis(self, count: int = 10) -> List[EmojiInfo]:
        """获取使用次数最多的表情包
        
        Args:
            count: 获取的表情包数量
            
        Returns:
            List[EmojiInfo]: 使用次数最多的表情包列表
        """
        sorted_emojis = sorted(self.emojis.values(), key=lambda x: x.usage_count, reverse=True)
        return sorted_emojis[:count]
    
    def get_count(self) -> int:
        """获取表情包总数
        
        Returns:
            int: 表情包总数
        """
        return len(self.emojis)
    
    def get_info(self) -> Dict[str, Any]:
        """获取表情包统计信息
        
        Returns:
            Dict[str, any]: 统计信息
        """
        total_count = len(self.emojis)
        
        # 情绪标签统计
        emotion_counts = {}
        for emoji in self.emojis.values():
            for emotion in emoji.emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 标签统计
        tag_counts = {}
        for emoji in self.emojis.values():
            for tag in emoji.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # 使用次数统计
        total_usage = sum(emoji.usage_count for emoji in self.emojis.values())
        average_usage = total_usage / total_count if total_count > 0 else 0
        
        return {
            "total_count": total_count,
            "emotion_counts": emotion_counts,
            "tag_counts": tag_counts,
            "total_usage": total_usage,
            "average_usage": average_usage
        }
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算两个字符串的编辑距离
        
        Args:
            s1: 第一个字符串
            s2: 第二个字符串
            
        Returns:
            int: 编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_emoji_for_text(self, text_emotion: str, count: int = 1, category: str = None) -> List[EmojiInfo]:
        """根据文本情感描述获取最匹配的表情包
        
        Args:
            text_emotion: 文本情感描述
            count: 获取的表情包数量
            category: 可选，分类过滤
            
        Returns:
            List[EmojiInfo]: 匹配的表情包列表
        """
        try:
            # 过滤表情包
            filtered_emojis = list(self.emojis.values())
            
            if category:
                filtered_emojis = self.get_emojis_by_category(category)
            
            if not filtered_emojis:
                return []
            
            # 计算每个表情包与输入文本的最大情感相似度
            emoji_similarities = []
            for emoji in filtered_emojis:
                emotions = emoji.emotions
                if not emotions:
                    continue
                
                # 计算与每个emotion标签的相似度，取最大值
                max_similarity = 0
                best_matching_emotion = ""
                for emotion in emotions:
                    # 使用编辑距离计算相似度
                    distance = self._levenshtein_distance(text_emotion, emotion)
                    max_len = max(len(text_emotion), len(emotion))
                    similarity = 1 - (distance / max_len if max_len > 0 else 0)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_matching_emotion = emotion
                
                if best_matching_emotion:
                    emoji_similarities.append((emoji, max_similarity, best_matching_emotion))
            
            # 按相似度降序排序
            emoji_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 获取前10个最相似的表情包
            top_emojis = emoji_similarities[:10] if len(emoji_similarities) > 10 else emoji_similarities
            
            if not top_emojis:
                return []
            
            # 从相似表情包中随机选择指定数量
            count = min(count, len(top_emojis))
            selected_emojis = random.sample([e[0] for e in top_emojis], count)
            
            # 更新使用次数
            for emoji in selected_emojis:
                emoji.increment_usage()
                self._save_emojis()
            
            logger.info(f"为[{text_emotion}]找到{count}个匹配的表情包")
            return selected_emojis
            
        except Exception as e:
            logger.error(f"获取匹配表情包失败: {e}")
            return []


# 全局表情包管理器实例
emoji_manager = None


def get_emoji_manager() -> Optional[EmojiManager]:
    """获取全局表情包管理器实例
    
    Returns:
        Optional[EmojiManager]: 表情包管理器实例
    """
    global emoji_manager
    return emoji_manager


def initialize_emoji_manager(data_dir: str = "emoji_data") -> bool:
    """初始化全局表情包管理器
    
    Args:
        data_dir: 表情包数据存储目录
        
    Returns:
        bool: 是否初始化成功
    """
    global emoji_manager
    try:
        emoji_manager = EmojiManager(data_dir)
        logger.info("全局表情包管理器初始化成功")
        return True
    except Exception as e:
        logger.error(f"初始化全局表情包管理器失败: {e}")
        return False
