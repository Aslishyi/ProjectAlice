# === Python代码文件: emoji_service.py ===
"""
表情包服务 - 统一的表情包功能入口，整合识别、分析、管理和回复功能
"""

import logging
import random
import re
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import io
import base64
from functools import lru_cache

from .emoji_manager import EmojiInfo, get_emoji_manager
from app.graph.nodes.perception import _classify_image, _analyze_emoji_with_llm, _process_image_with_llm

logger = logging.getLogger("EmojiService")


class EmojiService:
    """表情包服务类 - 统一管理表情包相关功能"""
    
    # 类常量：情绪关键词映射
    EMOTION_KEYWORDS = {
        "开心": ["开心", "快乐", "高兴", "愉悦", "喜悦", "欢乐", "兴奋", "愉快", "欢快", "开怀", "乐呵", "喜笑颜开", "眉开眼笑"],
        "难过": ["难过", "悲伤", "伤心", "痛苦", "难受", "忧伤", "忧郁", "沮丧", "失落", "悲哀", "悲痛", "哀伤", "心如刀割"],
        "生气": ["生气", "愤怒", "恼火", "发火", "恼怒", "气愤", "动怒", "怒火中烧", "怒不可遏"],
        "惊讶": ["惊讶", "惊喜", "吃惊", "震惊", "诧异", "惊异", "骇然", "惊叹", "瞠目结舌", "目瞪口呆"],
        "可爱": ["可爱", "萌", "萌物", "萌化", "卡哇伊"],
        "搞笑": ["搞笑", "幽默", "风趣", "好笑", "滑稽", "逗乐", "笑死", "笑死我了", "太搞笑了"],
        "无奈": ["无奈", "无语", "没法", "没办法", "无奈何", "无能为力", "无可奈何"],
        "尴尬": ["尴尬", "难堪", "难为情", "不好意思", "尴尬癌"],
        "困惑": ["困惑", "疑问", "疑惑", "不解", "迷茫", "懵", "懵圈", "一头雾水"],
        "平静": ["平静", "平和", "平稳", "宁静", "安静", "心平气和", "平静如水"]
    }
    
    # 类常量：否定词列表
    NEGATION_WORDS = ["不", "没", "没有", "不是", "不会", "不要", "不行", "不能", "无法"]
    
    def __init__(self):
        self.emoji_manager = get_emoji_manager()
        # 图片分类结果缓存
        self._image_classification_cache = {}
        # 表情包分析结果缓存
        self._emoji_analysis_cache = {}
        # 上下文情绪提取缓存
        self._context_emotion_cache = {}
        # 缓存大小限制
        self._CACHE_SIZE = 1000
        # 记录最近保存的表情包哈希值，用于避免重复发送
        self._recently_saved_emojis = []
        # 最近保存的表情包数量限制
        self._MAX_RECENT_EMOJIS = 2
    
    async def is_emoji(self, image: Image.Image, file_size_kb: float) -> bool:
        """
        判断图片是否为表情包
        
        Args:
            image: PIL Image对象
            file_size_kb: 图片文件大小（KB）
            
        Returns:
            bool: 是否为表情包
        """
        try:
            return await _classify_image(image, file_size_kb) == "sticker"
        except Exception as e:
            logger.error(f"❌ 图片分类失败: {e}")
            # 出错时使用本地备份逻辑
            try:
                width, height = image.size
                ratio = width / height if height > 0 else 0
                has_transparency = image.mode in ('RGBA', 'LA') or ('transparency' in image.info)
                return has_transparency or width <= 1024 or height <= 1024 or file_size_kb < 2048
            except:
                return False
    
    async def analyze_emoji(self, base64_data: str) -> Dict[str, Any]:
        """
        分析表情包，生成情绪标签、描述和分类
        
        Args:
            base64_data: 图片的base64编码数据
            
        Returns:
            dict: 包含分析结果的字典
        """
        try:
            return await _analyze_emoji_with_llm(base64_data)
        except Exception as e:
            logger.error(f"❌ 分析表情包时发生错误: {e}")
            return {
                "emotions": ["未知"],
                "description": "无法分析的表情包",
                "category": "其他"
            }
    
    async def process_emoji(self, image_url: str, user_qq: str = "", user_nickname: str = "") -> Dict[str, Any]:
        """
        完整处理表情包流程：下载、识别、分析、保存
        
        明确的图片与表情包边界判断：
        1. 首先通过_classify_image函数判断是否为表情包
        2. 仅对分类为"sticker"的图片进行后续处理
        3. 对普通图片（"photo"）直接返回失败，避免混淆处理逻辑
        4. 对小图标（"icon"）也直接返回失败，因为它们不是表情包
        
        Args:
            image_url: 图片URL
            user_qq: 发送者QQ号（可选）
            user_nickname: 发送者昵称（可选）
            
        Returns:
            dict: 包含处理结果的字典
        """
        try:
            if not self.emoji_manager:
                logger.error("❌ 表情包管理器未初始化")
                return {"success": False, "message": "表情包管理器未初始化"}
            
            # 下载图片并转换为base64
            base64_data = self.emoji_manager.download_image_to_base64(image_url)
            if not base64_data:
                return {"success": False, "message": "下载表情包失败"}
            
            # 判断是否为真正的表情包并同时分析 - 减少LLM调用次数
            base64_clean = base64_data.encode("ascii", errors="ignore").decode("ascii")
            image_bytes = base64.b64decode(base64_clean)
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            file_size_kb = len(image_bytes) / 1024
            
            # 小图标判断 - 仍然使用本地规则，因为小图标明显不是表情包
            if width < 50 or height < 50:
                classification = "icon"
                logger.info(f"� 跳过小图标，不保存为表情包 ({width}x{height}, {file_size_kb:.1f}KB)")
                return {"success": False, "message": f"不是表情包 (分类: {classification})", "classification": classification}
            
            # 确保图片尺寸适中，避免过大的图片被误分类为表情包
            if width > 2048 or height > 2048:
                logger.info(f"📏 跳过超大图片，不保存为表情包 ({width}x{height})")
                return {"success": False, "message": "图片尺寸过大，不是表情包", "classification": "photo"}
            
            # 确保文件大小适中，避免过大的文件被误分类为表情包
            if file_size_kb > 2048:  # 2MB
                logger.info(f"� 跳过超大文件，不保存为表情包 ({file_size_kb:.1f}KB)")
                return {"success": False, "message": "文件大小过大，不是表情包", "classification": "photo"}
            
            # 使用大模型同时进行判断和分析，减少LLM调用次数
            is_emoji, llm_result = await _process_image_with_llm(base64_data)
            
            # 明确的边界：只有判断为表情包的图片才被视为表情包
            if not is_emoji:
                classification = "photo"
                logger.info(f"� 跳过普通照片，不保存为表情包 ({width}x{height}, {file_size_kb:.1f}KB)")
                return {"success": False, "message": f"不是表情包 (分类: {classification})", "classification": classification}
            
            classification = "sticker"
            logger.info(f"🔍 图片分类结果: {classification} ({width}x{height}, {file_size_kb:.1f}KB)")
            
            # 从LLM结果中提取信息，充分利用所有有价值的情绪标签
            emotions = llm_result.get("emotions", ["未知"])
            # 过滤掉重复和无意义的情绪标签
            unique_emotions = []
            for emotion in emotions:
                if emotion and emotion != "未知" and emotion not in unique_emotions:
                    unique_emotions.append(emotion)
            # 如果没有有效的情绪标签，使用默认值
            if not unique_emotions:
                unique_emotions = ["未知"]
            
            description = llm_result.get("description", f"用户{user_nickname}发送的表情包")
            category = llm_result.get("category", "其他")
            
            # 设置标签
            tags = []
            if user_qq:
                tags.append("user_sent")
                tags.append(user_qq)
            else:
                tags.append("auto_detected")
            
            # 保存表情包
            success, message, emoji_info = self.emoji_manager.add_emoji(
                base64_data=base64_data,
                description=description,
                emotions=unique_emotions,
                tags=tags,
                category=category
            )
            
            if success:
                logger.info(f"✅ 成功保存表情包: {message}")
                # 记录最近保存的表情包，用于避免重复发送
                self._recently_saved_emojis.append(emoji_info.emoji_hash)
                # 限制最近保存的表情包数量
                if len(self._recently_saved_emojis) > self._MAX_RECENT_EMOJIS:
                    self._recently_saved_emojis.pop(0)
                return {
                    "success": True,
                    "message": message,
                    "emoji_info": emoji_info,
                    "description": description,
                    "emotions": unique_emotions,
                    "category": category,
                    "classification": classification
                }
            else:
                logger.error(f"❌ 保存表情包失败: {message}")
                return {"success": False, "message": message, "classification": classification}
                
        except Exception as e:
            logger.error(f"❌ 处理表情包时发生错误: {e}")
            return {"success": False, "message": str(e)}
    
    def get_emoji_for_context(self, context: Dict[str, Any], count: int = 1) -> List[EmojiInfo]:
        """
        根据对话上下文选择合适的表情包
        
        Args:
            context: 包含对话上下文信息的字典
            count: 需要获取的表情包数量
            
        Returns:
            List[EmojiInfo]: 选择的表情包列表
        """
        try:
            if not self.emoji_manager:
                logger.error("❌ 表情包管理器未初始化")
                return []
            
            # 从上下文提取情绪信息
            emotions = self._extract_emotions_from_context(context)
            
            # 提取对话元信息
            conversation_type = context.get("conversation_type", "private")  # private/group
            intimacy_level = context.get("intimacy_level", "medium")  # low/medium/high
            
            logger.info(f"🎯 分析上下文: 情绪={emotions}, 对话类型={conversation_type}, 亲密程度={intimacy_level}")
            
            # 如果没有提取到情绪信息，使用默认情绪并考虑对话类型
            if not emotions:
                logger.info(f"📊 未从上下文提取到情绪信息，使用默认情绪")
                return self.emoji_manager.get_random_emoji(count=count)
            
            # 根据情绪选择表情包
            matching_emojis = []
            for emotion in emotions:
                matching_emojis.extend(self.emoji_manager.get_emojis_by_emotion(emotion))
            
            # 如果有匹配的表情包，从中智能选择
            if matching_emojis:
                # 去重
                unique_emojis = list({emoji.emoji_hash: emoji for emoji in matching_emojis}.values())
                
                # 根据对话类型和亲密程度过滤表情包
                filtered_emojis = self._filter_emojis_by_context(unique_emojis, conversation_type, intimacy_level)
                
                # 过滤掉最近保存的表情包，避免重复发送
                filtered_emojis = [emoji for emoji in filtered_emojis 
                                 if emoji.emoji_hash not in self._recently_saved_emojis]
                
                count = min(count, len(filtered_emojis))
                if count > 0:
                    # 考虑使用频率，优先选择使用较少的表情包
                    selected_emojis = self._select_balanced_emojis(filtered_emojis, count)
                    logger.info(f"🎭 根据情绪和上下文选择了{count}个表情包: {[emoji.emoji_hash for emoji in selected_emojis]}")
                    return selected_emojis
            
            # 如果没有匹配的，使用最相似的情绪标签
            fallback_emojis = []
            for emotion in emotions:
                fallback_emojis.extend(self.emoji_manager.get_emoji_for_text(emotion, count=count*2))
            
            if fallback_emojis:
                unique_emojis = list({emoji.emoji_hash: emoji for emoji in fallback_emojis}.values())
                filtered_emojis = self._filter_emojis_by_context(unique_emojis, conversation_type, intimacy_level)
                
                # 过滤掉最近保存的表情包，避免重复发送
                filtered_emojis = [emoji for emoji in filtered_emojis 
                                 if emoji.emoji_hash not in self._recently_saved_emojis]
                
                count = min(count, len(filtered_emojis))
                if count > 0:
                    selected_emojis = self._select_balanced_emojis(filtered_emojis, count)
                    logger.info(f"🎭 根据相似情绪和上下文选择了{count}个表情包: {[emoji.emoji_hash for emoji in selected_emojis]}")
                    return selected_emojis
            
            # 最后兜底，随机选择但考虑对话类型
            logger.info(f"🎲 没有找到匹配的表情包，根据对话类型随机选择{count}个")
            
            # 获取随机表情包并过滤掉最近保存的
            random_emojis = self.emoji_manager.get_random_emoji(count=count * 2)  # 获取双倍数量以确保有足够的选择
            filtered_random = [emoji for emoji in random_emojis 
                             if emoji.emoji_hash not in self._recently_saved_emojis]
            
            # 确保返回正确数量
            if filtered_random:
                return filtered_random[:count]
            return []
            
        except Exception as e:
            logger.error(f"❌ 选择表情包失败: {e}")
            if self.emoji_manager:
                # 获取随机表情包并过滤掉最近保存的
                random_emojis = self.emoji_manager.get_random_emoji(count=count * 2)
                filtered_random = [emoji for emoji in random_emojis 
                                 if emoji.emoji_hash not in self._recently_saved_emojis]
                return filtered_random[:count] if filtered_random else []
            return []
    
    def _extract_emotions_from_context(self, context: Dict[str, Any]) -> List[str]:
        """
        从对话上下文中提取情绪信息
        
        Args:
            context: 包含对话上下文信息的字典
            
        Returns:
            List[str]: 提取的情绪标签列表
        """
        # 创建缓存键
        cache_key = self._create_context_cache_key(context)
        
        # 检查缓存
        if cache_key in self._context_emotion_cache:
            logger.debug(f"⚡ 上下文情绪提取缓存命中: {cache_key}")
            return self._context_emotion_cache[cache_key]
        
        emotions: list[str] = []
        
        # 1. 从最新消息中提取情绪标签
        last_message = context.get("last_message", "")
        if "【表情包:" in last_message:
            emotion_match = re.search(r"【表情包:(.*?)】", last_message)
            if emotion_match:
                emotion_tags = emotion_match.group(1).split("、")
                emotions.extend(emotion_tags)
        
        # 2. 从对话历史中提取情绪信息
        message_history = context.get("message_history", [])
        
        for message in message_history[-5:]:  # 查看最近5条消息
            # 统一获取消息内容
            if isinstance(message, dict) and "content" in message:
                content = message["content"]
            elif hasattr(message, "content"):
                content = str(message.content)
            else:
                content = str(message)
            
            # 检查每条消息的情绪
            for emotion, keywords in self.EMOTION_KEYWORDS.items():
                # 使用集合去重关键词，提高效率
                for keyword in set(keywords):
                    if keyword in content:
                        # 检查是否是否定句
                        is_negated = False
                        for negation in self.NEGATION_WORDS:
                            # 检查否定词是否在关键词前面
                            neg_pos = content.find(negation)
                            keyword_pos = content.find(keyword)
                            if neg_pos != -1 and neg_pos < keyword_pos:
                                # 简单判断：如果否定词在关键词前面，且距离不超过10个字符，则认为是否定
                                if keyword_pos - neg_pos < 10:
                                    is_negated = True
                                    break
                        
                        if not is_negated:
                            emotions.append(emotion)
        
        # 3. 从额外情绪标签中提取（如果有）
        additional_emotions = context.get("emotions", [])
        if isinstance(additional_emotions, list):
            emotions.extend(additional_emotions)
        
        # 4. 去重并限制数量
        result = list(set(emotions))[:5]  # 最多返回5个情绪标签
        
        # 保存到缓存
        self._context_emotion_cache[cache_key] = result
        
        # 清理缓存（保持固定大小）
        self._clean_cache(self._context_emotion_cache)
        
        return result
    
    def _create_context_cache_key(self, context: Dict[str, Any]) -> str:
        """
        创建上下文缓存键
        
        Args:
            context: 上下文字典
            
        Returns:
            str: 缓存键
        """
        last_message = context.get("last_message", "")
        message_history = context.get("message_history", [])
        
        # 只使用最近5条消息的内容创建缓存键
        history_content = ""
        for msg in message_history[-5:]:
            if isinstance(msg, dict) and "content" in msg:
                history_content += msg["content"]
            elif hasattr(msg, "content"):
                history_content += str(msg.content)
            else:
                history_content += str(msg)
        
        additional_emotions = context.get("emotions", [])
        
        # 使用内容的哈希值作为缓存键
        cache_content = f"{last_message}:{history_content}:{additional_emotions}"
        return hashlib.md5(cache_content.encode()).hexdigest()
    
    def _clean_cache(self, cache: Dict, max_size: int = None) -> None:
        """
        清理缓存，保持固定大小
        
        Args:
            cache: 要清理的缓存字典
            max_size: 最大缓存大小（默认使用类定义的大小）
        """
        max_size = max_size or self._CACHE_SIZE
        
        if len(cache) > max_size:
            # 删除最旧的缓存项（通过按键顺序，Python 3.7+ 字典保持插入顺序）
            items_to_remove = len(cache) - max_size
            for key in list(cache.keys())[:items_to_remove]:
                del cache[key]
    
    def _filter_emojis_by_context(self, emojis: List[EmojiInfo], conversation_type: str, intimacy_level: str) -> List[EmojiInfo]:
        """
        根据对话上下文过滤表情包
        
        Args:
            emojis: 表情包列表
            conversation_type: 对话类型 (private/group)
            intimacy_level: 亲密程度 (low/medium/high)
            
        Returns:
            List[EmojiInfo]: 过滤后的表情包列表
        """
        filtered = []
        
        for emoji in emojis:
            # 简单实现：群聊中避免使用过于私人或暧昧的表情包
            if conversation_type == "group" and intimacy_level == "low":
                # 假设emoji对象有category属性
                if hasattr(emoji, "category"):
                    # 群聊中避免使用过于私人的表情包类型
                    if emoji.category not in ["亲密", "暧昧"]:
                        filtered.append(emoji)
                else:
                    filtered.append(emoji)
            else:
                filtered.append(emoji)
        
        return filtered if filtered else emojis  # 如果过滤后为空，返回原始列表
    
    def _filter_recent_emojis(self, emojis: List[EmojiInfo]) -> List[EmojiInfo]:
        """
        过滤掉最近保存的表情包，避免重复发送
        
        Args:
            emojis: 表情包列表
            
        Returns:
            List[EmojiInfo]: 过滤后的表情包列表
        """
        if not self._recently_saved_emojis:
            return emojis
        
        # 过滤掉最近保存的表情包
        filtered = [emoji for emoji in emojis if emoji.emoji_hash not in self._recently_saved_emojis]
        
        # 如果过滤后没有表情包，返回原始列表（避免空列表）
        return filtered if filtered else emojis
    
    def _select_balanced_emojis(self, emojis: List[EmojiInfo], count: int) -> List[EmojiInfo]:
        """
        平衡选择表情包，考虑使用频率
        
        Args:
            emojis: 表情包列表
            count: 需要选择的数量
            
        Returns:
            List[EmojiInfo]: 选择的表情包列表
        """
        # 过滤掉最近保存的表情包
        filtered_emojis = self._filter_recent_emojis(emojis)
        
        # 如果过滤后没有表情包，返回原始列表
        if not filtered_emojis:
            filtered_emojis = emojis
        
        if len(filtered_emojis) <= count:
            return filtered_emojis
        
        # 假设emoji对象有usage_count属性记录使用次数
        # 这里使用随机选择，实际可以根据使用频率加权
        return random.sample(filtered_emojis, count)
    
    def get_default_emoji(self) -> str:
        """
        获取默认表情符号
        
        Returns:
            str: 默认表情符号
        """
        default_emojis = ["🐶", "🐱", "💖", "💕", "💝", "🤗", "👻", "👽"]
        return random.choice(default_emojis)


# 全局表情包服务实例
_emoji_service = None


def get_emoji_service() -> Optional[EmojiService]:
    """
    获取全局表情包服务实例
    
    Returns:
        Optional[EmojiService]: 表情包服务实例
    """
    global _emoji_service
    return _emoji_service


def initialize_emoji_service() -> bool:
    """
    初始化全局表情包服务
    
    Returns:
        bool: 是否初始化成功
    """
    global _emoji_service
    try:
        _emoji_service = EmojiService()
        logger.info("✅ 全局表情包服务初始化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 初始化全局表情包服务失败: {e}")
        return False
