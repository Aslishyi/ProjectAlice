from typing import List, Dict, Any
from app.tools.base_tool import BaseTool, ToolParam
import logging

# 使用动态导入来避免循环导入问题
def get_emoji_manager():
    from app.plugins.emoji_plugin.emoji_manager import get_emoji_manager as _get_emoji_manager
    return _get_emoji_manager()

def get_emoji_info_class():
    from app.plugins.emoji_plugin.emoji_manager import EmojiInfo
    return EmojiInfo

logger = logging.getLogger("EmojiTools")


class AddEmojiTool(BaseTool):
    """添加表情包工具"""
    
    name = "add_emoji"
    description = "添加或更新表情包，支持设置描述、情绪标签、自定义标签和分类"
    parameters = [
        ToolParam("base64_data", "string", "表情包的base64编码数据", required=True),
        ToolParam("description", "string", "表情包的描述", required=False),
        ToolParam("emotions", "array", "与表情包相关的情绪标签，如：['开心', '惊讶', '难过']", required=False),
        ToolParam("tags", "array", "表情包的自定义标签，如：['搞笑', '可爱', '工作']", required=False),
        ToolParam("category", "string", "表情包的分类，如：'general', 'funny', 'sad'", required=False)
    ]
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行添加表情包操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "result": None,
                "error": "表情包管理器未初始化"
            }
        
        try:
            base64_data = kwargs.get("base64_data")
            description = kwargs.get("description", "")
            emotions = kwargs.get("emotions", [])
            tags = kwargs.get("tags", [])
            category = kwargs.get("category", "general")
            
            success, message, emoji_info = emoji_manager.add_emoji(
                base64_data=base64_data,
                description=description,
                emotions=emotions,
                tags=tags,
                category=category
            )
            
            if success:
                return {
                    "success": True,
                    "result": {
                        "message": message,
                        "emoji_hash": emoji_info.emoji_hash,
                        "description": emoji_info.description,
                        "emotions": emoji_info.emotions,
                        "tags": emoji_info.tags
                    },
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": message
                }
                
        except Exception as e:
            logger.error(f"添加表情包失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"添加表情包失败: {str(e)}"
            }


class DeleteEmojiTool(BaseTool):
    """删除表情包工具"""
    
    name = "delete_emoji"
    description = "根据哈希值删除表情包"
    parameters = [
        ToolParam("emoji_hash", "string", "表情包的哈希值", required=True)
    ]
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行删除表情包操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "result": None,
                "error": "表情包管理器未初始化"
            }
        
        try:
            emoji_hash = kwargs.get("emoji_hash")
            
            if not emoji_hash:
                return {
                    "success": False,
                    "result": None,
                    "error": "缺少表情包哈希值"
                }
            
            success, message, emoji_info = emoji_manager.delete_emoji(emoji_hash)
            
            if success:
                return {
                    "success": True,
                    "result": {
                        "message": message,
                        "emoji_hash": emoji_hash,
                        "description": emoji_info.description if emoji_info else ""
                    },
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": message
                }
                
        except Exception as e:
            logger.error(f"删除表情包失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"删除表情包失败: {str(e)}"
            }


class ListEmojisTool(BaseTool):
    """列出表情包工具"""
    
    name = "list_emojis"
    description = "列出所有表情包信息，支持根据情绪或标签过滤"
    parameters = [
        ToolParam("emotion", "string", "可选，按情绪标签过滤", required=False),
        ToolParam("tag", "string", "可选，按自定义标签过滤", required=False),
        ToolParam("limit", "integer", "可选，返回结果的最大数量", required=False, enum_values=None)
    ]
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行列出表情包操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "result": None,
                "error": "表情包管理器未初始化"
            }
        
        try:
            emotion = kwargs.get("emotion")
            tag = kwargs.get("tag")
            limit = kwargs.get("limit", 100)
            
            if emotion:
                emojis = emoji_manager.get_emojis_by_emotion(emotion)
            elif tag:
                emojis = emoji_manager.get_emojis_by_tag(tag)
            else:
                emojis = emoji_manager.get_all_emojis()
            
            # 限制结果数量
            if limit > 0:
                emojis = emojis[:limit]
            
            # 转换为字典格式
            emoji_list = []
            for emoji in emojis:
                emoji_list.append({
                    "emoji_hash": emoji.emoji_hash,
                    "description": emoji.description,
                    "emotions": emoji.emotions,
                    "tags": emoji.tags,
                    "usage_count": emoji.usage_count,
                    "created_at": emoji.created_at
                })
            
            return {
                "success": True,
                "result": {
                    "total_count": len(emoji_list),
                    "emojis": emoji_list,
                    "filter_emotion": emotion,
                    "filter_tag": tag
                },
                "error": None
            }
            
        except Exception as e:
            logger.error(f"列出表情包失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"列出表情包失败: {str(e)}"
            }


class GetEmojiTool(BaseTool):
    """获取表情包工具"""
    
    name = "get_emoji"
    description = "根据哈希值获取表情包信息"
    parameters = [
        ToolParam("emoji_hash", "string", "表情包的哈希值", required=True)
    ]
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行获取表情包操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "result": None,
                "error": "表情包管理器未初始化"
            }
        
        try:
            emoji_hash = kwargs.get("emoji_hash")
            
            if not emoji_hash:
                return {
                    "success": False,
                    "result": None,
                    "error": "缺少表情包哈希值"
                }
            
            emoji_info = emoji_manager.get_emoji(emoji_hash)
            
            if emoji_info:
                return {
                    "success": True,
                    "result": {
                        "emoji_hash": emoji_info.emoji_hash,
                        "base64_data": emoji_info.base64_data,
                        "description": emoji_info.description,
                        "emotions": emoji_info.emotions,
                        "tags": emoji_info.tags,
                        "usage_count": emoji_info.usage_count,
                        "created_at": emoji_info.created_at,
                        "last_used_at": emoji_info.last_used_at
                    },
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": f"未找到哈希值为 {emoji_hash} 的表情包"
                }
                
        except Exception as e:
            logger.error(f"获取表情包失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"获取表情包失败: {str(e)}"
            }


class GetRandomEmojiTool(BaseTool):
    """获取随机表情包工具"""
    
    name = "get_random_emoji"
    description = "获取随机表情包，支持根据情绪、标签或分类过滤"
    parameters = [
        ToolParam("count", "integer", "获取的表情包数量，默认1", required=False),
        ToolParam("emotion", "string", "可选，按情绪标签过滤", required=False),
        ToolParam("tag", "string", "可选，按自定义标签过滤", required=False),
        ToolParam("category", "string", "可选，按分类过滤", required=False)
    ]
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行获取随机表情包操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "result": None,
                "error": "表情包管理器未初始化"
            }
        
        try:
            count = kwargs.get("count", 1)
            emotion = kwargs.get("emotion")
            tag = kwargs.get("tag")
            category = kwargs.get("category")
            
            # 限制数量范围
            count = max(1, min(count, 10))
            
            random_emojis = emoji_manager.get_random_emoji(count=count, emotion=emotion, tag=tag, category=category)
            
            if not random_emojis:
                return {
                    "success": False,
                    "result": None,
                    "error": "未找到符合条件的表情包"
                }
            
            # 转换为字典格式
            emoji_list = []
            for emoji in random_emojis:
                # 增加使用次数
                emoji.increment_usage()
                
                emoji_list.append({
                    "emoji_hash": emoji.emoji_hash,
                    "base64_data": emoji.base64_data,
                    "description": emoji.description,
                    "emotions": emoji.emotions,
                    "tags": emoji.tags,
                    "usage_count": emoji.usage_count
                })
            
            # 保存使用次数更新
            emoji_manager._save_emojis()
            
            return {
                "success": True,
                "result": {
                    "count": len(emoji_list),
                    "emojis": emoji_list,
                    "filter_emotion": emotion,
                    "filter_tag": tag
                },
                "error": None
            }
            
        except Exception as e:
            logger.error(f"获取随机表情包失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"获取随机表情包失败: {str(e)}"
            }


class GetEmojiStatsTool(BaseTool):
    """获取表情包统计信息工具"""
    
    name = "get_emoji_stats"
    description = "获取表情包的统计信息，包括总数、情绪标签统计、使用次数等"
    parameters = []
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行获取表情包统计信息操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "result": None,
                "error": "表情包管理器未初始化"
            }
        
        try:
            stats = emoji_manager.get_info()
            
            return {
                "success": True,
                "result": {
                    "total_count": stats["total_count"],
                    "emotion_counts": stats["emotion_counts"],
                    "tag_counts": stats["tag_counts"],
                    "total_usage": stats["total_usage"],
                    "average_usage": stats["average_usage"]
                },
                "error": None
            }
            
        except Exception as e:
            logger.error(f"获取表情包统计信息失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"获取表情包统计信息失败: {str(e)}"
            }


class AddEmojiFromUrlTool(BaseTool):
    """从URL添加表情包工具"""
    
    name = "add_emoji_from_url"
    description = "从图片URL添加表情包，自动下载图片并分析情绪标签"
    parameters = [
        ToolParam("image_url", "string", "图片的URL地址", required=True),
        ToolParam("description", "string", "表情包的描述，可选", required=False),
        ToolParam("emotions", "array", "与表情包相关的情绪标签，如：['开心', '惊讶', '难过']，不提供则自动分析", required=False),
        ToolParam("tags", "array", "表情包的自定义标签，如：['搞笑', '可爱', '工作']，可选", required=False),
        ToolParam("category", "string", "表情包的分类，如：'general', 'funny', 'sad'", required=False)
    ]
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行从URL添加表情包操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "result": None,
                "error": "表情包管理器未初始化"
            }
        
        try:
            image_url = kwargs.get("image_url")
            description = kwargs.get("description", "")
            emotions = kwargs.get("emotions", [])
            tags = kwargs.get("tags", [])
            category = kwargs.get("category", "general")
            
            if not image_url:
                return {
                    "success": False,
                    "result": None,
                    "error": "缺少图片URL"
                }
            
            success, message, emoji_info = emoji_manager.add_emoji_from_url(
                image_url=image_url,
                description=description,
                emotions=emotions,
                tags=tags,
                category=category
            )
            
            if success:
                return {
                    "success": True,
                    "result": {
                        "message": message,
                        "emoji_hash": emoji_info.emoji_hash,
                        "description": emoji_info.description,
                        "emotions": emoji_info.emotions,
                        "tags": emoji_info.tags
                    },
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": message
                }
                
        except Exception as e:
            logger.error(f"从URL添加表情包失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"从URL添加表情包失败: {str(e)}"
            }


class GetEmojisByCategoryTool(BaseTool):
    """根据分类获取表情包工具"""
    
    name = "get_emojis_by_category"
    description = "根据分类获取表情包列表"
    parameters = [
        ToolParam("category", "string", "表情包分类", required=True)
    ]
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行根据分类获取表情包操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "error": "表情包管理器未初始化"
            }
        
        try:
            category = kwargs.get("category")
            
            if not category:
                return {
                    "success": False,
                    "error": "缺少分类参数"
                }
            
            emojis = emoji_manager.get_emojis_by_category(category)
            
            return {
                "success": True,
                "result": {
                    "category": category,
                    "count": len(emojis),
                    "emojis": [{
                        "emoji_hash": emoji.emoji_hash,
                        "description": emoji.description,
                        "emotions": emoji.emotions,
                        "tags": emoji.tags,
                        "category": emoji.category
                    } for emoji in emojis]
                },
                "error": None
            }
            
        except Exception as e:
            logger.error(f"根据分类获取表情包失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"根据分类获取表情包失败: {str(e)}"
            }


class UpdateEmojiCategoryTool(BaseTool):
    """更新表情包分类工具"""
    
    name = "update_emoji_category"
    description = "更新表情包的分类"
    parameters = [
        ToolParam("emoji_hash", "string", "表情包的哈希值", required=True),
        ToolParam("category", "string", "新的分类", required=True)
    ]
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行更新表情包分类操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "error": "表情包管理器未初始化"
            }
        
        try:
            emoji_hash = kwargs.get("emoji_hash")
            category = kwargs.get("category")
            
            if not emoji_hash:
                return {
                    "success": False,
                    "error": "缺少表情包哈希值"
                }
            
            if not category:
                return {
                    "success": False,
                    "error": "缺少新分类参数"
                }
            
            success, message, emoji_info = emoji_manager.update_emoji_category(emoji_hash, category)
            
            if success:
                return {
                    "success": True,
                    "result": {
                        "message": message,
                        "emoji_hash": emoji_info.emoji_hash,
                        "description": emoji_info.description,
                        "old_category": emoji_info.category,
                        "new_category": category
                    },
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": message
                }
                
        except Exception as e:
            logger.error(f"更新表情包分类失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"更新表情包分类失败: {str(e)}"
            }


class GetAllCategoriesTool(BaseTool):
    """获取所有分类工具"""
    
    name = "get_all_categories"
    description = "获取所有表情包分类"
    parameters = []
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行获取所有分类操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "error": "表情包管理器未初始化"
            }
        
        try:
            categories = emoji_manager.get_all_categories()
            
            return {
                "success": True,
                "result": {
                    "categories": categories,
                    "count": len(categories)
                },
                "error": None
            }
            
        except Exception as e:
            logger.error(f"获取所有分类失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"获取所有分类失败: {str(e)}"
            }


class SearchEmojisTool(BaseTool):
    """搜索表情包工具"""
    
    name = "search_emojis"
    description = "根据关键词搜索表情包，支持在描述、情绪标签、自定义标签和分类中搜索"
    parameters = [
        ToolParam("keyword", "string", "搜索关键词", required=True)
    ]
    available_for_llm = True
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行搜索表情包操作"""
        emoji_manager = get_emoji_manager()
        if not emoji_manager:
            return {
                "success": False,
                "error": "表情包管理器未初始化"
            }
        
        try:
            keyword = kwargs.get("keyword")
            
            if not keyword:
                return {
                    "success": False,
                    "error": "缺少搜索关键词"
                }
            
            emojis = emoji_manager.search_emojis(keyword)
            
            return {
                "success": True,
                "result": {
                    "keyword": keyword,
                    "count": len(emojis),
                    "emojis": [{
                        "emoji_hash": emoji.emoji_hash,
                        "description": emoji.description,
                        "emotions": emoji.emotions,
                        "tags": emoji.tags,
                        "category": emoji.category
                    } for emoji in emojis]
                },
                "error": None
            }
            
        except Exception as e:
            logger.error(f"搜索表情包失败: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"搜索表情包失败: {str(e)}"
            }
