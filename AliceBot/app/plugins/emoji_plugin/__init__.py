# emoji_plugin插件初始化文件
import os
import sys
import logging
from typing import List, Dict, Any, Optional, Type

# 确保当前目录在Python路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.plugins.base_plugin import BasePlugin, PluginInfo
from app.tools.base_tool import BaseTool

# 延迟导入，避免循环导入
logger = logging.getLogger("EmojiPlugin")

class EmojiPlugin(BasePlugin):
    """表情包插件"""
    
    plugin_info = PluginInfo(
        name="emoji_plugin",
        version="1.0.0",
        description="提供表情包管理功能，支持添加、删除、查询和随机获取表情包",
        author="Your Name"
    )
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化插件
        
        Args:
            config: 插件配置
        """
        super().__init__(config)
        self.data_dir = self.config.get("data_dir", "emoji_data")
    
    def get_tools(self) -> List[Type[BaseTool]]:
        """获取插件提供的工具列表"""
        # 延迟导入工具类
        from .tools import (
            AddEmojiTool,
            DeleteEmojiTool,
            ListEmojisTool,
            GetEmojiTool,
            GetRandomEmojiTool,
            GetEmojiStatsTool,
            AddEmojiFromUrlTool,
            GetEmojisByCategoryTool,
            UpdateEmojiCategoryTool,
            GetAllCategoriesTool,
            SearchEmojisTool
        )
        return [
            AddEmojiTool,
            DeleteEmojiTool,
            ListEmojisTool,
            GetEmojiTool,
            GetRandomEmojiTool,
            GetEmojiStatsTool,
            AddEmojiFromUrlTool,
            GetEmojisByCategoryTool,
            UpdateEmojiCategoryTool,
            GetAllCategoriesTool,
            SearchEmojisTool
        ]
    
    async def _initialize(self) -> bool:
        """插件自定义初始化逻辑"""
        try:
            # 延迟导入并初始化表情包管理器
            from .emoji_manager import initialize_emoji_manager
            if initialize_emoji_manager(self.data_dir):
                logger.info(f"表情包管理器已初始化，数据目录: {self.data_dir}")
                
                # 初始化表情包服务
                from .emoji_service import initialize_emoji_service
                if initialize_emoji_service():
                    logger.info("表情包服务已初始化")
                    return True
                else:
                    logger.error("表情包服务初始化失败")
                    return False
            else:
                logger.error("表情包管理器初始化失败")
                return False
        except Exception as e:
            logger.error(f"插件初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def shutdown(self) -> bool:
        """关闭插件"""
        try:
            logger.info("表情包插件已关闭")
            return True
        except Exception as e:
            logger.error(f"插件关闭失败: {e}")
            return False
    
    async def _shutdown(self) -> bool:
        """插件自定义关闭逻辑"""
        try:
            logger.info("表情包管理器已关闭")
            return True
        except Exception as e:
            logger.error(f"表情包管理器关闭失败: {e}")
            return False
