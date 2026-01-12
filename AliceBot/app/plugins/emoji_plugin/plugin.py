from typing import List, Dict, Any, Optional, Type
from app.plugins.base_plugin import BasePlugin, PluginInfo
from app.tools.base_tool import BaseTool
from .emoji_manager import initialize_emoji_manager
from .tools import (
    AddEmojiTool,
    DeleteEmojiTool,
    ListEmojisTool,
    GetEmojiTool,
    GetRandomEmojiTool,
    GetEmojiStatsTool,
    AddEmojiFromUrlTool
)
import logging

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
        return [
            AddEmojiTool,
            DeleteEmojiTool,
            ListEmojisTool,
            GetEmojiTool,
            GetRandomEmojiTool,
            GetEmojiStatsTool,
            AddEmojiFromUrlTool
        ]
    
    async def _initialize(self) -> bool:
        """插件自定义初始化逻辑"""
        try:
            # 初始化表情包管理器
            if initialize_emoji_manager(self.data_dir):
                logger.info(f"表情包管理器已初始化，数据目录: {self.data_dir}")
                return True
            else:
                logger.error("表情包管理器初始化失败")
                return False
        except Exception as e:
            logger.error(f"插件初始化失败: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """关闭插件"""
        try:
            logger.info("表情包插件已关闭")
            return True
        except Exception as e:
            logger.error(f"插件关闭失败: {e}")
            return False
