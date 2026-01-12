from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from app.tools.base_tool import BaseTool
from app.tools.tool_registry import tool_registry
import logging

logger = logging.getLogger("BasePlugin")


class PluginInfo:
    """插件信息"""
    def __init__(self, name: str, version: str, description: str, author: str):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.enabled = True
        self.tools: List[str] = []  # 插件注册的工具列表


class BasePlugin(ABC):
    """插件基类"""
    
    plugin_info: PluginInfo
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化插件
        
        Args:
            config: 插件配置
        """
        self.config = config or {}
        self._initialized = False
        
        # 验证插件信息是否完整
        if not hasattr(self, 'plugin_info'):
            raise ValueError(f"插件类 {self.__class__.__name__} 必须定义 plugin_info 属性")
        
        if not self.plugin_info.name or not self.plugin_info.version:
            raise ValueError(f"插件类 {self.__class__.__name__} 的 plugin_info 必须包含 name 和 version")
    
    def get_plugin_info(self) -> PluginInfo:
        """获取插件信息"""
        return self.plugin_info
    
    @abstractmethod
    def get_tools(self) -> List[type[BaseTool]]:
        """获取插件提供的工具列表
        
        Returns:
            List[type[BaseTool]]: 工具类列表
        """
        return []
    
    async def initialize(self) -> bool:
        """初始化插件
        
        Returns:
            bool: 是否初始化成功
        """
        if self._initialized:
            logger.warning(f"插件 '{self.plugin_info.name}' 已初始化")
            return True
        
        try:
            # 注册插件的工具
            tools = self.get_tools()
            for tool_class in tools:
                if tool_registry.register_tool(tool_class):
                    self.plugin_info.tools.append(tool_class.name)
            
            # 调用插件的自定义初始化逻辑
            result = await self._initialize()
            
            if result:
                self._initialized = True
                logger.info(f"插件 '{self.plugin_info.name}' v{self.plugin_info.version} 初始化成功")
            
            return result
        except Exception as e:
            logger.error(f"插件 '{self.plugin_info.name}' 初始化失败: {e}")
            return False
    
    @abstractmethod
    async def _initialize(self) -> bool:
        """插件自定义初始化逻辑
        
        Returns:
            bool: 是否初始化成功
        """
        return True
    
    async def shutdown(self) -> bool:
        """关闭插件
        
        Returns:
            bool: 是否关闭成功
        """
        if not self._initialized:
            logger.warning(f"插件 '{self.plugin_info.name}' 未初始化")
            return True
        
        try:
            # 移除插件注册的工具
            for tool_name in self.plugin_info.tools:
                tool_registry.unregister_tool(tool_name)
            
            # 调用插件的自定义关闭逻辑
            result = await self._shutdown()
            
            if result:
                self._initialized = False
                logger.info(f"插件 '{self.plugin_info.name}' 关闭成功")
            
            return result
        except Exception as e:
            logger.error(f"插件 '{self.plugin_info.name}' 关闭失败: {e}")
            return False
    
    @abstractmethod
    async def _shutdown(self) -> bool:
        """插件自定义关闭逻辑
        
        Returns:
            bool: 是否关闭成功
        """
        return True
    
    def is_initialized(self) -> bool:
        """检查插件是否已初始化
        
        Returns:
            bool: 是否已初始化
        """
        return self._initialized
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取插件配置
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """设置插件配置
        
        Args:
            key: 配置键
            value: 配置值
        """
        self.config[key] = value