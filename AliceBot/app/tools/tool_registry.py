from typing import Dict, Type, List, Optional
from app.tools.base_tool import BaseTool
from app.tools.web_search import WebSearchTool
from app.tools.image_gen import ImageGenTool
from app.tools.data_analysis import DataAnalysisTool
from app.tools.forward_message import ForwardMessageTool, ListForwardMessagesTool
import logging

logger = logging.getLogger("ToolRegistry")


class ToolRegistry:
    """工具注册表，用于管理所有工具"""
    
    def __init__(self):
        """初始化工具注册表"""
        self._tools: Dict[str, Type[BaseTool]] = {}  # 工具名称 -> 工具类
        self._llm_available_tools: Dict[str, Type[BaseTool]] = {}  # LLM可用的工具名称 -> 工具类
        
        # 注册内置工具
        self.register_builtin_tools()
    
    def register_builtin_tools(self):
        """注册内置工具"""
        builtin_tools = [
            WebSearchTool,
            ImageGenTool,
            DataAnalysisTool,
            ForwardMessageTool,
            ListForwardMessagesTool
        ]
        
        for tool_class in builtin_tools:
            self.register_tool(tool_class)
        
        logger.info(f"已注册 {len(builtin_tools)} 个内置工具")
    
    def register_tool(self, tool_class: Type[BaseTool]) -> bool:
        """注册工具
        
        Args:
            tool_class: 工具类
            
        Returns:
            bool: 是否注册成功
        """
        tool_name = tool_class.name
        
        if tool_name in self._tools:
            logger.warning(f"工具 '{tool_name}' 已存在，跳过注册")
            return False
        
        self._tools[tool_name] = tool_class
        
        # 如果工具可供LLM使用，添加到LLM可用工具列表
        if tool_class.available_for_llm:
            self._llm_available_tools[tool_name] = tool_class
        
        logger.debug(f"已注册工具: {tool_name}")
        return True
    
    def unregister_tool(self, tool_name: str) -> bool:
        """注销工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            bool: 是否注销成功
        """
        if tool_name not in self._tools:
            logger.warning(f"工具 '{tool_name}' 不存在，跳过注销")
            return False
        
        # 从所有注册表中移除
        self._tools.pop(tool_name)
        if tool_name in self._llm_available_tools:
            self._llm_available_tools.pop(tool_name)
        
        logger.debug(f"已注销工具: {tool_name}")
        return True
    
    def get_tool(self, tool_name: str) -> Optional[Type[BaseTool]]:
        """获取工具类
        
        Args:
            tool_name: 工具名称
            
        Returns:
            Optional[Type[BaseTool]]: 工具类或None
        """
        return self._tools.get(tool_name)
    
    def get_tool_instance(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具实例
        
        Args:
            tool_name: 工具名称
            
        Returns:
            Optional[BaseTool]: 工具实例或None
        """
        tool_class = self.get_tool(tool_name)
        if tool_class:
            return tool_class()
        return None
    
    def get_all_tools(self) -> List[Type[BaseTool]]:
        """获取所有工具类
        
        Returns:
            List[Type[BaseTool]]: 所有工具类列表
        """
        return list(self._tools.values())
    
    def get_llm_available_tools(self) -> List[Type[BaseTool]]:
        """获取LLM可用的所有工具类
        
        Returns:
            List[Type[BaseTool]]: LLM可用的工具类列表
        """
        return list(self._llm_available_tools.values())
    
    def get_llm_tool_definitions(self) -> List[Dict]:
        """获取LLM工具调用的工具定义列表
        
        Returns:
            List[Dict]: LLM工具定义列表
        """
        return [tool_class.get_tool_definition() for tool_class in self.get_llm_available_tools()]
    
    def is_tool_available(self, tool_name: str) -> bool:
        """检查工具是否可用
        
        Args:
            tool_name: 工具名称
            
        Returns:
            bool: 工具是否可用
        """
        return tool_name in self._tools
    
    def is_tool_available_for_llm(self, tool_name: str) -> bool:
        """检查工具是否可供LLM使用
        
        Args:
            tool_name: 工具名称
            
        Returns:
            bool: 工具是否可供LLM使用
        """
        return tool_name in self._llm_available_tools


# 创建全局工具注册表实例
tool_registry = ToolRegistry()