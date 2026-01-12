from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("BaseTool")


class ToolParam:
    """工具参数定义"""
    def __init__(self, name: str, param_type: str, description: str, required: bool = True, enum_values: Optional[List[str]] = None):
        self.name = name
        self.param_type = param_type
        self.description = description
        self.required = required
        self.enum_values = enum_values

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "name": self.name,
            "type": self.param_type,
            "description": self.description,
            "required": self.required
        }
        if self.enum_values:
            result["enum"] = self.enum_values
        return result


class BaseTool(ABC):
    """所有工具的基类"""
    
    name: str = ""  # 工具名称
    description: str = ""  # 工具描述
    parameters: List[ToolParam] = []  # 工具参数列表
    available_for_llm: bool = True  # 是否可供LLM使用
    
    def __init__(self):
        """初始化工具"""
        # 验证工具定义是否完整
        if not self.name or not self.description:
            raise ValueError(f"工具类 {self.__class__.__name__} 必须定义 name 和 description 属性")
    
    @classmethod
    def get_tool_definition(cls) -> Dict[str, Any]:
        """获取工具定义，用于LLM工具调用"""
        return {
            "name": cls.name,
            "description": cls.description,
            "parameters": {
                "type": "object",
                "properties": {param.name: param.to_dict() for param in cls.parameters},
                "required": [param.name for param in cls.parameters if param.required]
            }
        }
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具函数
        
        Args:
            **kwargs: 工具调用参数
            
        Returns:
            Dict[str, Any]: 工具执行结果，包含以下字段：
                - success: bool，执行是否成功
                - result: Any，执行结果
                - error: str，错误信息（如果执行失败）
        """
        raise NotImplementedError("子类必须实现execute方法")
    
    def validate_params(self, **kwargs) -> Tuple[bool, str]:
        """验证工具参数
        
        Args:
            **kwargs: 工具调用参数
            
        Returns:
            Tuple[bool, str]: (是否验证通过, 错误信息)
        """
        # 检查必填参数
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return False, f"缺少必填参数: {param.name}"
        
        # 检查参数类型
        for param in self.parameters:
            if param.name in kwargs:
                value = kwargs[param.name]
                if param.param_type == "string" and not isinstance(value, str):
                    return False, f"参数 {param.name} 必须是字符串类型"
                elif param.param_type == "integer" and not isinstance(value, int):
                    return False, f"参数 {param.name} 必须是整数类型"
                elif param.param_type == "float" and not isinstance(value, float):
                    return False, f"参数 {param.name} 必须是浮点数类型"
                elif param.param_type == "boolean" and not isinstance(value, bool):
                    return False, f"参数 {param.name} 必须是布尔类型"
                
                # 检查枚举值
                if param.enum_values and value not in param.enum_values:
                    return False, f"参数 {param.name} 必须是以下值之一: {', '.join(param.enum_values)}"
        
        return True, ""
