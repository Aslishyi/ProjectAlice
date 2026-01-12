from app.plugins.base_plugin import BasePlugin, PluginInfo
from app.tools.base_tool import BaseTool, ToolParam
import logging

logger = logging.getLogger("ExamplePlugin")


class ExampleTool(BaseTool):
    """示例工具"""
    
    name = "example_tool"
    description = "一个示例工具，用于演示插件系统的使用"
    parameters = [
        ToolParam(
            name="message",
            param_type="string",
            description="要处理的消息",
            required=True
        )
    ]
    available_for_llm = True
    
    async def execute(self, message: str, **kwargs) -> dict:
        """执行示例工具"""
        try:
            result = f"示例工具已处理消息: {message}"
            return {
                "success": True,
                "result": result,
                "error": ""
            }
        except Exception as e:
            error_msg = f"示例工具执行失败: {e}"
            return {
                "success": False,
                "result": "",
                "error": error_msg
            }


class ExamplePlugin(BasePlugin):
    """示例插件"""
    
    plugin_info = PluginInfo(
        name="example_plugin",
        version="1.0.0",
        description="一个示例插件，用于演示插件系统的使用",
        author="AliceBot Team"
    )
    
    def get_tools(self) -> list[type[BaseTool]]:
        """获取插件提供的工具列表"""
        return [ExampleTool]
    
    async def _initialize(self) -> bool:
        """插件自定义初始化逻辑"""
        logger.info(f"示例插件 '{self.plugin_info.name}' v{self.plugin_info.version} 初始化完成")
        return True
    
    async def _shutdown(self) -> bool:
        """插件自定义关闭逻辑"""
        logger.info(f"示例插件 '{self.plugin_info.name}' v{self.plugin_info.version} 已关闭")
        return True