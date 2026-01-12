import logging
from datetime import datetime

from langchain_core.messages import ToolMessage  # 引入 ToolMessage
from app.core.state import AgentState
from app.tools.tool_registry import tool_registry
from app.utils.cache import cached_tool_result_get, cached_tool_result_set
import uuid

# 配置日志
logger = logging.getLogger("ToolHandler")


async def tool_node(state: AgentState):
    """
    执行工具调用，并将结果作为 ToolMessage 注入历史
    """
    current_messages = state.get("messages", [])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tool_data = state.get("tool_call", {})
    tool_name = tool_data.get("name")
    tool_args = tool_data.get("args") or {}

    # 生成一个随机的 tool_call_id，这对于某些模型（如 GPT/Claude）保持对话结构很重要
    # 虽然这里我们是通过 prompt 模拟的调用，但保持结构一致性有好处
    tool_call_id = str(uuid.uuid4())

    logger.info(f"[{ts}] --- [Tools] Executing: {tool_name} with {tool_args} --- ")

    result = "Tool execution failed."

    try:
        # 检查工具是否存在
        if not tool_registry.is_tool_available(tool_name):
            result = f"Unknown tool: {tool_name}"
            logger.error(f"[{ts}] [Tool Error] {result}")
        else:
            # 标准化参数格式
            if not isinstance(tool_args, dict):
                # 尝试获取工具的主要参数
                tool_class = tool_registry.get_tool(tool_name)
                if tool_class and tool_class.parameters:
                    primary_param = tool_class.parameters[0].name
                    tool_args = {primary_param: str(tool_args)}
                else:
                    tool_args = {}
            
            # 检查工具调用结果缓存
            cache_key_args = tool_args.copy()
            cached_result = await cached_tool_result_get(tool_name, cache_key_args)
            
            if cached_result:
                logger.info(f"[{ts}] [Tools Cache Hit] {tool_name}: {str(tool_args)[:30]}... ")
                result = cached_result
            else:
                # 缓存未命中，执行工具调用
                tool_instance = tool_registry.get_tool_instance(tool_name)
                if tool_instance:
                    # 使用新的工具API执行
                    execute_result = await tool_instance.execute(**tool_args)
                    
                    if execute_result["success"]:
                        if tool_name == "generate_image":
                            result = f"IMAGE_GENERATED: {execute_result['result']}"
                        else:
                            result = execute_result["result"]
                    else:
                        result = execute_result["error"]
                        logger.error(f"[{ts}] [Tool Execution Error] {result}")
                else:
                    result = f"Failed to create tool instance: {tool_name}"
                
                # 将结果存入缓存
                await cached_tool_result_set(tool_name, cache_key_args, result)
                logger.info(f"[{ts}] [Tools Cache Set] {tool_name}: {str(tool_args)[:30]}... ")

    except Exception as e:
        logger.error(f"[{ts}] [Tool Error] {e}")
        result = f"Tool Error: {str(e)}"

    # --- 改进点：使用 ToolMessage ---
    # content 前加上标识，帮助 LLM 识别
    tool_msg = ToolMessage(
        content=f"[System: Tool '{tool_name}' Result]\n{str(result)}",
        tool_call_id=tool_call_id,
        name=tool_name
    )

    # 兼容性处理：如果你之前的 Agent Prompt 极度依赖 SystemMessage，可以保持 SystemMessage
    # 但 ToolMessage 是 LangChain 标准。这里我保留 SystemMessage 风格的内容但用 ToolMessage 类

    return {
        "messages": current_messages + [tool_msg],
        "tool_call": {}
    }
