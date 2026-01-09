import logging
from datetime import datetime

from langchain_core.messages import ToolMessage  # 引入 ToolMessage
from app.core.state import AgentState
from app.tools.web_search import perform_web_search
from app.tools.image_gen import generate_image
from app.tools.data_analysis import run_python_analysis
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
    tool_args = tool_data.get("args")

    # 生成一个随机的 tool_call_id，这对于某些模型（如 GPT/Claude）保持对话结构很重要
    # 虽然这里我们是通过 prompt 模拟的调用，但保持结构一致性有好处
    tool_call_id = str(uuid.uuid4())

    logger.info(f"[{ts}] --- [Tools] Executing: {tool_name} with {tool_args} --- ")

    result = "Tool execution failed."

    try:
        # 参数清洗
        final_arg = tool_args
        if isinstance(tool_args, dict):
            # 尝试获取最可能的参数值
            if "query" in tool_args:
                final_arg = tool_args["query"]
            elif "prompt" in tool_args:
                final_arg = tool_args["prompt"]
            elif "code" in tool_args:
                final_arg = tool_args["code"]
            else:
                final_arg = list(tool_args.values())[0]

        final_arg = str(final_arg)

        # 检查工具调用结果缓存
        cache_key_args = {"arg": final_arg}
        cached_result = await cached_tool_result_get(tool_name, cache_key_args)
        
        if cached_result:
            logger.info(f"[{ts}] [Tools Cache Hit] {tool_name}: {final_arg[:30]}... ")
            result = cached_result
        else:
            # 缓存未命中，执行工具调用
            if tool_name == "web_search":
                # 直接调用异步函数，不通过ainvoke方法
                result = await perform_web_search(final_arg)
            elif tool_name == "generate_image":
                # 直接调用异步函数，不通过ainvoke方法
                url = await generate_image(final_arg)
                result = f"IMAGE_GENERATED: {url}"
            elif tool_name == "run_python_analysis":
                # 对于不支持异步的工具，使用asyncio的run_in_executor避免阻塞
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, run_python_analysis, final_arg)
            else:
                result = f"Unknown tool: {tool_name}"
            
            # 将结果存入缓存
            await cached_tool_result_set(tool_name, cache_key_args, result)
            logger.info(f"[{ts}] [Tools Cache Set] {tool_name}: {final_arg[:30]}... ")

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
