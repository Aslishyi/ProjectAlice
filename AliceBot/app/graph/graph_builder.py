"""
聊天机器人工作流程图构建模块

该模块使用 LangGraph 构建 ProjectAlice 聊天机器人的完整工作流程，
包括响应式回复和主动发起对话两种模式，以及各节点之间的路由逻辑。
"""
from langgraph.graph import StateGraph, END
from app.core.state import AgentState

# 节点引入
from app.graph.nodes.context_filter import context_filter_node
from app.graph.nodes.parallel_processor import parallel_processing_node
from app.graph.nodes.unified_agent import agent_node
from app.graph.nodes.tool_handler import tool_node
from app.graph.nodes.memory_saver import memory_saver_node
from app.graph.nodes.summarizer import summarizer_node
from app.graph.nodes.proactive_agent import proactive_node
from app.graph.nodes.perception import perception_node


def route_agent_output(state: AgentState) -> str:
    """
    路由智能体输出到工具处理或记忆保存
    
    Args:
        state: 当前智能体状态
    
    Returns:
        str: 下一个节点的名称
    """
    step = state.get("next_step", "save")
    if step == "tool":
        return "tools"
    return "saver"


def route_root(state: AgentState) -> str:
    """
    根路由：根据模式选择进入响应式或主动式流程
    
    Args:
        state: 当前智能体状态
    
    Returns:
        str: 下一个节点的名称
    """
    if state.get("is_proactive_mode", False):
        return "proactive"
    return "filter"


def route_filter(state: AgentState) -> str:
    """
    上下文过滤器路由：根据是否需要回复选择流程分支
    
    Args:
        state: 当前智能体状态
    
    Returns:
        str: 下一个节点的名称
    """
    if not state.get("should_reply", False):
        # 不需要回复时，直接保存记忆并结束
        return "summarizer"
    
    # 如果有短路回复字段，直接传递到agent_node，避免复杂的感知和推理
    if "short_circuit_emoji" in state or "short_circuit_text" in state:
        return "agent"
    
    # 需要回复且没有短路字段时，进行复杂的感知和推理
    return "parallel_processor"


def build_graph():
    """
    构建完整的聊天机器人工件流程图
    
    节点说明：
    - filter: 上下文过滤器，判断是否需要回复
    - parallel_processor: 并行处理器，同时进行视觉感知和心理分析
    - agent: 统一智能体，生成回复内容
    - tools: 工具处理节点，执行需要的工具调用
    - saver: 长期记忆保存节点
    - summarizer: 短期记忆总结和文件IO节点
    - perception: 视觉感知节点，分析图片内容
    - proactive: 主动社交引擎节点，决定是否主动发起对话
    
    Returns:
        StateGraph: 编译后的工作流程图
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("filter", context_filter_node)            # 上下文过滤器
    workflow.add_node("parallel_processor", parallel_processing_node)  # 并行处理器
    workflow.add_node("agent", agent_node)                    # 统一智能体
    workflow.add_node("tools", tool_node)                    # 工具处理
    workflow.add_node("saver", memory_saver_node)              # 长期记忆保存
    workflow.add_node("summarizer", summarizer_node)            # 短期记忆总结
    workflow.add_node("perception", perception_node)            # 视觉感知
    workflow.add_node("proactive", proactive_node)              # 主动社交引擎

    # 入口路由：根据模式选择流程
    workflow.set_conditional_entry_point(
        route_root,
        {
            "filter": "filter",    # 响应式模式：先经过上下文过滤器
            "proactive": "proactive"  # 主动式模式：直接进入主动社交引擎
        }
    )

    # 上下文过滤器路由：决定是否需要复杂处理
    workflow.add_conditional_edges(
        "filter",
        route_filter,
        {
            "parallel_processor": "parallel_processor",  # 需要回复：进行并行处理
            "summarizer": "summarizer",                # 不需要回复：直接保存记忆
            "agent": "agent"                        # 短路回复：直接传递到智能体
        }
    )

    # 响应式流程主线
    workflow.add_edge("parallel_processor", "agent")  # 并行处理后进入智能体
    workflow.add_conditional_edges(
        "agent",
        route_agent_output,
        {"tools": "tools", "saver": "saver"}  # 智能体输出到工具或记忆保存
    )
    workflow.add_edge("tools", "agent")  # 工具执行后回到智能体

    # 记忆处理流程
    workflow.add_edge("saver", "summarizer")  # 长期记忆保存后进行短期记忆总结
    workflow.add_edge("summarizer", END)      # 总结完成后结束流程

    # 主动式流程
    workflow.add_edge("proactive", "summarizer")  # 主动社交引擎直接进入记忆总结

    return workflow.compile()
