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


def route_agent_output(state: AgentState):
    step = state.get("next_step", "save")
    if step == "tool": return "tools"
    return "saver"


def route_root(state: AgentState):
    if state.get("is_proactive_mode", False):
        return "proactive"
    return "filter"


def route_filter(state: AgentState):
    if state.get("should_reply", False):
        # 只有需要回复时，才进行复杂的感知和推理
        return "parallel_processor"
    else:
        return "summarizer"


def build_graph():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("filter", context_filter_node)
    workflow.add_node("parallel_processor", parallel_processing_node)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("saver", memory_saver_node)  # 长期记忆
    workflow.add_node("summarizer", summarizer_node)  # 短期记忆 & 文件IO

    # Proactive
    workflow.add_node("perception", perception_node)
    workflow.add_node("proactive", proactive_node)

    # 入口
    workflow.set_conditional_entry_point(
        route_root,
        {
            "filter": "filter",
            "proactive": "proactive"
        }
    )

    # Filter 路由优化
    workflow.add_conditional_edges(
        "filter",
        route_filter,
        {
            "parallel_processor": "parallel_processor",
            "summarizer": "summarizer"  # 快速通道：静默记录
        }
    )

    # 其他连线保持不变
    workflow.add_edge("parallel_processor", "agent")
    workflow.add_conditional_edges(
        "agent",
        route_agent_output,
        {"tools": "tools", "saver": "saver"}
    )
    workflow.add_edge("tools", "agent")

    workflow.add_edge("saver", "summarizer")
    workflow.add_edge("summarizer", END)

    # Proactive
    workflow.add_edge("proactive", "summarizer")

    return workflow.compile()
