import pytest
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.graph_builder import build_graph
from app.core.state import AgentState


@pytest.mark.asyncio
async def test_graph_builder_compile():
    """测试图构建器能否成功编译工作流"""
    graph = build_graph()
    assert graph is not None
    assert hasattr(graph, "astream")
    assert callable(graph.astream)


@pytest.mark.asyncio
async def test_graph_reactive_route():
    """测试响应式路由"""
    graph = build_graph()
    
    # 创建一个响应式输入状态
    state = AgentState(
        messages=[HumanMessage(content="你好")],
        conversation_summary="",
        visual_input=None,
        image_urls=[],
        session_id="test-session",
        sender_qq="test-qq",
        sender_name="test-user",
        is_group=False,
        is_mentioned=True,
        user_profile={"relationship": {"intimacy": 50}},
        should_reply=True,
        is_proactive_mode=False,
        global_emotion_snapshot={"current_mood": "Calm"},
        psychological_context={},
        current_image_artifact=None,
        tool_call={},
        emotion={"current_mood": "Calm"},
        last_interaction_ts=0
    )
    
    # 测试图能否处理输入并产生输出
    async for output in graph.astream(state):
        assert isinstance(output, dict)
        # 验证至少有一个节点被执行
        assert len(output) > 0
        break


@pytest.mark.asyncio
async def test_graph_proactive_route():
    """测试主动触发路由"""
    graph = build_graph()
    
    # 创建一个主动输入状态
    state = AgentState(
        messages=[HumanMessage(content="你好")],
        conversation_summary="",
        visual_input=None,
        image_urls=[],
        session_id="test-session",
        sender_qq="test-qq",
        sender_name="test-user",
        is_group=False,
        is_mentioned=False,
        user_profile={"relationship": {"intimacy": 50}},
        should_reply=False,
        is_proactive_mode=True,
        global_emotion_snapshot={"current_mood": "Calm"},
        psychological_context={},
        current_image_artifact=None,
        tool_call={},
        last_interaction_ts=0
    )
    
    # 测试图能否处理主动输入
    async for output in graph.astream(state):
        assert isinstance(output, dict)
        # 检查是否进入了 proactive 节点
        if "proactive" in output:
            proactive_output = output["proactive"]
            # 验证 proactive 节点的输出格式
            assert isinstance(proactive_output, dict)
            break


if __name__ == "__main__":
    asyncio.run(test_graph_builder_compile())
    print("test_graph_builder_compile passed")
    
    asyncio.run(test_graph_reactive_route())
    print("test_graph_reactive_route passed")
    
    asyncio.run(test_graph_proactive_route())
    print("test_graph_proactive_route passed")
