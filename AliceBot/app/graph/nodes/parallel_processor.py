# app/graph/nodes/parallel_processor.py

import asyncio
from app.core.state import AgentState
from app.graph.nodes.perception import perception_node
from app.graph.nodes.psychology import psychology_node
from app.core.vision_router import vision_router  # <--- 新增导入


async def parallel_processing_node(state: AgentState) -> dict:
    """
    并行执行节点：同时运行 [视觉感知] 和 [心理分析]。
    优化：引入 Vision Router，仅在必要时启动视觉感知，节省时间和 Token。
    """

    # 1. 决定是否需要启动视觉感知
    should_see = False
    image_urls = state.get("image_urls", [])

    if image_urls:
        # A. 如果当前消息直接包含图片，必须看
        should_see = True
        print("⚡ [Parallel] New image detected. Vision activated.")
    else:
        # B. 如果是纯文本，询问 Router 是否需要回溯看图
        # 注意：这里传入 messages 历史，Router 会判断是否有 "看看这个" 之类的指代词
        should_see = await vision_router.should_see(state.get("messages", []))
        if should_see:
            print("⚡ [Parallel] Vision Router decided to look at context.")

    # 2. 构造任务列表
    tasks = []

    # 任务A: 心理分析 (总是运行)
    tasks.append(psychology_node(state))

    # 任务B: 视觉感知 (按需运行)
    if should_see:
        print("⚡ [Parallel] Running Perception & Psychology concurrently...")
        tasks.append(perception_node(state))
    else:
        print("⚡ [Parallel] Running Psychology ONLY (Vision skipped).")

    # 3. 并发执行
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 4. 合并结果
    merged_update = {}

    # 处理结果列表
    # 结果顺序取决于 append 的顺序
    psychology_res = results[0]

    # 处理心理分析结果
    if isinstance(psychology_res, dict):
        merged_update.update(psychology_res)
    else:
        print(f"⚠️ [Parallel] Psychology failed: {psychology_res}")

    # 处理视觉结果 (如果运行了的话)
    if should_see:
        perception_res = results[1]  # 因为 Perception 是第二个 append 的
        if isinstance(perception_res, dict):
            merged_update.update(perception_res)
        else:
            print(f"⚠️ [Parallel] Perception failed: {perception_res}")
    else:
        # 如果没运行视觉，显式重置视觉状态，防止上一轮的残留干扰
        merged_update.update({
            "visual_type": "none",
            "current_image_artifact": None
        })

    return merged_update
