import asyncio
import os
import sys
import threading
import time
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.graph_builder import build_graph
from app.core.state import AgentState, EmotionData
from app.utils.safety import safety_filter
from app.memory.relation_db import relation_db
from app.monitor.screen_monitor import ScreenMonitor

console = Console()


class UserInputEvent:
    def __init__(self, text: str):
        self.text = text


def print_agent_thought(monologue: str, emotion: EmotionData):
    if not monologue or monologue == "N/A": return
    console.print(f"[dim italic]Anima 思考: {monologue} (Mood: {emotion.current_mood})[/dim]")


async def main():
    console.clear()
    console.rule("[bold magenta]Project Anima – High Performance[/bold magenta]")

    user_id = console.input("[bold yellow]User ID: [/] ").strip() or "master"

    # --- 状态初始化 ---
    app_state: Dict[str, Any] = {
        "current_user_id": user_id,
        "user_profile": relation_db.get_user_profile(user_id),
        "global_relationship_graph": relation_db.get_all_relationships(),
        "emotion": EmotionData(current_mood="Calm", valence=0.1, arousal=0.4),
        "messages": [],
        "current_activity": "Idle",
        "activity_start_time": time.time(),
        "last_visual_summary": "",
        "last_interaction_ts": 0.0
    }

    graph = build_graph()
    event_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # --- 1. 启动监控 (修正参数) ---
    # interval=1.0: 每秒检查一次
    # stability_duration=2.0: 画面必须静止2秒才触发，防止看视频时疯狂触发
    monitor = ScreenMonitor(event_queue, interval=1.0, diff_threshold=5.0, stability_duration=2.0)

    # --- 2. 输入线程 ---
    def user_input_worker():
        while True:
            try:
                text = input()  # 阻塞式输入
                if not text: continue
                if text.lower() in ["quit", "exit"]:
                    os._exit(0)

                # 安全检查
                is_safe, _ = safety_filter.check_input(text)
                if not is_safe:
                    console.print("[red]安全拦截[/red]")
                    continue

                asyncio.run_coroutine_threadsafe(event_queue.put(UserInputEvent(text)), loop)
            except:
                break

    threading.Thread(target=user_input_worker, daemon=True).start()
    monitor.start()

    console.print(f"[green]系统已就绪。Anima 正在潜伏...[/green]")

    try:
        while True:
            # --- 3. 智能事件获取与去重 ---
            # 优先处理用户输入；如果队列里有多个屏幕事件，只取最新的一个，丢弃旧的！
            events = []
            try:
                # 阻塞等待第一个事件
                events.append(await event_queue.get())

                # 检查队列里还有没有积压的？
                while not event_queue.empty():
                    evt = event_queue.get_nowait()
                    # 如果是用户输入，必须保留
                    if isinstance(evt, UserInputEvent):
                        events.append(evt)
                    # 如果是屏幕事件，且列表中已经有一个屏幕事件了，覆盖它（只保留最新的）
                    elif isinstance(evt, dict) and evt.get('type') == 'screen_event':
                        # 移除列表中已有的旧屏幕事件
                        events = [e for e in events if not (isinstance(e, dict) and e.get('type') == 'screen_event')]
                        events.append(evt)
                    event_queue.task_done()
            except Exception:
                pass

            # 逐个处理去重后的事件
            for event in events:
                inputs = {**app_state}  # 浅拷贝当前状态
                inputs["visual_input"] = None
                inputs["is_proactive"] = False

                if isinstance(event, UserInputEvent):
                    console.print(f"\n[bold white]You:[/bold white] {event.text}")
                    inputs["messages"] = app_state["messages"] + [HumanMessage(content=event.text)]

                elif isinstance(event, dict) and event.get('type') == 'screen_event':
                    # 视觉事件触发
                    # 只有当距离上次交互超过一定时间，才允许视觉触发主动交互
                    if time.time() - app_state["last_interaction_ts"] < 10:
                        # console.print("[dim]冷却中，忽略视觉变化[/dim]")
                        continue

                    inputs["visual_input"] = event['data']
                    inputs["is_proactive"] = True
                    console.print("[dim]>> 捕捉到屏幕变化，Anima 正在观察...[/dim]")

                # --- 4. 执行 Graph ---
                # 使用 stream 模式
                async for output in graph.astream(inputs):
                    for node_name, node_val in output.items():
                        # 更新全局状态
                        if "messages" in node_val:
                            app_state["messages"] = node_val["messages"]
                        if "emotion" in node_val:
                            app_state["emotion"] = node_val["emotion"]
                        if "current_activity" in node_val:
                            app_state["current_activity"] = node_val["current_activity"]
                        if "last_visual_summary" in node_val:
                            app_state["last_visual_summary"] = node_val["last_visual_summary"]

                        # UI 反馈
                        if node_name == "reasoning":
                            print_agent_thought(node_val.get("internal_monologue"), app_state["emotion"])

                        elif node_name == "response":
                            # 打印回复
                            last_msg = node_val["messages"][-1]
                            console.print(Panel(last_msg.content, title="Anima", border_style="cyan"))
                            app_state["last_interaction_ts"] = time.time()

                        elif node_name == "proactive":
                            # 如果主动决策决定不说话，打印原因
                            if not node_val.get("should_speak", True):
                                # console.print("[dim]Anima 决定保持沉默[/dim]")
                                pass

                # 标记任务完成
                if isinstance(event, UserInputEvent) or (isinstance(event, dict)):
                    pass  # 这里的 task_done 需要和 get 次数对应，上面逻辑已简化，可忽略

    except KeyboardInterrupt:
        console.print("\n[yellow]Bye![/yellow]")
    finally:
        monitor.stop()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
