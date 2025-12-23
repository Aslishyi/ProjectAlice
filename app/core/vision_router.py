import json
from datetime import datetime
from typing import List, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from app.core.config import config

# --- 优化后的路由策略 (Few-Shot Context Aware) ---
ROUTER_SYSTEM_PROMPT = """你是 AI 代理的“视觉皮层”。
你的任务是判断：为了回答用户的最新问题，**是否必须**去看一眼用户的屏幕？

**请分析最近的对话上下文，而不仅仅是最后一句。**

### 🟢 需要看屏幕 (TRUE) 的情况：
1. **直接视觉请求**: "看看这个"、"我的屏幕上是什么"、"帮我读一下这个弹窗"。
2. **代词引用 (Deixis)**: "这行代码报错了"、"那个按钮在哪"、"你能解释一下这个图表吗"。
3. **上下文依赖**: 
   - 用户: (上一句发了图) "这画的是什么？"
   - 用户: "我现在正在看某某网页，怎么操作？"
4. **Debug/纠错**: 用户问 "为什么跑不通？" 且上下文中没有代码文本，暗示代码在屏幕上。

### 🔴 不需要看屏幕 (FALSE) 的情况：
1. **纯知识/闲聊**: "你好"、"讲个笑话"、"Python怎么写Hello World" (通用知识)。
2. **已有上下文**: 用户已经在文本里贴出了代码或报错信息。
3. **主观问题**: "你喜欢什么颜色"、"我是谁"。

**输出格式**: 仅输出 JSON: `{"needs_vision": true}` 或 `{"needs_vision": false}`
"""


class VisionRouter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.SMALL_LLM_MODEL_NAME,  # 建议用小模型如 Qwen-7B 或 GPT-3.5-Turbo 以保证速度
            temperature=0.0,
            max_tokens=60,
            api_key=config.SILICONFLOW_API_KEY,
            base_url=config.SILICONFLOW_BASE_URL
        )

    async def should_see(self, messages: List[BaseMessage]) -> bool:
        """
        :param messages: 最近的对话记录 (List[BaseMessage])
        """
        if not messages: return False

        # 1. 提取最近 3 条交互作为上下文 (避免 token 过多)
        recent_msgs = messages

        # 2. 构造 Prompt 输入
        # 将消息转为简单的文本描述，方便 Router 理解
        context_str = ""
        for m in recent_msgs:
            role = "User" if isinstance(m, HumanMessage) else "AI"
            content = str(m.content)
            # 截断过长的内容
            if len(content) > 100: content = content[:100] + "..."
            context_str += f"{role}: {content}\n"

        final_prompt = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"--- 对话历史 ---\n{context_str}\n\n判断用户最新的一句是否需要视觉支持？")
        ]

        try:
            response = await self.llm.ainvoke(final_prompt)
            content = response.content.strip().replace("```json", "").replace("```", "")
            data = json.loads(content)
            result = data.get("needs_vision", False)

            last_query = recent_msgs[-1].content if recent_msgs else ""
            if len(str(last_query)) > 20: last_query = str(last_query)[:20] + "..."

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] --- [Router] Needs Vision? {result} (Context: {last_query}) ---")
            return result

        except Exception as e:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] [Router Error] {e} -> Defaulting to TRUE (Safety Fallback)")
            return True


vision_router = VisionRouter()
