import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import config
from app.memory.vector_store import vector_db
from app.memory.relation_db import relation_db
from app.utils.cache import cached_llm_invoke

logger = logging.getLogger("SmartMemoryRetrieval")


class MemoryRetrievalTool:
    """智能记忆检索工具，模拟MaiBot的记忆检索功能"""
    
    def __init__(self):
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=config.SMALL_MODEL,
            temperature=0.0,
            api_key=config.SMALL_MODEL_API_KEY,
            base_url=config.SMALL_MODEL_URL
        )
        
        # 初始化记忆检索相关的prompt
        self._init_prompts()
    
    def _init_prompts(self):
        """初始化记忆检索相关的prompt模板"""
        self.question_prompt = """
你的名字是{bot_name}。现在是{time_now}。
群里正在进行的聊天内容：
{chat_history}

现在，{sender}发送了内容:{target_message},你想要回复ta。
请仔细分析聊天内容，考虑以下几点：
1. 对话中是否提到了过去发生的事情、人物、事件或信息
2. 是否有需要回忆的内容（比如"之前说过"、"上次"、"以前"等）
3. 是否有需要查找历史信息的问题
4. 是否有问题可以搜集信息帮助你聊天

重要提示：
- **每次只能提出一个问题**，选择最需要查询的关键问题
- 如果之前已经查询过某个问题并得到了答案，请避免重复生成相同或相似的问题
- 如果不需要检索记忆则输出空数组[]

如果你认为需要从记忆中检索信息来回答，请根据上下文提出**一个**最关键的问题来帮助你回复目标消息，放入"questions"字段

问题格式示例：
- "xxx在前几天干了什么"
- "xxx是什么，在什么时候提到过?"
- "xxxx和xxx的关系是什么"
- "xxx在某个时间点发生了什么"

问题要说明前因后果和上下文，使其全面且精准

输出格式示例：
```json
{{
  "questions": ["张三在前几天干了什么"] #问题数组（字符串数组），如果不需要检索记忆则输出空数组[]，如果需要检索则只输出包含一个问题的数组
}}
```
请只输出JSON对象，不要输出其他内容：
"""
        
        self.react_prompt = """
你的名字是{bot_name}。现在是{time_now}。
你正在参与聊天，你需要搜集信息来回答问题，帮助你参与聊天。
当前需要解答的问题：{question}
已收集的信息：
{collected_info}

**工具说明：**
- 如果涉及过往事件，或者查询某个过去可能提到过的概念，或者某段时间发生的事件。可以使用聊天记录查询工具查询过往事件
- 如果涉及人物，可以使用人物信息查询工具查询人物信息
- 如果没有可靠信息，也可以使用知识库查询，作为辅助信息

**思考**
- 你可以对查询思路给出简短的思考：思考要简短，直接切入要点
- 先思考当前信息是否足够回答问题
- 如果信息不足，则需要使用tool查询信息，你必须给出使用什么工具进行查询
- 如果当前已收集的信息足够或信息不足确定无法找到答案，你必须结束查询
"""
        
        self.final_prompt = """
你的名字是{bot_name}。现在是{time_now}。
你正在参与聊天，你需要根据搜集到的信息判断问题是否可以回答问题。

当前问题：{question}
已收集的信息：
{collected_info}

分析：
- 当前信息是否足够回答问题？
- **如果信息足够且能找到明确答案**，在思考中直接给出答案，格式为：found_answer(answer="你的答案内容")
- **如果信息不足或无法找到答案**，在思考中给出：not_enough_info(reason="信息不足或无法找到答案的原因")

**重要规则：**
- 必须严格使用检索到的信息回答问题，不要编造信息
- 答案必须精简，不要过多解释
- **只有在检索到明确、具体的答案时，才使用found_answer**
- **如果信息不足、无法确定、找不到相关信息，必须使用not_enough_info，不要使用found_answer**
- 答案必须给出，格式为 found_answer(answer="...") 或 not_enough_info(reason="...")。
"""
    
    async def generate_memory_questions(self, chat_history: str, sender: str, target_message: str) -> List[str]:
        """根据聊天历史生成记忆检索问题"""
        try:
            # 构建prompt
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            prompt = self.question_prompt.format(
                bot_name="Alice",
                time_now=current_time,
                chat_history=chat_history,
                sender=sender,
                target_message=target_message
            )
            
            # 调用LLM生成问题
            response = await cached_llm_invoke(
                self.llm,
                [SystemMessage(content=prompt)],
                temperature=0.0,
                conversation_type="private",
                query_type="memory_retrieval_question"
            )
            
            # 处理响应
            if isinstance(response, str):
                response_content = response.strip()
            else:
                response_content = response.content.strip()
            
            # 清理响应内容，去除可能的代码块标记和无关内容
            if response_content.startswith('```json'):
                response_content = response_content[7:]
            if response_content.endswith('```'):
                response_content = response_content[:-3]
            # 去除首尾空白
            response_content = response_content.strip()
            
            # 解析JSON
            response_json = json.loads(response_content)
            return response_json.get("questions", [])
            
        except Exception as e:
            logger.error(f"生成记忆检索问题失败: {e}")
            return []
    
    async def retrieve_memory(self, question: str, chat_history: str = "") -> Tuple[bool, str]:
        """根据问题检索记忆"""
        try:
            # 步骤1: 从向量数据库检索相关记忆
            vector_results = await vector_db.search(question, k=5)
            vector_info = "\n".join(vector_results)
            
            # 步骤2: 如果需要，从关系数据库检索人物信息
            person_info = await self._retrieve_person_info(question)
            
            # 步骤3: 综合所有信息，判断是否足够回答问题
            collected_info = f"向量数据库检索结果:\n{vector_info}\n\n人物信息:\n{person_info}"
            
            # 步骤4: 使用LLM判断信息是否足够并生成答案
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            prompt = self.final_prompt.format(
                bot_name="Alice",
                time_now=current_time,
                question=question,
                collected_info=collected_info
            )
            
            response = await cached_llm_invoke(
                self.llm,
                [SystemMessage(content=prompt)],
                temperature=0.0,
                conversation_type="private",
                query_type="memory_retrieval_final"
            )
            
            # 处理响应
            if isinstance(response, str):
                response_content = response.strip()
            else:
                response_content = response.content.strip()
            
            # 解析答案
            if "found_answer(answer=" in response_content:
                # 提取答案
                answer_start = response_content.find("found_answer(answer=") + len("found_answer(answer=")
                answer_end = response_content.find(")", answer_start)
                answer = response_content[answer_start:answer_end].strip('"')
                return True, answer
            else:
                # 提取原因
                reason_start = response_content.find("not_enough_info(reason=") + len("not_enough_info(reason=")
                reason_end = response_content.find(")", reason_start)
                reason = response_content[reason_start:reason_end].strip('"')
                return False, reason
                
        except Exception as e:
            logger.error(f"检索记忆失败: {e}")
            return False, str(e)
    
    async def _retrieve_person_info(self, question: str) -> str:
        """从问题中提取人物信息并检索"""
        try:
            # 简单实现：提取问题中的所有中文人名
            # 这是一个简化的实现，实际可以使用更复杂的命名实体识别
            import re
            # 匹配中文人名（简化版）
            person_names = re.findall(r'[\u4e00-\u9fa5]{2,4}', question)
            
            if not person_names:
                return "未识别到人物"
            
            # 去重
            unique_names = list(set(person_names))
            
            # 检索每个人物的信息
            person_info = []
            for name in unique_names:
                try:
                    # 从关系数据库检索人物信息
                    user_profile = await relation_db.get_user_profile_by_name(name)
                    if user_profile:
                        person_info.append(f"{name}: 亲密程度={user_profile.relationship.intimacy}, 熟悉程度={user_profile.relationship.familiarity}")
                except Exception:
                    continue
            
            if person_info:
                return "\n".join(person_info)
            else:
                return "未找到相关人物信息"
                
        except Exception as e:
            logger.error(f"检索人物信息失败: {e}")
            return "检索人物信息出错"
    
    async def smart_retrieve_for_query(self, query: str, chat_history: str, sender: str, user_id: str) -> Dict[str, Any]:
        """为给定查询执行智能记忆检索"""
        # 步骤1: 生成记忆检索问题
        questions = await self.generate_memory_questions(chat_history, sender, query)
        
        if not questions:
            return {
                "has_relevant_memory": False,
                "memory_content": "",
                "questions": []
            }
        
        # 步骤2: 检索记忆
        retrieved_info = []
        for question in questions:
            found, content = await self.retrieve_memory(question, chat_history)
            if found:
                retrieved_info.append(content)
        
        if retrieved_info:
            return {
                "has_relevant_memory": True,
                "memory_content": "\n".join(retrieved_info),
                "questions": questions
            }
        else:
            return {
                "has_relevant_memory": False,
                "memory_content": "",
                "questions": questions
            }


# 全局智能记忆检索工具实例
smart_memory_retriever = None

def get_smart_memory_retriever() -> Optional[MemoryRetrievalTool]:
    """获取全局智能记忆检索工具实例"""
    global smart_memory_retriever
    if smart_memory_retriever is None:
        smart_memory_retriever = MemoryRetrievalTool()
    return smart_memory_retriever


def initialize_smart_memory_retrieval() -> bool:
    """初始化智能记忆检索工具"""
    try:
        global smart_memory_retriever
        smart_memory_retriever = MemoryRetrievalTool()
        logger.info("✅ 智能记忆检索工具初始化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 初始化智能记忆检索工具失败: {e}")
        return False