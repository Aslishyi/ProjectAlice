from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from app.core.config import config
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 Tavily 客户端，确保使用正确的 API Key
_search = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5
)

@tool
def perform_web_search(query: str) -> str:
    """
    Search the web for up-to-date information, news, or factual verification.
    Use this when you don't know the answer or need current events.
    """
    try:
        # 结果是 list[dict]，转化为字符串
        results = _search.invoke(query)
        return str(results)
    except Exception as e:
        return f"Search failed: {e}"
