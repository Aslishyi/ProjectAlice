from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# 初始化 Tavily 客户端
_search = TavilySearchResults(max_results=5)

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
