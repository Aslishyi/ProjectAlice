from langchain_tavily import TavilySearch
from app.core.config import config
from app.tools.base_tool import BaseTool, ToolParam
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 Tavily 客户端，确保使用正确的 API Key
_search = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5
)


class WebSearchTool(BaseTool):
    """网络搜索工具"""
    
    name = "web_search"
    description = "Search the web for up-to-date information, news, or factual verification. Use this when you don't know the answer or need current events."
    parameters = [
        ToolParam(
            name="query",
            param_type="string",
            description="The search query",
            required=True
        )
    ]
    available_for_llm = True
    
    async def execute(self, query: str, **kwargs) -> dict:
        """执行网络搜索"""
        try:
            # 结果是 list[dict]，转化为字符串
            results = await _search.ainvoke(query)
            return {
                "success": True,
                "result": str(results),
                "error": ""
            }
        except Exception as e:
            error_msg = f"Search failed: {e}"
            return {
                "success": False,
                "result": "",
                "error": error_msg
            }


# 导出工具实例
web_search_tool = WebSearchTool()


async def perform_web_search(query: str) -> str:
    """兼容旧接口的网络搜索函数"""
    result = await web_search_tool.execute(query=query)
    return result["result"] if result["success"] else result["error"]
