from langchain_experimental.utilities import PythonREPL
from app.tools.base_tool import BaseTool, ToolParam
import asyncio

repl = PythonREPL()


class DataAnalysisTool(BaseTool):
    """数据分���工具"""
    
    name = "run_python_analysis"
    description = "Execute Python code to perform data analysis, math calculations, or string processing. Input should be valid Python code. The code should print() the final result."
    parameters = [
        ToolParam(
            name="code",
            param_type="string",
            description="The Python code to execute",
            required=True
        )
    ]
    available_for_llm = True
    
    async def execute(self, code: str, **kwargs) -> dict:
        """执行Python代码分析"""
        try:
            # 安全检查：禁止导入危险模块
            dangerous_imports = ["import os", "import sys", "import subprocess", "import shutil"]
            for dangerous_import in dangerous_imports:
                if dangerous_import in code:
                    return {
                        "success": False,
                        "result": "",
                        "error": "Security Alert: System modules are restricted."
                    }
            
            # 在单独的执行器中运行同步代码，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, repl.run, code)
            
            return {
                "success": True,
                "result": f"Execution Result:\n{result}",
                "error": ""
            }
        except Exception as e:
            error_msg = f"Python Error: {e}"
            return {
                "success": False,
                "result": "",
                "error": error_msg
            }


# 导出工具实例
data_analysis_tool = DataAnalysisTool()


async def run_python_analysis(code: str) -> str:
    """兼容旧接口的数据分析函数"""
    result = await data_analysis_tool.execute(code=code)
    return result["result"] if result["success"] else result["error"]
