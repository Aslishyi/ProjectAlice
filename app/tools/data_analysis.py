from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool

repl = PythonREPL()


@tool
def run_python_analysis(code: str) -> str:
    """
    Execute Python code to perform data analysis, math calculations, or string processing.
    Input should be valid Python code. The code should print() the final result.
    """
    try:
        # 为了安全，这里可以添加简单的静态分析，禁止 import os, sys 等
        if "import os" in code or "import sys" in code:
            return "Security Alert: System modules are restricted."

        result = repl.run(code)
        return f"Execution Result:\n{result}"
    except Exception as e:
        return f"Python Error: {e}"
