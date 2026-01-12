# 假设你安装了 openai 包: pip install openai
from openai import OpenAI
from app.core.config import config
from app.tools.base_tool import BaseTool, ToolParam

# client = OpenAI(api_key=config.OPENAI_API_KEY)
client = OpenAI(
    api_key=config.SILICONFLOW_API_KEY,
    base_url=config.SILICONFLOW_BASE_URL
)


class ImageGenTool(BaseTool):
    """图像生成工具"""
    
    name = "generate_image"
    description = "Generate an image based on the text description (prompt). Use this when the user explicitly asks to 'draw', 'paint', or 'generate an image'."
    parameters = [
        ToolParam(
            name="prompt",
            param_type="string",
            description="The image generation prompt",
            required=True
        ),
        ToolParam(
            name="size",
            param_type="string",
            description="The image size (default: 1024x1024)",
            required=False,
            enum_values=["1024x1024", "512x512", "256x256"]
        )
    ]
    available_for_llm = True
    
    async def execute(self, prompt: str, size: str = "1024x1024", **kwargs) -> dict:
        """执行图像生成"""
        try:
            response = await client.images.agenerate(
                model="Qwen/Qwen-Image-Edit-2509",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1,
            )
            return {
                "success": True,
                "result": response.data[0].url,
                "error": ""
            }
        except Exception as e:
            error_msg = f"Image generation failed: {e}"
            return {
                "success": False,
                "result": "",
                "error": error_msg
            }


# 导出工具实例
image_gen_tool = ImageGenTool()


async def generate_image(prompt: str) -> str:
    """兼容旧接口的图像生成函数"""
    result = await image_gen_tool.execute(prompt=prompt)
    return result["result"] if result["success"] else result["error"]
