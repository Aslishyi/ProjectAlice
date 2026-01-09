# 假设你安装了 openai 包: pip install openai
from openai import OpenAI
from app.core.config import config

# client = OpenAI(api_key=config.OPENAI_API_KEY)
client = OpenAI(
    api_key=config.SILICONFLOW_API_KEY,
    base_url=config.SILICONFLOW_BASE_URL
)

async def generate_image(prompt: str) -> str:
    """
    Generate an image based on the text description (prompt).
    Use this when the user explicitly asks to 'draw', 'paint', or 'generate an image'.
    Returns the URL of the generated image.
    """
    try:
        response = await client.images.agenerate(
            model="Qwen/Qwen-Image-Edit-2509",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        return f"Image generation failed: {e}"
