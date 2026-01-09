import base64
import httpx
import io
import re  # <--- 新增
import logging
from PIL import Image
from langchain_core.messages import HumanMessage
from app.core.state import AgentState

# 配置日志
logger = logging.getLogger("Perception")

# 用于在内存中临时缓存已处理的图片尺寸信息，避免重复下载
_IMG_CACHE = {}


def _compress_image(image: Image.Image, max_dimension: int = 1536, quality: int = 85) -> str:
    """图片压缩逻辑 (保持不变)"""
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    width, height = image.size
    max_side = max(width, height)
    if max_side > max_dimension:
        scale_ratio = max_dimension / max_side
        image = image.resize((int(width * scale_ratio), int(height * scale_ratio)), Image.Resampling.LANCZOS)
    output_buffer = io.BytesIO()
    image.save(output_buffer, format="JPEG", quality=quality)
    return base64.b64encode(output_buffer.getvalue()).decode('utf-8')


def _find_image_urls(state: AgentState) -> list:
    """
    查找图片URLs，优先使用当前图片，否则回溯历史消息
    """
    image_urls = state.get("image_urls", [])
    if image_urls:
        return image_urls
    
    # 历史回溯
    msgs = state.get("messages", [])
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            hist_urls = m.additional_kwargs.get("image_urls", [])
            if hist_urls:
                return hist_urls
    
    return []


def _classify_image(image: Image.Image, file_size_kb: float) -> str:
    """
    对图片进行分类：sticker、icon 或 photo
    """
    width, height = image.size
    ratio = width / height if height > 0 else 0
    is_square_ish = 0.8 < ratio < 1.2
    
    if width < 50 or height < 50:
        return "icon"
    elif is_square_ish and (width <= 1024 or height <= 1024 or file_size_kb < 1024):
        logger.info(f"👁️ -> Classified as STICKER ({width}x{height})")
        return "sticker"
    else:
        logger.info(f"👁️ -> Classified as PHOTO. Compressing...")
        return "photo"


async def _download_and_process_image(target_url: str) -> tuple:
    """
    下载并处理图片
    """
    logger.info(f"👁️ [Perception] Downloading: {target_url[:50]}...")
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(target_url, timeout=(3.0, 10.0))
            
            if resp.status_code == 200:
                try:
                    img_bytes = resp.content
                    image = Image.open(io.BytesIO(img_bytes))
                    width, height = image.size
                    file_size_kb = len(img_bytes) / 1024
                    
                    visual_type = _classify_image(image, file_size_kb)
                    
                    # 只对照片进行压缩
                    final_image_data = _compress_image(image) if visual_type == "photo" else None
                    
                    # 更新缓存
                    _IMG_CACHE[target_url] = (visual_type, width, height, file_size_kb)
                    
                    return visual_type, final_image_data
                    
                except Exception as img_err:
                    logger.warning(f"⚠️ [Perception] Image processing error: {img_err}")
                    _IMG_CACHE[target_url] = ("failed", 0, 0, 0)
                    return "error", None
            else:
                logger.warning(f"⚠️ [Perception] Download Failed: HTTP {resp.status_code}.")
                _IMG_CACHE[target_url] = ("failed", 0, 0, 0)
                return "failed", None
                
    except httpx.TimeoutException:
        logger.warning("⚠️ [Perception] Download TIMEOUT. Skipping.")
        _IMG_CACHE[target_url] = ("failed", 0, 0, 0)
        return "timeout", None
    except Exception as e:
        logger.warning(f"⚠️ [Perception] Network error: {e}")
        return "error", None


async def perception_node(state: AgentState) -> dict:
    """
    感知节点：增加缓存与超时优化
    """
    # 查找图片URLs
    image_urls = _find_image_urls(state)
    if not image_urls:
        return {"visual_type": "none", "current_image_artifact": None}
    
    target_url = image_urls[0]
    
    # 过滤非法URL
    if not target_url.startswith("http"):
        return {"visual_type": "none", "current_image_artifact": None}
    
    # 缓存检查
    if target_url in _IMG_CACHE:
        cached_type, w, h, size = _IMG_CACHE[target_url]
        logger.info(f"⚡ [Perception] Cache Hit: {cached_type} ({w}x{h})")
        if cached_type in ["sticker", "icon", "failed"]:
            return {"visual_type": cached_type, "current_image_artifact": None}
    
    # 下载并处理图片
    visual_type, final_image_data = await _download_and_process_image(target_url)
    
    # 构造返回
    updates = {"visual_type": visual_type}
    if visual_type == "photo" and final_image_data:
        updates["current_image_artifact"] = final_image_data
    else:
        updates["current_image_artifact"] = None
    
    return updates
