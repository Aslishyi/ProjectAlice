import base64
import httpx
import io
import re  # <--- 新增
from PIL import Image
from langchain_core.messages import HumanMessage
from app.core.state import AgentState

# ... (保持 _IMG_CACHE 和 _compress_image 不变) ...
# 请保留原文件中的 _IMG_CACHE = {} 和 _compress_image 函数代码
# 此处为了篇幅省略，请确保文件中存在这些代码

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


async def perception_node(state: AgentState) -> dict:
    """
    感知节点：增加缓存与超时优化
    """
    image_urls = state.get("image_urls", [])

    # 1. 历史回溯 (如果没有新图)
    # 优化：Router 已经做过判断了，这里只负责找 URL
    if not image_urls:
        msgs = state.get("messages", [])
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                hist_urls = m.additional_kwargs.get("image_urls", [])
                if hist_urls:
                    image_urls = hist_urls
                    break
            if len(image_urls) > 0: break

    # 如果回溯也没找到，直接返回
    if not image_urls:
        return {"visual_type": "none", "current_image_artifact": None}

    final_image_data = None
    visual_type = "none"

    if image_urls:
        target_url = image_urls[0]

        # --- 🚀 [额外优化] 过滤掉显然非法的 URL (例如本地路径或空字符串) ---
        if not target_url.startswith("http"):
            return {"visual_type": "none", "current_image_artifact": None}

        # --- 🚀 [优化 1] 缓存命中检查 ---
        if target_url in _IMG_CACHE:
            cached_type, w, h, size = _IMG_CACHE[target_url]
            print(f"⚡ [Perception] Cache Hit: {cached_type} ({w}x{h})")
            if cached_type in ["sticker", "icon", "failed"]:
                return {"visual_type": cached_type, "current_image_artifact": None}
            # 注意：如果之前是 photo，这里需要重新下载吗？
            # 实际上，最好把 Base64 也缓存起来 (LRU Cache)，但为了内存安全，这里还是重新下载吧
            # 只要 Router 起作用了，这里的重新下载频率会非常低。

        print(f"👁️ [Perception] Downloading: {target_url[:50]}...")

        try:
            # --- 🚀 [优化 2] 缩短超时时间 ---
            async with httpx.AsyncClient() as client:
                # QQ 图片服务器有时候会因为链接过期卡住，设置较短的 connect timeout
                resp = await client.get(target_url, timeout=(3.0, 10.0))

                if resp.status_code == 200:
                    try:
                        img_bytes = resp.content
                        image = Image.open(io.BytesIO(img_bytes))
                        width, height = image.size
                        file_size_kb = len(img_bytes) / 1024

                        ratio = width / height if height > 0 else 0
                        is_square_ish = 0.8 < ratio < 1.2

                        # --- 分类逻辑 ---
                        if width < 50 or height < 50:
                            visual_type = "icon"
                        elif is_square_ish and (width <= 1024 or height <= 1024 or file_size_kb < 1024):
                            visual_type = "sticker"
                            print(f"👁️ -> Classified as STICKER ({width}x{height})")
                        else:
                            visual_type = "photo"
                            print(f"👁️ -> Classified as PHOTO. Compressing...")
                            final_image_data = _compress_image(image)

                        # 更新缓存
                        _IMG_CACHE[target_url] = (visual_type, width, height, file_size_kb)

                    except Exception as img_err:
                        print(f"⚠️ [Perception] Image processing error: {img_err}")
                        visual_type = "error"
                        _IMG_CACHE[target_url] = ("failed", 0, 0, 0)

                else:
                    # 如果是 403/404，说明图片过期了
                    print(f"⚠️ [Perception] Download Failed: HTTP {resp.status_code}.")
                    visual_type = "failed"
                    _IMG_CACHE[target_url] = ("failed", 0, 0, 0)

        except httpx.TimeoutException:
            print("⚠️ [Perception] Download TIMEOUT. Skipping.")
            visual_type = "timeout"
            _IMG_CACHE[target_url] = ("failed", 0, 0, 0)
        except Exception as e:
            print(f"⚠️ [Perception] Network error: {e}")
            visual_type = "error"

    # 构造返回
    updates = {"visual_type": visual_type}
    if visual_type == "photo" and final_image_data:
        updates["current_image_artifact"] = final_image_data
    else:
        updates["current_image_artifact"] = None

    return updates
