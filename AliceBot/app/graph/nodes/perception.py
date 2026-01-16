import base64
import httpx
import io
import re  # <--- 新增
import logging
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.state import AgentState
from app.core.config import config
from app.plugins.emoji_plugin.emoji_manager import get_emoji_manager
from app.utils.cache import cached_llm_invoke
from langchain_openai import ChatOpenAI
from typing import List, Optional, Dict, Tuple, Any

# 初始化LLM实例
llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=0.3,  # 使用较低的temperature以获得更稳定的分类结果
    api_key=config.MODEL_API_KEY,
    base_url=config.MODEL_URL
)

# 配置日志
logger = logging.getLogger("Perception")

# 用于在内存中临时缓存已处理的图片尺寸信息，避免重复下载
_IMG_CACHE = {}


async def _process_image_with_llm(base64_data: str) -> tuple[bool, dict]:
    """
    使用大模型同时完成图片是否为表情包的判断和分析
    
    Args:
        base64_data: 图片的base64编码数据
        
    Returns:
        tuple[bool, dict]: (是否为表情包, 分析结果)
        分析结果包含: emotions (情绪标签), description (描述), category (分类)
    """
    try:
        logger.info(f"🎨 [Perception] 开始使用大模型判断和分析图片")
        
        # 构造系统提示词 - 整合判断和分析功能
        system_prompt = ("你是一个专业的表情包分析专家，具有丰富的网络文化知识和情感分析能力。\n" 
                        "请仔细观察图片内容，完成以下任务：\n" 
                        "\n" 
                        "1. 首先判断这张图片是否为表情包（sticker）\n" 
                        "   - 表情包的定义：\n" 
                        "     * 通常是具有夸张表情、动作或文字的图片\n" 
                        "     * 用于在聊天中表达情感或调侃\n" 
                        "     * 通常具有卡通风格或经过特殊处理\n" 
                        "     * 尺寸通常较小，比例接近正方形\n" 
                        "   - 普通图片的定义：\n" 
                        "     * 真实的照片（如风景、人物、食物等）\n" 
                        "     * 没有明显的夸张表情或动作\n" 
                        "     * 通常用于记录真实场景\n" 
                        "\n" 
                        "2. 如果是表情包，请从以下几个方面分析：\n" 
                        "   - 情绪标签：精确识别表情包传达的核心情绪，使用中文关键词，最多5个，按情绪强度排序\n" 
                        "   - 描述：简洁明了地描述表情包的视觉内容和核心元素，不超过50字\n" 
                        "   - 分类：从以下选项中选择唯一最合适的：表情符号、人物形象、动物植物、场景生活、文字梗图、其他\n" 
                        "\n" 
                        "请严格按照以下JSON格式输出，不要添加任何额外内容、解释或说明：\n" 
                        "{\"is_emoji\": true/false, \"emotions\": [\"情绪标签1\", \"情绪标签2\"], \"description\": \"描述内容\", \"category\": \"分类名称\"}")
        
        # 构造用户消息，使用正确的多模态格式
        message_content: list[str | dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}},
            {"type": "text", "text": "请判断这张图片是否为表情包，如果是，请生成情绪标签、描述和分类信息。"}
        ]
        
        # 构造消息列表
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message_content)
        ]
        
        # 调用LLM
        response = await cached_llm_invoke(
            llm, 
            messages, 
            temperature=0.2,  # 适中的温度以获得精确且丰富的分析结果
            query_type="image_classification_and_analysis"
        )
        
        # 处理响应
        response_content: str
        if isinstance(response, str):
            response_content = response.strip()
        else:
            response_content = response.content.strip()
        
        logger.info(f"🎨 [Perception] LLM响应: {response_content[:150]}...")
        
        # 解析JSON响应
        import json
        import re
        
        # 提取Markdown JSON
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_content, re.DOTALL)
        if match:
            json_content = match.group(1)
        else:
            # 尝试找到JSON的开始和结束位置
            start = response_content.find("{")
            end = response_content.rfind("}")
            if start != -1 and end != -1:
                json_content = response_content[start: end + 1]
            else:
                json_content = response_content
        
        try:
            result = json.loads(json_content)
            
            # 验证并清理结果
            is_emoji = result.get("is_emoji", False)
            valid_result: dict[str, Any] = {}
            
            # 如果是表情包，验证情绪标签、描述和分类
            if is_emoji:
                allowed_categories = ["表情符号", "人物形象", "动物植物", "场景生活", "文字梗图", "其他"]
                
                # 处理情绪标签
                emotions = result.get("emotions", [])
                if isinstance(emotions, list) and emotions:
                    # 过滤空标签并确保是字符串类型
                    valid_emotions = [str(e).strip() for e in emotions if e and isinstance(e, (str, int, float))]
                    # 限制最多5个标签
                    valid_result["emotions"] = valid_emotions[:5]
                else:
                    valid_result["emotions"] = ["未知"]
                
                # 处理描述
                description = result.get("description", "")
                if isinstance(description, str) and description.strip():
                    valid_result["description"] = description.strip()[:50]  # 限制50字
                else:
                    valid_result["description"] = ""
                
                # 处理分类
                category = result.get("category", "其他")
                if isinstance(category, str) and category in allowed_categories:
                    valid_result["category"] = category
                else:
                    valid_result["category"] = "其他"
            
            logger.info(f"🎨 [Perception] LLM判断结果: {'是表情包' if is_emoji else '不是表情包'}")
            if is_emoji:
                logger.info(f"🎨 [Perception] LLM分析结果 (已验证): {valid_result}")
            
            return is_emoji, valid_result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ [Perception] JSON解析失败: {e}, 处理后的内容: {json_content[:100]}...")
            # 失败时返回默认值
            return False, {
                "emotions": ["未知"],
                "description": "",
                "category": "其他"
            }
            
    except Exception as e:
        logger.error(f"❌ [Perception] LLM判断和分析图片失败: {e}")
        # 失败时返回默认值
        return False, {
            "emotions": ["未知"],
            "description": "",
            "category": "其他"
        }


# 保留原有函数作为兼容层
async def _is_emoji_with_llm(base64_data: str) -> bool:
    """
    使用大模型判断图片是否为表情包（兼容旧接口）
    
    Args:
        base64_data: 图片的base64编码数据
        
    Returns:
        bool: 是否为表情包
    """
    is_emoji, _ = await _process_image_with_llm(base64_data)
    return is_emoji


async def _analyze_emoji_with_llm(base64_data: str) -> dict:
    """
    使用大模型分析表情包（兼容旧接口）
    
    Args:
        base64_data: 图片的base64编码数据
        
    Returns:
        dict: 包含情绪标签、描述和分类的字典
    """
    _, analysis = await _process_image_with_llm(base64_data)
    return analysis



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



async def _classify_image(image: Image.Image, file_size_kb: float) -> str:
    """
    对图片进行分类：sticker、icon 或 photo
    
    使用大模型API进行判断，提高分类准确率
    """
    width, height = image.size
    ratio = width / height if height > 0 else 0
    
    # 小图标判断 - 仍然使用本地规则，因为小图标明显不是表情包
    if width < 50 or height < 50:
        logger.info(f"👁️ -> Classified as ICON ({width}x{height}, {file_size_kb:.1f}KB)")
        return "icon"
    
    # 将图片转换为base64，用于大模型判断
    try:
        import io
        import base64
        
        # 保存图片到字节流
        buffer = io.BytesIO()
        image_format = image.format or "JPEG"
        if image.mode in ('RGBA', 'LA'):
            # 对于有透明通道的图片，使用PNG格式
            image_format = "PNG"
        image.save(buffer, format=image_format)
        buffer.seek(0)
        
        # 转换为base64
        base64_data = base64.b64encode(buffer.read()).decode('utf-8')
        
        # 使用大模型同时进行判断和分析
        is_emoji, _ = await _process_image_with_llm(base64_data)
        
        if is_emoji:
            logger.info(f"👁️ -> LLM Classified as STICKER ({width}x{height}, {file_size_kb:.1f}KB, ratio: {ratio:.2f})")
            return "sticker"
        else:
            logger.info(f"👁️ -> LLM Classified as PHOTO ({width}x{height}, {file_size_kb:.1f}KB, ratio: {ratio:.2f})")
            return "photo"
            
    except Exception as e:
        logger.error(f"❌ 大模型分类失败，使用本地备份规则: {e}")
        # 出错时使用本地备份逻辑
        try:
            has_transparency = image.mode in ('RGBA', 'LA') or ('transparency' in image.info)
            is_square_ish = 0.5 < ratio < 1.6
            is_small_to_medium = 100 <= width <= 1024 and 100 <= height <= 1024
            is_small_file = file_size_kb < 1024  # 小于1MB
            has_sticker_characteristics = (is_square_ish and (has_transparency or is_small_file or is_small_to_medium))
            
            if has_sticker_characteristics:
                logger.info(f"👁️ -> Backup Rule Classified as STICKER ({width}x{height}, {file_size_kb:.1f}KB, ratio: {ratio:.2f})")
                return "sticker"
            else:
                logger.info(f"👁️ -> Backup Rule Classified as PHOTO ({width}x{height}, {file_size_kb:.1f}KB, ratio: {ratio:.2f})")
                return "photo"
        except Exception as backup_e:
            logger.error(f"❌ 本地备份规则也失败: {backup_e}")
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
                    
                    visual_type = await _classify_image(image, file_size_kb)
                    
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
    感知节点：增加缓存与超时优化，支持智能处理多张图片
    """
    # 查找图片URLs
    image_urls = _find_image_urls(state)
    if not image_urls:
        return {"visual_type": "none", "current_image_artifact": None}
    
    # 过滤非法URL
    valid_image_urls = [url for url in image_urls if url.startswith("http")]
    if not valid_image_urls:
        return {"visual_type": "none", "current_image_artifact": None}
    
    # 智能选择需要处理的图片
    processed_images = []
    photos = []
    stickers = []
    
    # 首先对所有图片进行初步分类（使用缓存或快速分类）
    for url in valid_image_urls:
        if url in _IMG_CACHE:
            cached_type, w, h, size = _IMG_CACHE[url]
            if cached_type == "photo":
                photos.append((url, cached_type))
            elif cached_type == "sticker":
                stickers.append((url, cached_type))
        else:
            # 对于未缓存的图片，先快速下载并分类
            visual_type, _ = await _download_and_process_image(url)
            if visual_type == "photo":
                photos.append((url, visual_type))
            elif visual_type == "sticker":
                stickers.append((url, visual_type))
    
    # 决定处理哪些图片
    # 1. 优先处理所有照片类型的图片（通常包含重要信息）
    # 2. 对于表情包，最多处理2张代表性的
    # 3. 总处理图片数不超过5张，避免性能问题
    target_images = []
    
    # 添加所有照片
    for photo_url, _ in photos:
        target_images.append(photo_url)
    
    # 添加最多2张表情包
    for sticker_url, _ in stickers[:2]:
        target_images.append(sticker_url)
    
    # 限制总数量
    target_images = target_images[:5]
    
    # 处理选中的图片
    processed_image_data: list[dict[str, Any]] = []
    main_visual_type = "none"
    main_image_artifact = None
    all_image_artifacts = []
    
    for i, target_url in enumerate(target_images):
        # 缓存检查
        if target_url in _IMG_CACHE:
            cached_type, w, h, size = _IMG_CACHE[target_url]
            logger.info(f"⚡ [Perception] Cache Hit: {cached_type} ({w}x{h}) - Image {i+1}/{len(target_images)}")
            if cached_type == "photo":
                # 下载并处理照片，获取完整的image_artifact
                _, final_image_data = await _download_and_process_image(target_url)
                all_image_artifacts.append({
                    "type": cached_type,
                    "data": final_image_data
                })
                if not main_image_artifact:
                    main_image_artifact = final_image_data
                    main_visual_type = cached_type
            elif cached_type == "sticker":
                if not main_visual_type:
                    main_visual_type = cached_type
        else:
            # 下载并处理图片
            visual_type, final_image_data = await _download_and_process_image(target_url)
            
            if visual_type == "photo":
                all_image_artifacts.append({
                    "type": visual_type,
                    "data": final_image_data
                })
                if not main_image_artifact:
                    main_image_artifact = final_image_data
                    main_visual_type = visual_type
            elif visual_type == "sticker" and not main_visual_type:
                main_visual_type = visual_type
        
        # 记录处理的图片
        processed_images.append({
            "url": target_url,
            "type": visual_type if 'visual_type' in locals() else _IMG_CACHE.get(target_url, ("unknown",))[0]
        })
    
    # 记录处理信息
    logger.info(f"📸 [Perception] Processed {len(processed_images)}/{len(valid_image_urls)} images")
    
    # 构造返回
    updates = {
        "visual_type": main_visual_type,
        "current_image_artifact": main_image_artifact,
        "all_image_artifacts": all_image_artifacts,  # 包含所有处理过的图片数据
        "processed_images": processed_images  # 记录所有处理过的图片信息
    }
    
    return updates
