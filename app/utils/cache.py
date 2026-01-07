import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Deque
from collections import deque
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMCache:
    """
    LLM调用缓存系统
    用于缓存LLM调用的请求和响应，减少重复调用，提高性能
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        初始化缓存系统
        
        Args:
            max_size: 缓存的最大条目数
            default_ttl: 默认的缓存过期时间（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.lock = asyncio.Lock()
    
    def _generate_key(self, messages: List[BaseMessage], model: str, temperature: float) -> str:
        """
        根据输入消息生成唯一的缓存键
        
        Args:
            messages: LLM调用的输入消息列表
            model: 使用的模型名称
            temperature: 模型的温度参数
            
        Returns:
            唯一的缓存键字符串
        """
        # 将消息转换为可哈希的字符串表示
        message_strs = []
        for msg in messages:
            msg_dict = {
                "type": msg.__class__.__name__,  # 消息类型
                "content": msg.content,  # 消息内容
                "additional_kwargs": msg.additional_kwargs,  # 附加参数
            }
            message_strs.append(json.dumps(msg_dict, sort_keys=True, ensure_ascii=False))
        
        # 添加模型和温度参数
        cache_key_data = {
            "messages": message_strs,
            "model": model,
            "temperature": temperature,
        }
        
        # 使用SHA256生成哈希值作为缓存键
        cache_key_str = json.dumps(cache_key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(cache_key_str.encode('utf-8')).hexdigest()
    
    async def get(self, messages: List[BaseMessage], model: str, temperature: float) -> Optional[Any]:
        """
        从缓存中获取LLM调用结果
        
        Args:
            messages: LLM调用的输入消息列表
            model: 使用的模型名称
            temperature: 模型的温度参数
            
        Returns:
            缓存的LLM响应，如果没有缓存或已过期则返回None
        """
        cache_key = self._generate_key(messages, model, temperature)
        
        async with self.lock:
            if cache_key in self.cache:
                value, expire_time = self.cache[cache_key]
                if datetime.now() < expire_time:
                    return value
                else:
                    # 缓存已过期，删除
                    del self.cache[cache_key]
            return None
    
    async def set(self, messages: List[BaseMessage], model: str, temperature: float, value: Any, ttl: Optional[int] = None) -> None:
        """
        将LLM调用结果存入缓存
        
        Args:
            messages: LLM调用的输入消息列表
            model: 使用的模型名称
            temperature: 模型的温度参数
            value: LLM的响应结果
            ttl: 缓存过期时间（秒），如果为None则使用默认值
        """
        cache_key = self._generate_key(messages, model, temperature)
        expire_time = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        
        async with self.lock:
            # 检查缓存大小，如果超过最大值则清理最旧的条目
            if len(self.cache) >= self.max_size:
                # 按过期时间排序，删除最早过期的条目
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[cache_key] = (value, expire_time)
    
    async def clear(self) -> None:
        """
        清空缓存
        """
        async with self.lock:
            self.cache.clear()
    
    async def remove_expired(self) -> int:
        """
        清理所有过期的缓存条目
        
        Returns:
            清理的过期条目数量
        """
        now = datetime.now()
        expired_keys = []
        
        async with self.lock:
            for key, (_, expire_time) in self.cache.items():
                if now >= expire_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含缓存统计信息的字典
        """
        now = datetime.now()
        total = len(self.cache)
        expired = 0
        size_bytes = 0
        
        async with self.lock:
            for _, (value, expire_time) in self.cache.items():
                if now >= expire_time:
                    expired += 1
                
                # 估算缓存大小
                value_str = str(value)
                size_bytes += len(value_str.encode('utf-8'))
        
        return {
            "total_entries": total,
            "expired_entries": expired,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl
        }


class LLMRequestQueue:
    """
    LLM请求队列系统
    用于管理LLM调用请求，控制并发数，防止请求堆积和超时
    """
    
    def __init__(self, max_concurrent: int = 5, timeout: int = 30):
        """
        初始化请求队列
        
        Args:
            max_concurrent: 最大并发请求数
            timeout: 请求超时时间（秒）
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.queue: Deque = deque()
        self.current_concurrent = 0
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def add_request(self, llm: Any, messages: List[BaseMessage], temperature: float = 0.7) -> Any:
        """
        添加LLM请求到队列并等待结果
        
        Args:
            llm: LLM实例
            messages: 输入消息列表
            temperature: 温度参数
            
        Returns:
            LLM响应结果
        """
        async with self.semaphore:
            try:
                # 使用asyncio.wait_for设置请求超时
                result = await asyncio.wait_for(
                    llm.ainvoke(messages),
                    timeout=self.timeout
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"LLM请求超时，已超过{self.timeout}秒")
                raise
            except Exception as e:
                logger.error(f"LLM请求执行出错: {str(e)}")
                raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        获取队列统计信息
        
        Returns:
            包含队列统计信息的字典
        """
        async with self.lock:
            return {
                "queue_length": len(self.queue),
                "current_concurrent": self.current_concurrent,
                "max_concurrent": self.max_concurrent,
                "timeout": self.timeout
            }


# 全局缓存实例
llm_cache = LLMCache(max_size=500, default_ttl=7200)  # 缓存500条，默认过期时间2小时

# 全局请求队列实例
llm_queue = LLMRequestQueue(max_concurrent=3, timeout=60)  # 最大3个并发请求，超时60秒


async def cached_llm_invoke(llm: Any, messages: List[BaseMessage], temperature: float = 0.7, max_retries: int = 2) -> Any:
    """
    带缓存和错误处理的LLM调用函数
    
    Args:
        llm: LLM实例
        messages: 输入消息列表
        temperature: 温度参数
        max_retries: 最大重试次数
        
    Returns:
        LLM响应结果（可能来自缓存）
    
    Raises:
        Exception: 如果所有重试都失败，抛出最终异常
    """
    # 获取模型名称
    model = getattr(llm, "model", "unknown")
    
    # 尝试从缓存获取
    cached_result = await llm_cache.get(messages, model, temperature)
    if cached_result:
        logger.debug(f"LLM调用缓存命中，模型: {model}")
        return cached_result
    
    # 缓存未命中，尝试调用LLM
    retry_count = 0
    last_exception = None
    
    while retry_count <= max_retries:
        try:
            logger.debug(f"LLM调用缓存未命中，尝试调用，模型: {model}, 重试次数: {retry_count}")
            
            # 通过请求队列调用LLM
            result = await llm_queue.add_request(llm, messages, temperature)
            
            # 将结果存入缓存
            await llm_cache.set(messages, model, temperature, result)
            
            logger.debug(f"LLM调用成功，模型: {model}")
            return result
            
        except asyncio.TimeoutError as e:
            last_exception = e
            retry_count += 1
            logger.warning(f"LLM调用超时，将进行第{retry_count}次重试: {str(e)}")
            
        except (ConnectionError, BrokenPipeError, OSError) as e:
            last_exception = e
            retry_count += 1
            logger.warning(f"LLM调用连接错误，将进行第{retry_count}次重试: {str(e)}")
            
        except Exception as e:
            # 其他异常，不重试
            logger.error(f"LLM调用发生非重试异常: {str(e)}")
            raise
        
        # 重试前等待一段时间，避免立即重试
        if retry_count <= max_retries:
            await asyncio.sleep(1)  # 等待1秒后重试
    
    # 所有重试都失败
    logger.error(f"LLM调用在{max_retries+1}次尝试后失败: {str(last_exception)}")
    raise last_exception
