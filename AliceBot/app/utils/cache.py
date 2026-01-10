import asyncio
import hashlib
import json
import logging
import msgpack
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Deque
from collections import deque, OrderedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

# 配置日志
logger = logging.getLogger(__name__)

# 获取AliceBot根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cache_dir = os.path.join(BASE_DIR, "cache")
os.makedirs(cache_dir, exist_ok=True)


class LLMCache:
    """
    LLM调用缓存系统
    用于缓存LLM调用的请求和响应，减少重复调用，提高性能
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 7200, persist_file: Optional[str] = None, persist_interval: int = 300):
        """
        初始化缓存系统
        
        Args:
            max_size: 缓存的最大条目数
            default_ttl: 默认的缓存过期时间（秒）
            persist_file: 缓存持久化文件路径，None表示不持久化
            persist_interval: 自动持久化间隔（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self.lock = asyncio.Lock()
        
        # 统计信息
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        
        # 持久化相关配置
        self.persist_file = persist_file
        self.persist_interval = persist_interval
        self.last_persist_time = datetime.now()
        
        # 缓存策略配置
        self.ttl_overrides: Dict[str, int] = {
            # 为不同类型的请求设置不同的TTL
            "simple_query": 3600,  # 简单查询缓存1小时
            "complex_query": 1800,  # 复杂查询缓存30分钟
            "creative_query": 600,  # 创造性查询缓存10分钟
            "group_chat": 1800,  # 群聊消息缓存30分钟
            "private_chat": 3600,  # 私聊消息缓存1小时
            "mention_response": 1200,  # 被@回复缓存20分钟
            "generic_response": 300,  # 通用响应缓存5分钟
            # 新增组件特定的缓存策略
            "context_filter": 3600,  # ContextFilter结果缓存1小时（确定性决策）
            "vision_router": 1800,  # VisionRouter结果缓存30分钟（视觉判断）
            "psychology_analysis": 1200,  # Psychology分析结果缓存20分钟（情绪分析）
            "memory_extraction": 3600  # 记忆提取结果缓存1小时（用户信息提取）
        }
        
        # 如果配置了持久化文件，尝试从磁盘加载
        if self.persist_file:
            try:
                self._load_from_disk()
                logger.info(f"成功从磁盘加载缓存: {self.persist_file}")
            except Exception as e:
                logger.error(f"从磁盘加载缓存失败: {str(e)}")
    
    def _load_from_disk(self):
        """
        从磁盘加载缓存数据
        """
        import os
        
        if not self.persist_file or not os.path.exists(self.persist_file):
            return
        
        with open(self.persist_file, 'rb') as f:
            data = msgpack.unpackb(f.read(), raw=False)
        
        # 重建缓存，只保留未过期的条目
        now = datetime.now()
        for key, (value, expire_timestamp) in data.items():
            expire_time = datetime.fromtimestamp(expire_timestamp)
            if now < expire_time:
                # 处理可能的序列化后的数据格式
                # 如果是字典且包含type字段，可能是我们序列化的AIMessage
                if isinstance(value, dict) and value.get("type") == "AIMessage":
                    # 尝试从字典还原AIMessage对象
                    from langchain_core.messages import AIMessage
                    try:
                        value = AIMessage(**value)
                    except Exception as e:
                        logger.debug(f"无法还原AIMessage对象: {str(e)}")
                        # 如果还原失败，直接使用字典
                
                self.cache[key] = (value, expire_time)
    
    def _generate_key(self, messages: List[BaseMessage], model: str, temperature: float, query_type: str = "default") -> str:
        """
        根据输入消息生成唯一的缓存键
        
        Args:
            messages: LLM调用的输入消息列表
            model: 使用的模型名称
            temperature: 模型的温度参数
            query_type: 查询类型，用于区分不同的缓存策略
            
        Returns:
            唯一的缓存键字符串
        """
        # 简化消息转换，只保留必要信息
        message_data = []
        for msg in messages:
            msg_data = {
                "type": msg.__class__.__name__,  # 消息类型
                "content": msg.content,  # 消息内容
                "additional_kwargs": msg.additional_kwargs,  # 附加参数
            }
            message_data.append(msg_data)
        
        # 添加模型、温度参数和查询类型
        cache_key_data = {
            "messages": message_data,
            "model": model,
            "temperature": temperature,
            "query_type": query_type
        }
        
        # 使用msgpack替代json序列化，提高性能
        cache_key_bytes = msgpack.packb(cache_key_data, use_bin_type=True)
        return hashlib.sha256(cache_key_bytes).hexdigest()
    
    async def get(self, messages: List[BaseMessage], model: str, temperature: float, query_type: str = "default") -> Optional[Any]:
        """
        从缓存中获取LLM调用结果
        
        Args:
            messages: LLM调用的输入消息列表
            model: 使用的模型名称
            temperature: 模型的温度参数
            query_type: 查询类型，用于区分不同的缓存策略
            
        Returns:
            缓存的LLM响应，如果没有缓存或已过期则返回None
        """
        cache_key = self._generate_key(messages, model, temperature, query_type)
        
        async with self.lock:
            self.total_requests += 1
            
            if cache_key in self.cache:
                value, expire_time = self.cache.pop(cache_key)  # 移除条目
                if datetime.now() < expire_time:
                    self.cache[cache_key] = (value, expire_time)  # 重新添加到末尾，实现LRU
                    self.hits += 1
                    return value
            
            self.misses += 1
            return None
    
    async def set(self, messages: List[BaseMessage], model: str, temperature: float, value: Any, ttl: Optional[int] = None, query_type: str = "default") -> None:
        """
        将LLM调用结果存入缓存
        
        Args:
            messages: LLM调用的输入消息列表
            model: 使用的模型名称
            temperature: 模型的温度参数
            value: LLM的响应结果
            ttl: 缓存过期时间（秒），如果为None则使用默认值或根据query_type选择
            query_type: 查询类型，用于区分不同的缓存策略
        """
        # 选择合适的TTL
        if ttl is None:
            ttl = self.ttl_overrides.get(query_type, self.default_ttl)
            
            # 对于创造性内容，即使没有指定query_type，也使用较短的TTL
            if temperature > 0.8:
                ttl = min(ttl, 600)  # 最多缓存10分钟
            elif temperature > 0.5:
                ttl = min(ttl, 1800)  # 最多缓存30分钟
        
        cache_key = self._generate_key(messages, model, temperature, query_type)
        expire_time = datetime.now() + timedelta(seconds=ttl)
        
        async with self.lock:
            # 如果条目已存在，先移除（会自动移到末尾）
            if cache_key in self.cache:
                del self.cache[cache_key]
            
            # 检查缓存大小，如果超过最大值则清理最旧的条目
            if len(self.cache) >= self.max_size:
                # OrderedDict的第一个条目就是最旧的
                self.cache.popitem(last=False)
            
            self.cache[cache_key] = (value, expire_time)
        
        # 检查是否需要自动持久化
        if self.persist_file:
            time_since_last_persist = (datetime.now() - self.last_persist_time).total_seconds()
            if time_since_last_persist >= self.persist_interval:
                self._save_to_disk()
    
    def _save_to_disk(self):
        """
        将缓存数据保存到磁盘
        """
        if not self.persist_file:
            return
        
        try:
            # 准备要保存的数据
            data_to_save = {}
            for key, (value, expire_time) in self.cache.items():
                # 转换响应结果为可序列化格式
                serializable_value = value
                
                # 如果是AIMessage或其他非序列化对象，转换为字典
                if hasattr(serializable_value, "dict"):
                    try:
                        serializable_value = serializable_value.dict()
                    except Exception as e:
                        logger.debug(f"无法通过dict()方法序列化对象: {str(e)}")
                        # 尝试提取主要内容
                        if hasattr(serializable_value, "content"):
                            serializable_value = {
                                "content": serializable_value.content,
                                "type": serializable_value.__class__.__name__
                            }
                elif hasattr(serializable_value, "content"):
                    # 处理可能的消息对象
                    serializable_value = {
                        "content": serializable_value.content,
                        "type": serializable_value.__class__.__name__
                    }
                
                # 将datetime转换为时间戳以便存储
                data_to_save[key] = (serializable_value, expire_time.timestamp())
            
            # 创建父目录（如果不存在）
            import os
            os.makedirs(os.path.dirname(self.persist_file), exist_ok=True)
            
            # 写入文件
            with open(self.persist_file, 'wb') as f:
                f.write(msgpack.packb(data_to_save, use_bin_type=True))
            
            self.last_persist_time = datetime.now()
            logger.debug(f"缓存已保存到磁盘: {self.persist_file}")
        except Exception as e:
            logger.error(f"保存缓存到磁盘失败: {str(e)}")
    
    async def clear(self) -> None:
        """
        清空缓存
        """
        async with self.lock:
            self.cache.clear()
        
        # 清空后立即持久化
        if self.persist_file:
            self._save_to_disk()
    
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
        
        # 计算命中率
        hit_rate = self.hits / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "total_entries": total,
            "expired_entries": expired,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": self.total_requests,
            "hit_rate": round(hit_rate, 4)
        }


class LLMRequestQueue:
    """
    LLM请求队列系统
    用于管理LLM调用请求，控制并发数，防止请求堆积和超时
    """
    
    def __init__(self, max_concurrent: int = 15, timeout: int = 60):
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
        self.pending_requests: Dict[str, List[asyncio.Future]] = {}  # 存储正在处理的请求
        
        # 请求合并优化
        self.merge_window = 0.5  # 请求合并窗口（秒）
        self.last_request_time: Dict[str, float] = {}  # 记录每个请求键的最后请求时间
        
        # 统计信息
        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0
        self.merged_requests = 0  # 通过请求合并节省的请求数
        self.total_processing_time = 0.0
        self.request_times = []  # 最近请求的响应时间列表
    
    def _generate_request_key(self, llm: Any, messages: List[BaseMessage], temperature: float) -> str:
        """
        生成请求的唯一标识
        
        Args:
            llm: LLM实例
            messages: 输入消息列表
            temperature: 温度参数
            
        Returns:
            唯一的请求标识字符串
        """
        # 获取模型名称
        model = getattr(llm, "model", "unknown")
        
        # 使用与LLMCache相同的键生成逻辑
        llm_cache_instance = LLMCache()  # 创建临时实例来复用键生成逻辑
        return llm_cache_instance._generate_key(messages, model, temperature)
    
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
        # 生成请求的唯一标识
        request_key = self._generate_request_key(llm, messages, temperature)
        
        # 检查是否有相同的请求正在处理
        async with self.lock:
            if request_key in self.pending_requests:
                # 创建新的future并加入等待列表
                future = asyncio.Future()
                self.pending_requests[request_key].append(future)
                self.total_requests += 1
                self.merged_requests += 1  # 这是一个被合并的请求
                
                # 等待结果
                try:
                    return await future
                except Exception as e:
                    # 移除future以防止内存泄漏
                    self.pending_requests[request_key].remove(future)
                    raise
            
            # 创建新的请求列表
            self.pending_requests[request_key] = []
            self.total_requests += 1
        
        # 处理请求
        async with self.semaphore:
            start_time = datetime.now()
            try:
                # 使用asyncio.wait_for设置请求超时
                result = await asyncio.wait_for(
                    llm.ainvoke(messages),
                    timeout=self.timeout
                )
                
                # 计算处理时间
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # 更新统计信息
                self.success_requests += 1
                self.total_processing_time += processing_time
                
                # 记录最近的请求响应时间（只保留最近100个）
                self.request_times.append(processing_time)
                if len(self.request_times) > 100:
                    self.request_times.pop(0)
                
                # 通知所有等待的请求
                async with self.lock:
                    if request_key in self.pending_requests:
                        # 更新合并请求统计
                        self.merged_requests += len(self.pending_requests[request_key])
                        
                        for future in self.pending_requests[request_key]:
                            if not future.done():
                                future.set_result(result)
                        # 清理请求
                        del self.pending_requests[request_key]
                
                return result
            except Exception as e:
                # 计算处理时间
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # 更新统计信息
                self.failed_requests += 1
                
                # 通知所有等待的请求发生错误
                async with self.lock:
                    if request_key in self.pending_requests:
                        # 更新合并请求统计
                        self.merged_requests += len(self.pending_requests[request_key])
                        
                        for future in self.pending_requests[request_key]:
                            if not future.done():
                                future.set_exception(e)
                        # 清理请求
                        del self.pending_requests[request_key]
                
                logger.error(f"LLM请求执行出错: {str(e)}")
                raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        获取队列统计信息
        
        Returns:
            包含队列统计信息的字典
        """
        async with self.lock:
            # 计算平均处理时间
            avg_processing_time = 0.0
            if self.success_requests > 0:
                avg_processing_time = self.total_processing_time / self.success_requests
            
            # 计算请求成功率
            success_rate = 0.0
            if self.total_requests > 0:
                success_rate = self.success_requests / self.total_requests
            
            # 计算请求合并率
            merge_rate = 0.0
            if self.total_requests > 0:
                merge_rate = self.merged_requests / self.total_requests
            
            # 计算最近请求的平均、最大、最小响应时间
            recent_avg_time = 0.0
            recent_max_time = 0.0
            recent_min_time = 0.0
            if self.request_times:
                recent_avg_time = sum(self.request_times) / len(self.request_times)
                recent_max_time = max(self.request_times)
                recent_min_time = min(self.request_times)
            
            return {
                "queue_length": len(self.queue),
                "current_concurrent": self.current_concurrent,
                "max_concurrent": self.max_concurrent,
                "timeout": self.timeout,
                "total_requests": self.total_requests,
                "success_requests": self.success_requests,
                "failed_requests": self.failed_requests,
                "merged_requests": self.merged_requests,
                "success_rate": round(success_rate, 4),
                "merge_rate": round(merge_rate, 4),
                "avg_processing_time": round(avg_processing_time, 4),
                "total_processing_time": round(self.total_processing_time, 4),
                "recent_avg_time": round(recent_avg_time, 4),
                "recent_max_time": round(recent_max_time, 4),
                "recent_min_time": round(recent_min_time, 4),
                "in_flight_requests": len(self.pending_requests)
            }


# 全局缓存实例
llm_cache = LLMCache(
    max_size=1000,  # 增加缓存大小到1000条
    default_ttl=7200,  # 缓存默认过期时间2小时
    persist_file=os.path.join(cache_dir, "llm_cache.msgpack"),  # 缓存持久化文件路径
    persist_interval=300  # 每5分钟自动持久化一次，减少IO操作
)

# 用户信息缓存
user_cache = LLMCache(
    max_size=2000,  # 增加用户信息缓存到2000个
    default_ttl=86400,  # 缓存1天
    persist_file=os.path.join(cache_dir, "user_cache.msgpack")
)

# 上下文缓存
context_cache = LLMCache(
    max_size=2000,  # 增加上下文缓存到2000个
    default_ttl=1800,  # 缓存30分钟
    persist_file=os.path.join(cache_dir, "context_cache.msgpack")
)

# 工具调用结果缓存
tool_cache = LLMCache(
    max_size=4000,  # 增加工具调用缓存到4000个
    default_ttl=3600,  # 缓存1小时
    persist_file=os.path.join(cache_dir, "tool_cache.msgpack")
)

# 全局请求队列实例
llm_queue = LLMRequestQueue(max_concurrent=15, timeout=60)  # 增加并发数到15，提高并行处理能力


async def cached_llm_invoke(llm: Any, messages: List[BaseMessage], temperature: float = 0.7, max_retries: int = 2, query_type: str = "default", conversation_type: str = "private") -> Any:
    """
    带缓存和错误处理的LLM调用函数
    
    Args:
        llm: LLM实例
        messages: 输入消息列表
        temperature: 温度参数
        max_retries: 最大重试次数
        query_type: 查询类型，用于区分不同的缓存策略
        conversation_type: 对话类型，用于优化缓存策略（group/private）
        
    Returns:
        LLM响应结果（可能来自缓存）
    
    Raises:
        Exception: 如果所有重试都失败，抛出最终异常
    """
    # 自动判断查询类型
    auto_query_type = query_type
    if auto_query_type == "default":
        # 检查是否为群聊被@的情况
        last_msg = messages[-1] if messages else None
        if isinstance(last_msg, HumanMessage):
            content = str(last_msg.content)
            # 检查是否包含@标记
            if "@" in content or "CQ:at" in content:
                auto_query_type = "mention_response"
            # 简单判断消息复杂度
            elif len(content) < 50:
                auto_query_type = "simple_query"
            elif len(content) > 200:
                auto_query_type = "complex_query"
            else:
                auto_query_type = "generic_response"
        
        # 根据对话类型调整查询类型
        if conversation_type == "group":
            auto_query_type = f"group_{auto_query_type}"
        else:
            auto_query_type = f"private_{auto_query_type}"
    
    # 获取模型名称
    model = getattr(llm, "model", "unknown")
    
    # 尝试从缓存获取
    cached_result = await llm_cache.get(messages, model, temperature, auto_query_type)
    if cached_result:
        logger.debug(f"LLM调用缓存命中，模型: {model}, 查询类型: {auto_query_type}")
        return cached_result
    
    # 缓存未命中，尝试调用LLM
    retry_count = 0
    last_exception = None
    
    while retry_count <= max_retries:
        try:
            logger.debug(f"LLM调用缓存未命中，尝试调用，模型: {model}, 查询类型: {auto_query_type}, 重试次数: {retry_count}")
            
            # 通过请求队列调用LLM
            result = await llm_queue.add_request(llm, messages, temperature)
            
            # 将结果存入缓存
            await llm_cache.set(messages, model, temperature, result, query_type=auto_query_type)
            
            logger.debug(f"LLM调用成功，模型: {model}, 查询类型: {auto_query_type}")
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
            # 使用指数退避算法
            wait_time = 2 ** retry_count  # 1, 2, 4秒
            await asyncio.sleep(wait_time)
    
    # 所有重试都失败
    logger.error(f"LLM调用在{max_retries+1}次尝试后失败: {str(last_exception)}")
    raise last_exception


async def cached_user_info_get(user_qq: str) -> Optional[Any]:
    """
    从缓存获取用户信息
    
    Args:
        user_qq: 用户QQ号
        
    Returns:
        用户信息，如果缓存未命中则返回None
    """
    result = await user_cache.get([HumanMessage(content=f"user_info:{user_qq}")], "user_info", 0.0)
    return result


async def cached_user_info_set(user_qq: str, user_info: Any, ttl: Optional[int] = None) -> None:
    """
    将用户信息存入缓存
    
    Args:
        user_qq: 用户QQ号
        user_info: 用户信息
        ttl: 缓存过期时间（秒），如果为None则使用默认值
    """
    await user_cache.set([HumanMessage(content=f"user_info:{user_qq}")], "user_info", 0.0, user_info, ttl)


async def cached_context_get(context_key: str) -> Optional[Any]:
    """
    从缓存获取上下文信息
    
    Args:
        context_key: 上下文键
        
    Returns:
        上下文信息，如果缓存未命中则返回None
    """
    result = await context_cache.get([HumanMessage(content=f"context:{context_key}")], "context", 0.0)
    return result


async def cached_context_set(context_key: str, context_data: Any, ttl: Optional[int] = None) -> None:
    """
    将上下文信息存入缓存
    
    Args:
        context_key: 上下文键
        context_data: 上下文数据
        ttl: 缓存过期时间（秒），如果为None则使用默认值
    """
    await context_cache.set([HumanMessage(content=f"context:{context_key}")], "context", 0.0, context_data, ttl)


async def cached_tool_result_get(tool_name: str, tool_args: dict) -> Optional[Any]:
    """
    从缓存获取工具调用结果
    
    Args:
        tool_name: 工具名称
        tool_args: 工具调用参数
        
    Returns:
        工具调用结果，如果缓存未命中则返回None
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    
    cache_key_messages = [
        SystemMessage(content=f"tool:{tool_name}"),
        HumanMessage(content=json.dumps(tool_args, sort_keys=True))
    ]
    
    result = await tool_cache.get(cache_key_messages, "tool", 0.0)
    return result


async def cached_tool_result_set(tool_name: str, tool_args: dict, result: Any, ttl: Optional[int] = None) -> None:
    """
    将工具调用结果存入缓存
    
    Args:
        tool_name: 工具名称
        tool_args: 工具调用参数
        result: 工具调用结果
        ttl: 缓存过期时间（秒），如果为None则使用默认值
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    
    cache_key_messages = [
        SystemMessage(content=f"tool:{tool_name}"),
        HumanMessage(content=json.dumps(tool_args, sort_keys=True))
    ]
    
    await tool_cache.set(cache_key_messages, "tool", 0.0, result, ttl)


# 嵌入向量缓存
embedding_cache = LLMCache(
    max_size=5000, 
    default_ttl=86400,  # 缓存5000个嵌入向量，默认过期时间1天
    persist_file=os.path.join(cache_dir, "embedding_cache.msgpack")
)


async def cached_embedding_get(text: str, model: str) -> Optional[List[float]]:
    """
    从缓存获取文本的嵌入向量
    
    Args:
        text: 要获取嵌入向量的文本
        model: 使用的嵌入模型名称
        
    Returns:
        嵌入向量，如果缓存未命中则返回None
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    
    cache_key_messages = [
        SystemMessage(content=f"embedding:{model}"),
        HumanMessage(content=text)
    ]
    
    result = await embedding_cache.get(cache_key_messages, "embedding", 0.0)
    return result


async def cached_embedding_set(text: str, model: str, embedding: List[float], ttl: Optional[int] = None) -> None:
    """
    将文本的嵌入向量存入缓存
    
    Args:
        text: 要缓存嵌入向量的文本
        model: 使用的嵌入模型名称
        embedding: 嵌入向量
        ttl: 缓存过期时间（秒），如果为None则使用默认值
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    
    cache_key_messages = [
        SystemMessage(content=f"embedding:{model}"),
        HumanMessage(content=text)
    ]
    
    await embedding_cache.set(cache_key_messages, "embedding", 0.0, embedding, ttl)
