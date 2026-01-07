import pytest
import asyncio
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, AIMessage
from app.utils.cache import LLMCache, LLMRequestQueue, cached_llm_invoke


class MockLLM:
    def __init__(self, model="mock-model"):
        self.model = model
        self.call_count = 0
    
    async def ainvoke(self, messages, temperature=0.7):
        self.call_count += 1
        return AIMessage(content=f"Response to: {messages[-1].content}")


@pytest.mark.asyncio
async def test_llm_cache_basic():
    """测试LLMCache的基本功能"""
    cache = LLMCache(max_size=10, default_ttl=30)
    
    # 测试设置和获取
    messages = [HumanMessage(content="Hello")]
    model = "test-model"
    temperature = 0.7
    value = "Test response"
    
    await cache.set(messages, model, temperature, value)
    retrieved = await cache.get(messages, model, temperature)
    assert retrieved == value
    
    # 测试不同参数的缓存键不同
    retrieved_diff_temp = await cache.get(messages, model, temperature=0.5)
    assert retrieved_diff_temp is None
    
    retrieved_diff_model = await cache.get(messages, "other-model", temperature)
    assert retrieved_diff_model is None


@pytest.mark.asyncio
async def test_llm_cache_expiration():
    """测试LLMCache的过期功能"""
    cache = LLMCache(max_size=10, default_ttl=1)
    
    messages = [HumanMessage(content="Hello")]
    model = "test-model"
    temperature = 0.7
    value = "Test response"
    
    await cache.set(messages, model, temperature, value)
    
    # 立即获取应该存在
    retrieved = await cache.get(messages, model, temperature)
    assert retrieved == value
    
    # 等待过期
    await asyncio.sleep(1.1)
    
    # 过期后应该不存在
    retrieved = await cache.get(messages, model, temperature)
    assert retrieved is None


@pytest.mark.asyncio
async def test_llm_cache_size_limit():
    """测试LLMCache的大小限制"""
    cache = LLMCache(max_size=2, default_ttl=30)
    
    model = "test-model"
    temperature = 0.7
    
    # 添加3个缓存项
    for i in range(3):
        messages = [HumanMessage(content=f"Message {i}")]
        await cache.set(messages, model, temperature, f"Response {i}")
    
    # 获取统计信息
    stats = await cache.get_stats()
    assert stats["total_entries"] == 2


@pytest.mark.asyncio
async def test_llm_request_queue():
    """测试LLMRequestQueue的基本功能"""
    queue = LLMRequestQueue(max_concurrent=2, timeout=5)
    mock_llm = MockLLM()
    
    messages = [HumanMessage(content="Hello")]
    
    # 测试基本请求
    result = await queue.add_request(mock_llm, messages)
    assert result.content == "Response to: Hello"
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_llm_request_queue_concurrency():
    """测试LLMRequestQueue的并发控制"""
    queue = LLMRequestQueue(max_concurrent=2, timeout=5)
    mock_llm = MockLLM()
    
    async def make_request(i):
        messages = [HumanMessage(content=f"Hello {i}")]
        return await queue.add_request(mock_llm, messages)
    
    # 同时发起3个请求，应该限制为2个并发
    tasks = [make_request(i) for i in range(3)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 3
    assert mock_llm.call_count == 3
    for i, result in enumerate(results):
        assert result.content == f"Response to: Hello {i}"


@pytest.mark.asyncio
async def test_cached_llm_invoke():
    """测试cached_llm_invoke函数"""
    mock_llm = MockLLM()
    
    messages = [HumanMessage(content="Hello")]
    
    # 第一次调用，应该调用LLM
    result1 = await cached_llm_invoke(mock_llm, messages)
    assert mock_llm.call_count == 1
    
    # 第二次调用，应该使用缓存
    result2 = await cached_llm_invoke(mock_llm, messages)
    assert mock_llm.call_count == 1
    
    # 确保结果一致
    assert result1.content == result2.content


@pytest.mark.asyncio
async def test_cached_llm_invoke_retry():
    """测试cached_llm_invoke的重试功能"""
    class FailingLLM:
        def __init__(self):
            self.model = "test-model"
            self.call_count = 0
        
        async def ainvoke(self, messages, temperature=0.7):
            self.call_count += 1
            if self.call_count <= 2:
                raise ConnectionError("Test connection error")
            return AIMessage(content="Success after retry")
    
    mock_llm = FailingLLM()
    messages = [HumanMessage(content="Hello")]
    
    # 应该重试2次后成功
    result = await cached_llm_invoke(mock_llm, messages, max_retries=2)
    assert mock_llm.call_count == 3
    assert result.content == "Success after retry"


if __name__ == "__main__":
    asyncio.run(test_llm_cache_basic())
    print("test_llm_cache_basic passed")
    
    asyncio.run(test_llm_cache_expiration())
    print("test_llm_cache_expiration passed")
    
    asyncio.run(test_llm_cache_size_limit())
    print("test_llm_cache_size_limit passed")
    
    asyncio.run(test_llm_request_queue())
    print("test_llm_request_queue passed")
    
    asyncio.run(test_llm_request_queue_concurrency())
    print("test_llm_request_queue_concurrency passed")
    
    asyncio.run(test_cached_llm_invoke())
    print("test_cached_llm_invoke passed")
    
    asyncio.run(test_cached_llm_invoke_retry())
    print("test_cached_llm_invoke_retry passed")
