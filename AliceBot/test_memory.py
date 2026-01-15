#!/usr/bin/env python3
"""测试记忆模块功能"""

import asyncio
import sys
import os
from datetime import datetime

# 确保能导入 app.*
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入memory模块
from app.memory import memory_manager, relation_db, vector_db

async def test_combined_memory():
    """测试组合记忆管理器"""
    print("\n=== 测试组合记忆管理器 ===")
    
    # 测试更新记忆
    await memory_manager.update_memory("你好，我叫张三", "你好，张三，很高兴认识你", "test_user", "张三")
    print("✓ 更新记忆成功")
    
    # 测试获取相关记忆
    relevant_memory = await memory_manager.get_relevant_memory("张三是谁", "test_user")
    print(f"✓ 获取相关记忆成功: {str(relevant_memory)[:100]}...")
    
    # 测试智能检索
    smart_result = await memory_manager.smart_retrieve("张三是谁", "", "张三", "test_user")
    print(f"✓ 智能检索成功: {smart_result}")

async def test_vector_store():
    """测试向量存储"""
    print("\n=== 测试向量存储 ===")
    
    # 测试添加文本
    await vector_db.add_texts(["测试文本1", "测试文本2"], [
        {"user_id": "test_user", "created_at": "2023-01-01 12:00:00"},
        {"user_id": "test_user", "created_at": "2023-01-02 12:00:00"}
    ])
    print("✓ 添加文本成功")
    
    # 测试搜索
    results = await vector_db.search("测试文本", k=2)
    print(f"✓ 搜索成功: {len(results)} 个结果")

async def test_relation_db():
    """测试关系数据库"""
    print("\n=== 测试关系数据库 ===")
    
    # 测试获取用户资料
    user_profile = await relation_db.get_user_profile("test_user")
    print(f"✓ 获取用户资料成功: {user_profile.name}")
    
    # 测试更新亲密度
    relation_db.update_intimacy("test_user", 5)
    print("✓ 更新亲密度成功")
    
    # 测试更新关系维度（好感度、熟悉度等）
    await relation_db.update_relationship_dimensions("test_user", {"intimacy": 2, "familiarity": 1})
    print("✓ 更新关系维度成功")
    
    # 测试添加记忆点
    relation_db.add_memory_point("test_user", "测试分类", "测试内容", 1.0)
    print("✓ 添加记忆点成功")
    
    # 测试添加表达习惯
    relation_db.add_expression_habit("test_user", "经常使用表情符号")
    print("✓ 添加表达习惯成功")

async def test_memory_module_api():
    """测试记忆模块API统一接口"""
    print("\n=== 测试记忆模块API统一接口 ===")
    
    # 测试导入是否正常
    from app.memory import (
        MemoryManager, CombinedMemoryManager,
        RelationDB, GlobalRelationDB,
        VectorDB, VectorMemory,
        SmartRetriever, MemoryRetrievalTool,
        memory_manager, combined_memory,
        relation_db,
        vector_db,
        smart_memory_retriever,
        get_smart_memory_retriever,
        initialize_smart_memory_retrieval
    )
    
    print("✓ API接口导入成功")
    print(f"✓ MemoryManager 是 CombinedMemoryManager 的别名: {MemoryManager is CombinedMemoryManager}")
    print(f"✓ RelationDB 是 GlobalRelationDB 的别名: {RelationDB is GlobalRelationDB}")
    print(f"✓ VectorDB 是 VectorMemory 的别名: {VectorDB is VectorMemory}")
    print(f"✓ SmartRetriever 是 MemoryRetrievalTool 的别名: {SmartRetriever is MemoryRetrievalTool}")
    print(f"✓ memory_manager 是 combined_memory 的别名: {memory_manager is combined_memory}")
    
    # 测试智能检索工具初始化
    result = initialize_smart_memory_retrieval()
    print(f"✓ 智能检索工具初始化: {'成功' if result else '失败'}")
    
    # 测试获取智能检索工具
    retriever = get_smart_memory_retriever()
    print(f"✓ 获取智能检索工具: {'成功' if retriever else '失败'}")

async def main():
    """主测试函数"""
    print(f"开始测试记忆模块 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 手动启动定时清理任务
        vector_db.start_cleanup_task()
        relation_db.start_cleanup_task()
        print("✓ 定时清理任务已启动")
        
        await test_combined_memory()
        await test_vector_store()
        await test_relation_db()
        await test_memory_module_api()
        
        print("\n=== 所有测试完成 ===")
        print("✅ 记忆模块功能正常！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
