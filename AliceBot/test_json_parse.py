#!/usr/bin/env python3
"""测试优化后的robust_json_parse函数"""

import sys
import os
import json

# 确保能导入 app.*
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 导入robust_json_parse函数
from app.graph.nodes.unified_agent import robust_json_parse

def test_json_parse():
    """测试各种情况下的JSON解析"""
    print("=== 测试robust_json_parse函数 ===")
    
    # 测试1: 标准JSON格式
    test1 = '{"monologue": "这是一个标准的JSON响应", "action": "reply", "args": "", "response": "你好，世界！"}'
    result1 = robust_json_parse(test1)
    print(f"\n测试1 - 标准JSON格式:")
    print(f"输入: {test1}")
    print(f"输出: {result1}")
    print(f"结果: {'✓' if result1 else '✗'}")
    
    # 测试2: 带有Markdown格式的JSON
    test2 = '```json\n{"monologue": "这是一个带有Markdown格式的JSON响应", "action": "reply", "args": "", "response": "你好，世界！"}\n```'
    result2 = robust_json_parse(test2)
    print(f"\n测试2 - 带有Markdown格式的JSON:")
    print(f"输入: {test2}")
    print(f"输出: {result2}")
    print(f"结果: {'✓' if result2 else '✗'}")
    
    # 测试3: 纯文本响应
    test3 = "嗯，东欧文学确实有独特的魅力。他们的小说常带着浓厚的历史感和文化底蕴，让人读起来有种穿越时空的感觉。"
    result3 = robust_json_parse(test3)
    print(f"\n测试3 - 纯文本响应:")
    print(f"输入: {test3}")
    print(f"输出: {result3}")
    print(f"结果: {'✓' if result3 else '✗'}")
    print(f"是否包含所有必要字段: {'✓' if all(key in result3 for key in ['monologue', 'action', 'args', 'response']) else '✗'}")
    print(f"action是否为'reply': {'✓' if result3.get('action') == 'reply' else '✗'}")
    
    # 测试4: 带有system hint的响应
    test4 = "[system hint: This is a system hint]\n{\"monologue\": \"这是一个带有system hint的JSON响应\", \"action\": \"reply\", \"args\": \"\", \"response\": \"你好，世界！\"}"
    result4 = robust_json_parse(test4)
    print(f"\n测试4 - 带有system hint的响应:")
    print(f"输入: {test4}")
    print(f"输出: {result4}")
    print(f"结果: {'✓' if result4 else '✗'}")
    
    # 测试5: 不完整的JSON格式
    test5 = '{"monologue": "这是一个不完整的JSON响应", "action": "reply", "args": "", "response": "你好，世界！"'  # 缺少最后一个}}
    result5 = robust_json_parse(test5)
    print(f"\n测试5 - 不完整的JSON格式:")
    print(f"输入: {test5}")
    print(f"输出: {result5}")
    print(f"结果: {'✓' if result5 else '✗'}")
    print(f"是否作为纯文本处理: {'✓' if result5.get('response') == test5 else '✗'}")
    
    # 测试6: 空字符串
    test6 = ""
    result6 = robust_json_parse(test6)
    print(f"\n测试6 - 空字符串:")
    print(f"输入: ''")
    print(f"输出: {result6}")
    print(f"结果: {'✓' if result6 is None else '✗'}")
    
    print("\n=== 所有测试完成 ===")

if __name__ == "__main__":
    test_json_parse()
