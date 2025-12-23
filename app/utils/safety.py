import re
from typing import Tuple


class SafetyFilter:
    def __init__(self):
        # 1. 关键词黑名单 (硬过滤)
        # 这里仅作示例，实际项目中应加载更完善的词库
        self.blacklisted_patterns = [
            r"(sex|porn|nude|erotic|xxx)",
            r"(kill yourself|suicide method)",
            # ... 更多正则表达式
        ]

    def check_input(self, text: str) -> Tuple[bool, str]:
        """
        检查输入是否安全。
        Returns: (is_safe: bool, reason: str)
        """
        text_lower = text.lower()

        # 1. 关键词正则匹配
        for pattern in self.blacklisted_patterns:
            if re.search(pattern, text_lower):
                return False, "Contains restricted keywords."

        # 2. (可选) 调用 OpenAI Moderation API
        # from langchain_openai import OpenAIModerationChain
        # moderation = OpenAIModerationChain()
        # output = moderation.invoke(text)
        # if output['output_flagged']: return False, "Flagged by Moderation API"

        return True, ""

    def get_refusal_response(self) -> str:
        """返回标准的拒绝话术"""
        return (
            "I cannot fulfill that request. My programming prevents me from engaging "
            "in explicit, harmful, or unsafe content. Let's discuss something else."
        )


safety_filter = SafetyFilter()
