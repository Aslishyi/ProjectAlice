import pytest
import asyncio
import io
import base64
from PIL import Image, ImageDraw
from unittest.mock import MagicMock, patch
from app.plugins.emoji_plugin.emoji_service import EmojiService, get_emoji_service, initialize_emoji_service
from app.plugins.emoji_plugin.emoji_manager import EmojiInfo


class MockEmojiManager:
    def __init__(self):
        self.emojis = []
        self.download_count = 0
    
    def download_image_to_base64(self, url):
        self.download_count += 1
        # 创建一个简单的表情包图片
        img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), '😊', font=None, fill=(0, 0, 0, 255))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def add_emoji(self, base64_data, description, emotions, tags, category):
        emoji_info = EmojiInfo(
            emoji_hash=f"test_hash_{len(self.emojis)}",
            base64_data=base64_data,
            file_path=f"test_path_{len(self.emojis)}.png",
            description=description,
            emotions=emotions,
            tags=tags,
            category=category
        )
        self.emojis.append(emoji_info)
        return True, "Success", emoji_info
    
    def get_emojis_by_emotion(self, emotion):
        matching = []
        for emoji in self.emojis:
            if emotion in emoji.emotions:
                matching.append(emoji)
        return matching if matching else [self.emojis[0]] if self.emojis else []
    
    def get_emoji_for_text(self, text, count=1):
        return [self.emojis[0]] if self.emojis else []
    
    def get_random_emoji(self, count=1):
        return self.emojis[:count]
    
    def get_emoji(self, emoji_hash):
        for emoji in self.emojis:
            if emoji.emoji_hash == emoji_hash:
                return emoji
        return None


@patch('app.plugins.emoji_plugin.emoji_service.get_emoji_manager')
@patch('app.plugins.emoji_plugin.emoji_service._analyze_emoji_with_llm')
def test_emoji_service_initialization(mock_analyze, mock_get_emoji_manager):
    """测试表情包服务的初始化"""
    mock_manager = MockEmojiManager()
    mock_get_emoji_manager.return_value = mock_manager
    
    # 测试初始化
    emoji_service = EmojiService()
    assert emoji_service.emoji_manager is not None
    mock_get_emoji_manager.assert_called_once()


@patch('app.plugins.emoji_plugin.emoji_service.get_emoji_manager')
@patch('app.plugins.emoji_plugin.emoji_service._analyze_emoji_with_llm')
def test_is_emoji(mock_analyze, mock_get_emoji_manager):
    """测试表情包识别功能"""
    mock_manager = MockEmojiManager()
    mock_get_emoji_manager.return_value = mock_manager
    
    emoji_service = EmojiService()
    
    # 创建测试图片
    # 1. 表情包图片（小尺寸，透明背景）
    emoji_img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(emoji_img)
    draw.text((50, 50), '😊', font=None, fill=(0, 0, 0, 255))
    
    # 2. 普通照片（大尺寸，不透明背景）
    photo_img = Image.new('RGB', (1200, 800), (255, 0, 0))
    
    # 3. 小图标
    icon_img = Image.new('RGB', (30, 30), (0, 255, 0))
    
    # 测试结果
    assert emoji_service.is_emoji(emoji_img, 50) == True
    assert emoji_service.is_emoji(photo_img, 500) == False
    assert emoji_service.is_emoji(icon_img, 5) == False


@patch('app.plugins.emoji_plugin.emoji_service.get_emoji_manager')
@pytest.mark.asyncio
async def test_analyze_emoji(mock_get_emoji_manager):
    """测试表情包情绪分析功能"""
    mock_manager = MockEmojiManager()
    mock_get_emoji_manager.return_value = mock_manager
    
    emoji_service = EmojiService()
    
    # 创建测试图片
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), '😊', font=None, fill=(0, 0, 0, 255))
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 模拟LLM分析结果
    expected_result = {
        "emotions": ["开心", "愉快", "可爱"],
        "description": "一个笑脸表情包",
        "category": "表情符号"
    }
    
    with patch('app.plugins.emoji_plugin.emoji_service._analyze_emoji_with_llm', return_value=expected_result):
        result = await emoji_service.analyze_emoji(base64_data)
        assert result == expected_result


@patch('app.plugins.emoji_plugin.emoji_service.get_emoji_manager')
@pytest.mark.asyncio
async def test_get_emoji_for_context(mock_get_emoji_manager):
    """测试基于上下文的表情包回复功能"""
    mock_manager = MockEmojiManager()
    mock_get_emoji_manager.return_value = mock_manager
    
    # 添加测试表情包
    mock_manager.add_emoji(
        base64_data="test_data",
        description="开心表情包",
        emotions=["开心"],
        tags=["test"],
        category="表情符号"
    )
    
    mock_manager.add_emoji(
        base64_data="test_data",
        description="悲伤表情包",
        emotions=["悲伤"],
        tags=["test"],
        category="表情符号"
    )
    
    emoji_service = EmojiService()
    
    # 测试上下文情绪提取
    context = {
        "last_message": "今天很开心！",
        "message_history": [
            {"content": "你好啊！"},
            {"content": "我今天真的很开心！"}
        ]
    }
    
    result = emoji_service.get_emoji_for_context(context, count=1)
    assert len(result) == 1
    assert "开心" in result[0].emotions


@patch('app.plugins.emoji_plugin.emoji_service.get_emoji_manager')
@pytest.mark.asyncio
async def test_process_emoji(mock_get_emoji_manager):
    """测试完整的表情包处理流程"""
    mock_manager = MockEmojiManager()
    mock_get_emoji_manager.return_value = mock_manager
    
    emoji_service = EmojiService()
    
    # 模拟LLM分析结果
    expected_analysis = {
        "emotions": ["开心", "愉快"],
        "description": "一个笑脸表情包",
        "category": "表情符号"
    }
    
    with patch('app.plugins.emoji_plugin.emoji_service._analyze_emoji_with_llm', return_value=expected_analysis):
        result = await emoji_service.process_emoji(
            "http://test.com/emoji.png",
            "test_user",
            "Test User"
        )
        
        assert result["success"] == True
        assert result["emotions"] == ["开心", "愉快"]
        assert result["description"] == "一个笑脸表情包"
        assert result["category"] == "表情符号"


@patch('app.plugins.emoji_plugin.emoji_service.get_emoji_manager')
@patch('app.plugins.emoji_plugin.emoji_service._analyze_emoji_with_llm')
def test_image_emoji_boundary(mock_analyze, mock_get_emoji_manager):
    """测试图片和表情包的边界处理"""
    mock_manager = MockEmojiManager()
    mock_get_emoji_manager.return_value = mock_manager
    
    emoji_service = EmojiService()
    
    # 创建测试图片
    # 1. 超大图片（应该被拒绝）
    large_img = Image.new('RGBA', (3000, 3000), (255, 255, 255, 0))
    draw = ImageDraw.Draw(large_img)
    draw.text((50, 50), '😊', font=None, fill=(0, 0, 0, 255))
    
    buffer = io.BytesIO()
    large_img.save(buffer, format='PNG')
    large_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    large_file_size = len(buffer.getvalue()) / 1024  # KB
    
    # 2. 普通表情包
    emoji_img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(emoji_img)
    draw.text((50, 50), '😊', font=None, fill=(0, 0, 0, 255))
    
    buffer = io.BytesIO()
    emoji_img.save(buffer, format='PNG')
    emoji_file_size = len(buffer.getvalue()) / 1024  # KB
    
    # 测试边界判断
    assert emoji_service.is_emoji(emoji_img, emoji_file_size) == True
    
    # 注意：由于_is_emoji使用的是_classify_image，我们需要单独测试process_emoji中的边界检查


@patch('app.plugins.emoji_plugin.emoji_service.get_emoji_manager')
@patch('app.plugins.emoji_plugin.emoji_service._classify_image')
@pytest.mark.asyncio
async def test_process_emoji_boundary_checks(mock_classify, mock_get_emoji_manager):
    """测试process_emoji中的边界检查"""
    mock_manager = MockEmojiManager()
    mock_get_emoji_manager.return_value = mock_manager
    
    emoji_service = EmojiService()
    
    # 1. 测试普通照片被拒绝
    mock_classify.return_value = "photo"
    result = await emoji_service.process_emoji("http://test.com/photo.jpg")
    assert result["success"] == False
    assert "不是表情包 (分类: photo)" in result["message"]
    
    # 2. 测试小图标被拒绝
    mock_classify.return_value = "icon"
    result = await emoji_service.process_emoji("http://test.com/icon.png")
    assert result["success"] == False
    assert "不是表情包 (分类: icon)" in result["message"]
    
    # 3. 测试表情包被接受
    mock_classify.return_value = "sticker"
    with patch('app.plugins.emoji_plugin.emoji_service._analyze_emoji_with_llm', return_value={
        "emotions": ["开心"],
        "description": "测试表情包",
        "category": "表情符号"
    }):
        result = await emoji_service.process_emoji("http://test.com/emoji.png")
        assert result["success"] == True


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
