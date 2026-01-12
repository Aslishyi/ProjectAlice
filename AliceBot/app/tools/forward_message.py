# === 新文件: app/tools/forward_message.py ===

import json
import logging
from typing import Optional, Dict, Any
from sqlalchemy.exc import SQLAlchemyError

# 导入基础工具类
from app.tools.base_tool import BaseTool, ToolParam

# 导入数据库配置
from app.core.database import SessionLocal, ForwardMessageModel

# 配置日志
logger = logging.getLogger(__name__)


class ForwardMessageTool(BaseTool):
    """
    获取完整的转发消息内容工具。当需要查看被省略的转发消息详情时使用此工具。
    """
    name = "get_forward_message"
    description = "获取完整的转发消息内容。当需要查看被省略的转发消息详情时使用此工具。"
    available_for_llm = True
    
    parameters = [
        ToolParam(
            name="forward_id",
            param_type="string",
            description="转发消息的ID，格式为数字字符串",
            required=True
        )
    ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行获取转发消息的操作
        
        Args:
            forward_id: 转发消息的ID
            
        Returns:
            Dict[str, Any]: 工具执行结果
        """
        forward_id = kwargs.get("forward_id")
        
        try:
            with SessionLocal() as db:
                # 查询数据库
                forward_message = db.query(ForwardMessageModel).filter(ForwardMessageModel.forward_id == forward_id).first()
                
                if forward_message:
                    # 更新最后访问时间
                    db.commit()
                    
                    logger.info(f"🔍 [Forward Tool] Retrieved forward message: {forward_id}")
                    
                    return {
                        "success": True,
                        "result": {
                            "forward_id": forward_id,
                            "content": forward_message.full_content,
                            "summary": forward_message.summary,
                            "message_count": forward_message.message_count,
                            "image_count": forward_message.image_count
                        },
                        "error": None
                    }
                else:
                    logger.warning(f"🔍 [Forward Tool] Forward message not found: {forward_id}")
                    return {
                        "success": False,
                        "result": None,
                        "error": f"未找到ID为 {forward_id} 的转发消息"
                    }
        
        except SQLAlchemyError as e:
            logger.error(f"❌ [Forward Tool] Database error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"数据库查询错误: {str(e)}"
            }
        except Exception as e:
            logger.error(f"❌ [Forward Tool] Unexpected error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"发生意外错误: {str(e)}"
            }


class ListForwardMessagesTool(BaseTool):
    """
    列出最近存储的转发消息工具。
    """
    name = "list_forward_messages"
    description = "列出最近存储的转发消息。"
    available_for_llm = True
    
    parameters = [
        ToolParam(
            name="limit",
            param_type="integer",
            description="返回的最大数量，默认10",
            required=False,
            enum_values=None
        )
    ]
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行列出转发消息的操作
        
        Args:
            limit: 返回的最大数量
            
        Returns:
            Dict[str, Any]: 工具执行结果
        """
        limit = kwargs.get("limit", 10)
        
        try:
            with SessionLocal() as db:
                # 查询最近的转发消息
                forward_messages = db.query(ForwardMessageModel).order_by(ForwardMessageModel.created_at.desc()).limit(limit).all()
                
                result_list = []
                for forward in forward_messages:
                    result_list.append({
                        "forward_id": forward.forward_id,
                        "summary": forward.summary,
                        "message_count": forward.message_count,
                        "image_count": forward.image_count,
                        "created_at": forward.created_at.isoformat(),
                        "accessed_at": forward.accessed_at.isoformat()
                    })
                
                logger.info(f"📋 [Forward Tool] Listed {len(result_list)} forward messages")
                
                return {
                    "success": True,
                    "result": result_list,
                    "error": None
                }
        
        except SQLAlchemyError as e:
            logger.error(f"❌ [Forward Tool] Database error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"数据库查询错误: {str(e)}"
            }
        except Exception as e:
            logger.error(f"❌ [Forward Tool] Unexpected error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"发生意外错误: {str(e)}"
            }