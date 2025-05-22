"""
用户输入处理器 (UserInputHandler)

作为系统的入口，负责接收和处理来自前端的原始用户输入，包括：
- 用户在聊天框输入的文本
- 在阅读器中选中的句子/段落
- 当前阅读书本的元数据（如书名、章节ID）
- 完整的对话历史记录

并将这些原始数据结构化，以便后续模块处理。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class BookMetadata:
    """书籍元数据类，包含书籍相关的基本信息"""
    book_id: str
    book_title: str
    current_chapter_id: Optional[str] = None
    current_chapter_title: Optional[str] = None
    # 可以根据需要扩展更多元数据字段


@dataclass
class SelectedText:
    """用户在阅读器中选中的文本信息"""
    text: str
    # 选中文本在文档中的位置信息
    chapter_id: Optional[str] = None
    paragraph_id: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None


@dataclass
class ConversationTurn:
    """对话中的一轮交互"""
    role: str  # "user" 或 "assistant"
    content: str
    timestamp: Optional[float] = None


@dataclass
class StructuredUserInput:
    """结构化的用户输入，包含所有相关上下文信息"""
    query: str  # 用户当前的查询文本
    book_metadata: BookMetadata
    selected_text: Optional[SelectedText] = None
    conversation_history: List[ConversationTurn] = None


class UserInputHandler:
    """
    用户输入处理器类
    
    负责接收前端传来的原始用户输入，并将其转换为结构化格式，
    以便后续模块（如意图分类器、上下文管理器等）处理。
    """
    
    def __init__(self):
        """初始化用户输入处理器"""
        pass
    
    def process_raw_input(self, 
                          query: str, 
                          book_metadata: Dict[str, Any],
                          selected_text: Optional[Dict[str, Any]] = None,
                          conversation_history: Optional[List[Dict[str, Any]]] = None) -> StructuredUserInput:
        """
        处理前端传来的原始用户输入
        
        Args:
            query: 用户的原始查询文本
            book_metadata: 书籍元数据字典
            selected_text: 用户选中文本的信息字典
            conversation_history: 对话历史列表
            
        Returns:
            StructuredUserInput: 结构化后的用户输入对象
        """
        # 处理书籍元数据
        processed_book_metadata = BookMetadata(
            book_id=book_metadata.get("book_id", ""),
            book_title=book_metadata.get("book_title", ""),
            current_chapter_id=book_metadata.get("current_chapter_id"),
            current_chapter_title=book_metadata.get("current_chapter_title")
        )
        
        # 处理选中文本
        processed_selected_text = None
        if selected_text and selected_text.get("text"):
            processed_selected_text = SelectedText(
                text=selected_text.get("text", ""),
                chapter_id=selected_text.get("chapter_id"),
                paragraph_id=selected_text.get("paragraph_id"),
                start_index=selected_text.get("start_index"),
                end_index=selected_text.get("end_index")
            )
        
        # 处理对话历史
        processed_conversation_history = []
        if conversation_history:
            for turn in conversation_history:
                processed_conversation_history.append(
                    ConversationTurn(
                        role=turn.get("role", ""),
                        content=turn.get("content", ""),
                        timestamp=turn.get("timestamp")
                    )
                )
        
        # 创建并返回结构化用户输入
        return StructuredUserInput(
            query=query,
            book_metadata=processed_book_metadata,
            selected_text=processed_selected_text,
            conversation_history=processed_conversation_history
        ) 