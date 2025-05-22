"""
上下文管理器 (ContextManager)

负责收集、处理和维护与当前用户查询相关的上下文信息，包括：
- 选中文本上下文：获取用户选中的文本及其紧邻的上下文（如前后段落、所在小节标题）
- 书籍级上下文：获取当前章节标题、书名等宏观信息
- 对话上下文：从对话历史中提取与当前查询相关的信息
- 上下文结构化：将收集到的上下文信息组织成结构化的格式
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from bookrag.user_input_handler import StructuredUserInput
from bookrag.intent_classifier import IntentClassificationResult, IntentType


@dataclass
class SelectedTextContext:
    """选中文本及其上下文"""
    selected_text: str
    # 选中文本前后的上下文（如果有）
    previous_paragraphs: List[str] = field(default_factory=list)
    following_paragraphs: List[str] = field(default_factory=list)
    section_title: Optional[str] = None
    chapter_title: Optional[str] = None


@dataclass
class BookLevelContext:
    """书籍层面的上下文信息"""
    book_title: str
    book_id: str
    current_chapter_title: Optional[str] = None
    current_chapter_id: Optional[str] = None
    # 可以扩展更多书籍元数据，如作者、出版信息等


@dataclass
class ConversationalContext:
    """对话上下文信息"""
    # 与当前查询相关的历史对话轮次
    relevant_turns: List[Dict[str, str]] = field(default_factory=list)
    # 对话主题或焦点
    topic_focus: Optional[str] = None
    # 上一轮问答（如果有）
    last_question: Optional[str] = None
    last_answer: Optional[str] = None


@dataclass
class StructuredContext:
    """结构化的上下文信息"""
    selected_text_context: Optional[SelectedTextContext] = None
    book_level_context: Optional[BookLevelContext] = None
    conversational_context: Optional[ConversationalContext] = None
    # 可以添加其他类型的上下文信息


class ContextManager:
    """
    上下文管理器类
    
    负责从各种来源收集、处理和组织与当前用户查询相关的上下文信息。
    """
    
    def __init__(self, book_content_provider=None):
        """
        初始化上下文管理器
        
        Args:
            book_content_provider: 提供书籍内容的对象，用于获取选中文本周围的段落
        """
        # 书籍内容提供者，用于获取选中文本周围的段落
        # 在实际实现中，这可能是一个访问数据库或文件系统的组件
        self.book_content_provider = book_content_provider
    
    def collect_selected_text_context(self, user_input: StructuredUserInput) -> Optional[SelectedTextContext]:
        """
        收集选中文本及其上下文
        
        Args:
            user_input: 结构化的用户输入
            
        Returns:
            Optional[SelectedTextContext]: 选中文本上下文，如果没有选中文本则为None
        """
        if not user_input.selected_text:
            return None
        
        selected_text = user_input.selected_text.text
        chapter_id = user_input.selected_text.chapter_id
        
        # 通过book_content_provider获取选中文本周围的段落
        # 这里是一个简化的示例，实际实现需要根据真实的数据结构
        previous_paragraphs = []
        following_paragraphs = []
        section_title = None
        
        if self.book_content_provider:
            try:
                # 假设book_content_provider有一个方法可以获取前后文
                context_data = self.book_content_provider.get_surrounding_context(
                    book_id=user_input.book_metadata.book_id,
                    chapter_id=chapter_id,
                    selected_text=selected_text,
                    paragraphs_before=2,  # 获取前2段
                    paragraphs_after=2    # 获取后2段
                )
                
                previous_paragraphs = context_data.get("previous_paragraphs", [])
                following_paragraphs = context_data.get("following_paragraphs", [])
                section_title = context_data.get("section_title")
            except Exception as e:
                print(f"获取选中文本上下文出错: {e}")
        
        # 构建并返回选中文本上下文
        return SelectedTextContext(
            selected_text=selected_text,
            previous_paragraphs=previous_paragraphs,
            following_paragraphs=following_paragraphs,
            section_title=section_title,
            chapter_title=user_input.book_metadata.current_chapter_title
        )
    
    def collect_book_level_context(self, user_input: StructuredUserInput) -> BookLevelContext:
        """
        收集书籍层面的上下文信息
        
        Args:
            user_input: 结构化的用户输入
            
        Returns:
            BookLevelContext: 书籍层面的上下文信息
        """
        # 直接从用户输入的book_metadata中提取信息
        return BookLevelContext(
            book_title=user_input.book_metadata.book_title,
            book_id=user_input.book_metadata.book_id,
            current_chapter_title=user_input.book_metadata.current_chapter_title,
            current_chapter_id=user_input.book_metadata.current_chapter_id
        )
    
    def collect_conversational_context(self, user_input: StructuredUserInput, intent_result: IntentClassificationResult) -> ConversationalContext:
        """
        从对话历史中提取与当前查询相关的信息
        
        Args:
            user_input: 结构化的用户输入
            intent_result: 意图分类结果
            
        Returns:
            ConversationalContext: 对话上下文信息
        """
        conversational_context = ConversationalContext()
        
        # 如果没有对话历史，直接返回空的上下文
        if not user_input.conversation_history or len(user_input.conversation_history) == 0:
            return conversational_context
        
        # 提取上一轮问答（如果有）
        if len(user_input.conversation_history) >= 2:
            for i in range(len(user_input.conversation_history) - 1, 0, -1):
                if user_input.conversation_history[i].role == "assistant" and i > 0 and user_input.conversation_history[i-1].role == "user":
                    conversational_context.last_question = user_input.conversation_history[i-1].content
                    conversational_context.last_answer = user_input.conversation_history[i].content
                    break
        
        # 根据意图决定如何处理对话历史
        if intent_result.intent == IntentType.FOLLOW_UP:
            # 如果是追问，则需要更多的对话历史
            # 这里简单地取最近3轮对话（如果有）
            max_turns = min(6, len(user_input.conversation_history))
            for i in range(len(user_input.conversation_history) - max_turns, len(user_input.conversation_history)):
                turn = user_input.conversation_history[i]
                conversational_context.relevant_turns.append({
                    "role": turn.role,
                    "content": turn.content
                })
        else:
            # 对于其他意图，可能只需要最近的1-2轮对话
            if len(user_input.conversation_history) >= 2:
                for i in range(len(user_input.conversation_history) - 2, len(user_input.conversation_history)):
                    turn = user_input.conversation_history[i]
                    conversational_context.relevant_turns.append({
                        "role": turn.role,
                        "content": turn.content
                    })
        
        # 提取对话主题或焦点（这可能需要更复杂的分析）
        # 这里简单地使用最后一个用户查询作为话题
        for turn in reversed(user_input.conversation_history):
            if turn.role == "user":
                conversational_context.topic_focus = turn.content
                break
        
        return conversational_context
    
    def get_structured_context(self, user_input: StructuredUserInput, intent_result: IntentClassificationResult) -> StructuredContext:
        """
        获取结构化的上下文信息
        
        Args:
            user_input: 结构化的用户输入
            intent_result: 意图分类结果
            
        Returns:
            StructuredContext: 结构化的上下文信息
        """
        # 收集各类上下文信息
        selected_text_context = self.collect_selected_text_context(user_input)
        book_level_context = self.collect_book_level_context(user_input)
        conversational_context = self.collect_conversational_context(user_input, intent_result)
        
        # 构建并返回结构化上下文
        return StructuredContext(
            selected_text_context=selected_text_context,
            book_level_context=book_level_context,
            conversational_context=conversational_context
        )
    
    def get_context_as_dict(self, structured_context: StructuredContext) -> Dict[str, Any]:
        """
        将结构化上下文转换为字典形式，便于传递给其他组件
        
        Args:
            structured_context: 结构化的上下文信息
            
        Returns:
            Dict[str, Any]: 字典形式的上下文信息
        """
        context_dict = {}
        
        # 处理选中文本上下文
        if structured_context.selected_text_context:
            context_dict["selected_text"] = {
                "text": structured_context.selected_text_context.selected_text,
                "previous_paragraphs": structured_context.selected_text_context.previous_paragraphs,
                "following_paragraphs": structured_context.selected_text_context.following_paragraphs,
                "section_title": structured_context.selected_text_context.section_title,
                "chapter_title": structured_context.selected_text_context.chapter_title
            }
        
        # 处理书籍层面上下文
        if structured_context.book_level_context:
            context_dict["book"] = {
                "title": structured_context.book_level_context.book_title,
                "id": structured_context.book_level_context.book_id,
                "current_chapter_title": structured_context.book_level_context.current_chapter_title,
                "current_chapter_id": structured_context.book_level_context.current_chapter_id
            }
        
        # 处理对话上下文
        if structured_context.conversational_context:
            context_dict["conversation"] = {
                "relevant_turns": structured_context.conversational_context.relevant_turns,
                "topic_focus": structured_context.conversational_context.topic_focus,
                "last_question": structured_context.conversational_context.last_question,
                "last_answer": structured_context.conversational_context.last_answer
            }
        
        return context_dict 