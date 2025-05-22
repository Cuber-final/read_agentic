"""
回复合成器模块 (ResponseSynthesizer)

在获取到经过ReflectionAgent确认的高质量RAG结果后，
此模块负责整合原始用户查询、相关上下文以及RAG召回的知识片段，
调用LLM生成最终的、流畅自然的回答。
"""

from typing import Dict, List, Any, Optional

from autochain.models.chat_openai import ChatOpenAI

from bookrag.intent_classifier import IntentClassificationResult, IntentType


class ResponseSynthesizer:
    """
    回复合成器类
    
    负责基于RAG检索结果和用户查询生成最终回复。
    """
    
    def __init__(self, llm=None):
        """
        初始化回复合成器
        
        Args:
            llm: 用于生成回复的语言模型，默认为None，将使用默认模型
        """
        # 使用功能较强的模型生成回复
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0.7)
    
    def _prepare_context_info(self, context: Dict[str, Any]) -> str:
        """
        准备上下文信息的摘要
        
        Args:
            context: 上下文信息字典
            
        Returns:
            str: 上下文摘要文本
        """
        context_parts = []
        
        # 书籍信息
        if "book" in context:
            book = context["book"]
            context_parts.append(f"书籍: {book.get('title', '')}")
            if book.get('current_chapter_title'):
                context_parts.append(f"当前章节: {book.get('current_chapter_title', '')}")
        
        # 选中文本信息
        if "selected_text" in context and context["selected_text"].get("text"):
            selected_text = context["selected_text"]
            context_parts.append(f"用户选中的文本: \"{selected_text.get('text', '')}\"")
            
            # 如果有小节标题，也包括进来
            if selected_text.get("section_title"):
                context_parts.append(f"所在小节: {selected_text.get('section_title', '')}")
        
        return "\n".join(context_parts)
    
    def _prepare_rag_content(self, rag_result: Dict[str, Any]) -> str:
        """
        准备RAG检索结果的内容
        
        Args:
            rag_result: RAG检索结果字典
            
        Returns:
            str: 格式化后的RAG内容
        """
        combined_chunks = rag_result.get("combined_chunks", [])
        
        if not combined_chunks:
            return "未找到相关内容。"
        
        # 按相关性分数排序
        chunks_with_score = []
        for chunk in combined_chunks:
            score = chunk.get("metadata", {}).get("score", 0)
            chunks_with_score.append((chunk, score))
        
        # 降序排序
        chunks_with_score.sort(key=lambda x: x[1], reverse=True)
        
        # 格式化内容
        content_parts = []
        for i, (chunk, score) in enumerate(chunks_with_score):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            chapter_title = metadata.get("chapter_title", "未知章节")
            
            content_parts.append(f"[内容片段 {i+1}] 来自 {chapter_title}:")
            content_parts.append(text)
            content_parts.append("")  # 空行分隔
        
        return "\n".join(content_parts)
    
    def generate_response(
        self, 
        user_query: str, 
        intent_result: IntentClassificationResult, 
        rag_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        生成最终回复
        
        Args:
            user_query: 用户的原始查询
            intent_result: 意图分类结果
            rag_result: RAG检索结果
            context: 上下文信息
            
        Returns:
            str: 生成的回复文本
        """
        # 准备上下文信息
        context_info = self._prepare_context_info(context)
        
        # 准备RAG内容
        rag_content = self._prepare_rag_content(rag_result)
        
        # 根据不同意图调整指令
        instruction = ""
        if intent_result.intent == IntentType.RAG_SPECIFIC_TEXT:
            instruction = "用户询问了选中文本的相关内容，请重点关注与选中文本相关的信息。"
        elif intent_result.intent == IntentType.RAG_BOOK_GENERAL:
            instruction = "用户询问了书籍的一般性问题，请提供一个综合性的回答。"
        elif intent_result.intent == IntentType.TOOL_REQUEST_DEFINITION:
            instruction = "用户请求解释或定义某个术语，请提供清晰的定义和解释。"
        elif intent_result.intent == IntentType.TOOL_REQUEST_SUMMARY:
            instruction = "用户请求总结内容，请提供简明扼要的摘要。"
        elif intent_result.intent == IntentType.FOLLOW_UP:
            instruction = "这是一个后续问题，请确保回答与之前的对话上下文保持连贯。"
        
        # 构建Prompt
        prompt = f"""
你是一个专业的书籍问答助手。请基于提供的信息，为用户生成一个准确、有帮助的回答。

用户查询: "{user_query}"

上下文信息:
{context_info}

从书籍中检索到的相关内容:
{rag_content}

特别说明:
{instruction}

在回答时，请遵循以下原则:
1. 直接回答用户的问题，不要重复用户的问题或提及你是基于什么资料回答的
2. 如果检索结果包含足够信息，基于这些信息提供详细的解答
3. 如果检索结果信息不足或不相关，坦率地告知用户你无法提供完整答案
4. 回答应当流畅自然，语气友好专业
5. 如果有必要，可以引用书中的原文，但要用引号标明
6. 避免生成虚假信息，只基于提供的检索结果回答

请直接开始你的回答，无需任何前缀词如"好的"、"这是我的回答"等。
        """
        
        try:
            # 调用LLM生成回复
            response = self.llm.invoke(prompt)
            
            # 返回生成的回复文本
            return response.content
            
        except Exception as e:
            print(f"生成回复出错: {e}")
            
            # 返回一个出错提示
            return f"抱歉，在生成回答时遇到了问题。请稍后再试或重新表述您的问题。" 