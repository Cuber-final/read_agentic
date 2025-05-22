"""
意图分类器 (IntentClassifier)

此模块负责分析用户的输入，判断其主要意图。例如：
- RAG_SpecificText: 用户针对选中的书中特定文本提问
- RAG_BookGeneral: 用户提出关于当前书籍的普遍性问题，但未选中特定文本
- ChitChat: 用户进行闲聊，或提出与书本内容无关的问题
- ToolRequest_Definition/Summary: 用户明确要求进行生词释义或章节摘要
- FollowUp: 用户的问题是基于上一轮对话的追问
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Dict, Any

from autochain.models.chat_openai import ChatOpenAI

from bookrag.user_input_handler import StructuredUserInput


class IntentType(Enum):
    """用户意图的枚举类型"""
    RAG_SPECIFIC_TEXT = auto()      # 针对选中文本的问题
    RAG_BOOK_GENERAL = auto()       # 关于书籍的一般性问题
    CHIT_CHAT = auto()              # 闲聊或与书本无关的问题
    TOOL_REQUEST_DEFINITION = auto() # 请求定义或解释术语
    TOOL_REQUEST_SUMMARY = auto()    # 请求总结或概述
    FOLLOW_UP = auto()              # 基于前一轮对话的追问


class ConfidenceLevel(Enum):
    """意图分类的置信度级别"""
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


@dataclass
class IntentClassificationResult:
    """意图分类的结果"""
    intent: IntentType
    confidence: ConfidenceLevel
    # 可选的附加信息，如分类的依据、备选意图等
    additional_info: Optional[Dict[str, Any]] = None


class IntentClassifier:
    """
    意图分类器类
    
    使用LLM模型对用户输入进行意图分类，判断用户查询的主要意图。
    """
    
    def __init__(self, llm=None):
        """
        初始化意图分类器
        
        Args:
            llm: 用于分类的语言模型，默认为None，将使用OpenAI的轻量级模型
        """
        # 如果没有传入LLM，则使用默认的轻量级模型
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
    def _prepare_prompt(self, user_input: StructuredUserInput) -> str:
        """
        准备用于意图分类的Prompt
        
        Args:
            user_input: 结构化的用户输入
            
        Returns:
            str: 格式化的Prompt文本
        """
        # 提取选中文本信息
        selected_text_str = "无" if not user_input.selected_text else user_input.selected_text.text
        
        # 准备对话历史摘要
        conversation_summary = ""
        if user_input.conversation_history and len(user_input.conversation_history) > 0:
            # 简单示例：仅取最近一轮对话作为摘要
            # 实际应用中可能需要更复杂的处理
            last_turn = user_input.conversation_history[-1]
            conversation_summary = f"上一轮: {last_turn.role}说: {last_turn.content}"
        
        # 构建Prompt
        prompt = f"""
你是一个专业的文本意图分类机器人。根据用户提供的输入信息，你需要将其分类到以下意图之一： 
RAG_SPECIFIC_TEXT, RAG_BOOK_GENERAL, CHIT_CHAT, TOOL_REQUEST_DEFINITION, TOOL_REQUEST_SUMMARY, FOLLOW_UP。

意图定义与示例：
- RAG_SPECIFIC_TEXT: 用户针对选中的书中特定文本提问。
  特征：通常会附带引用的文本。
  例子：用户问："关于'他认为经济基础决定上层建筑'这句话，作者具体是怎么解释的？" 同时选中了"他认为经济基础决定上层建筑"。
- RAG_BOOK_GENERAL: 用户提出关于当前书籍的普遍性问题，但未选中特定文本。
  特征：问题宽泛，关于书籍主题、作者观点等。
  例子："这本书的核心论点是什么？"
- CHIT_CHAT: 用户闲聊或提出与书本无关的问题。
  例子："你叫什么名字？"
- TOOL_REQUEST_DEFINITION: 用户明确要求定义术语。
  例子："'范式转移'是什么意思？"
- TOOL_REQUEST_SUMMARY: 用户明确要求总结内容。
  例子："能帮我总结一下第三章吗？"
- FOLLOW_UP: 用户基于上一轮对话追问。
  特征：通常包含指代词，需要结合对话历史。
  例子：（上一轮回答了X）用户："那它和Y有什么不同？"

输入信息：
- 用户查询: "{user_input.query}"
- 选中文本: "{selected_text_str}"
- 相关对话历史摘要: "{conversation_summary}"
- 当前阅读书籍: "{user_input.book_metadata.book_title}", 章节: "{user_input.book_metadata.current_chapter_title or '未知'}"

请输出最可能的意图类别和置信度 (高/中/低)。
输出格式： {{"intent": "意图类别", "confidence": "置信度"}}
        """
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> IntentClassificationResult:
        """
        解析LLM的响应，提取意图和置信度
        
        Args:
            response: LLM返回的原始响应文本
            
        Returns:
            IntentClassificationResult: 解析后的意图分类结果
        """
        # 实际实现中需要更健壮的解析逻辑
        # 这里使用简单的示例解析
        try:
            # 假设LLM返回的是格式化的JSON字符串
            import json
            import re
            
            # 使用正则表达式找到JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result_dict = json.loads(json_str)
                
                # 提取意图
                intent_str = result_dict.get("intent", "").strip().upper()
                intent = getattr(IntentType, intent_str) if hasattr(IntentType, intent_str) else IntentType.CHIT_CHAT
                
                # 提取置信度
                confidence_str = result_dict.get("confidence", "").strip()
                if confidence_str == "高":
                    confidence = ConfidenceLevel.HIGH
                elif confidence_str == "中":
                    confidence = ConfidenceLevel.MEDIUM
                else:
                    confidence = ConfidenceLevel.LOW
                
                return IntentClassificationResult(
                    intent=intent,
                    confidence=confidence,
                    additional_info={"raw_response": response}
                )
            
        except Exception as e:
            # 解析失败，返回默认结果
            print(f"意图解析出错: {e}")
        
        # 默认返回闲聊意图，低置信度
        return IntentClassificationResult(
            intent=IntentType.CHIT_CHAT,
            confidence=ConfidenceLevel.LOW,
            additional_info={"error": "解析失败", "raw_response": response}
        )
    
    def classify(self, user_input: StructuredUserInput) -> IntentClassificationResult:
        """
        对用户输入进行意图分类
        
        Args:
            user_input: 结构化的用户输入
            
        Returns:
            IntentClassificationResult: 意图分类结果
        """
        # 准备Prompt
        prompt = self._prepare_prompt(user_input)
        
        # 调用LLM进行分类
        try:
            # 使用ChatOpenAI的接口调用模型
            response = self.llm.invoke(prompt)
            
            # 解析LLM的响应
            result = self._parse_llm_response(response.content)
            return result
            
        except Exception as e:
            # 处理调用LLM时可能出现的错误
            print(f"调用LLM出错: {e}")
            
            # 返回默认结果
            return IntentClassificationResult(
                intent=IntentType.CHIT_CHAT,
                confidence=ConfidenceLevel.LOW,
                additional_info={"error": str(e)}
            ) 