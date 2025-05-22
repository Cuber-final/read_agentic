"""
反思Agent模块 (ReflectionAgent)

这是实现Reflection范式的核心模块。它接收RAGInterface返回的检索结果，
并对其进行评估，判断召回内容是否满足用户需求，以及是否需要对查询策略进行调整。

主要功能：
- 相关性评估：判断召回的chunks与用户原始查询及上下文的匹配程度
- 充分性评估：判断召回的chunks是否足以回答用户的问题
- 迭代改进：如果评估结果不理想，生成反馈建议，要求重新规划查询策略
"""

from typing import Dict, List, Any, Optional, Tuple

from autochain.models.chat_openai import ChatOpenAI

from bookrag.rag_interface import RAGResult
from bookrag.intent_classifier import IntentClassificationResult


class ReflectionAgent:
    """
    反思Agent类
    
    负责评估RAG检索结果的质量，判断是否满足用户需求，并提供改进建议。
    """
    
    def __init__(self, llm=None):
        """
        初始化反思Agent
        
        Args:
            llm: 用于评估的语言模型，默认为None，将使用默认模型
        """
        # 使用较轻量级的模型进行评估
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # 最大迭代次数，防止无限循环
        self.max_iterations = 3
    
    def evaluate_rag_result(
        self, 
        user_query: str, 
        intent_result: IntentClassificationResult,
        rag_result: Dict[str, Any],
        context: Dict[str, Any],
        iteration: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """
        评估RAG检索结果的质量
        
        Args:
            user_query: 用户的原始查询
            intent_result: 意图分类结果
            rag_result: RAG检索结果
            context: 上下文信息
            iteration: 当前迭代次数
            
        Returns:
            Tuple[bool, Optional[str]]: (是否满足需求, 改进建议)
        """
        # 如果已达到最大迭代次数，直接返回满足
        if iteration >= self.max_iterations:
            return True, None
        
        # 提取RAG结果中的文档片段
        combined_chunks = rag_result.get("combined_chunks", [])
        
        # 如果没有检索到任何结果，直接返回不满足
        if not combined_chunks:
            return False, "未检索到任何相关内容，建议扩大检索范围或使用更通用的关键词。"
        
        # 从上下文中提取选中文本（如果有）
        selected_text = ""
        if "selected_text" in context and context["selected_text"]:
            selected_text = context["selected_text"].get("text", "")
        
        # 构建评估Prompt
        chunks_text = []
        for i, chunk in enumerate(combined_chunks[:5]):  # 限制为前5个结果
            text = chunk.get("text", "")
            score = chunk.get("metadata", {}).get("score", "未知")
            source = chunk.get("source_query", "")
            purpose = chunk.get("query_purpose", "")
            
            chunks_text.append(f"结果 #{i+1} [相关性分数: {score}]:")
            chunks_text.append(f"来源查询: '{source}' (目的: {purpose})")
            chunks_text.append(f"内容: {text}")
            chunks_text.append("---")
        
        chunks_text_str = "\n".join(chunks_text)
        
        prompt = f"""
你是一个专业的RAG检索结果评估专家。请对以下检索结果进行评估，判断它们是否满足用户的查询需求。

用户原始查询: "{user_query}"
用户意图: {intent_result.intent.name}
选中文本: "{selected_text}"

检索结果:
{chunks_text_str}

请从以下几个方面评估检索结果:
1. 相关性: 结果与用户查询的相关程度如何？是否包含与查询主题相关的信息？
2. 完整性: 结果是否提供了足够完整的信息来回答用户的问题？
3. 准确性: 结果是否准确地针对用户关心的点？
4. 上下文匹配: 如果用户选中了特定文本，结果是否与该文本及其上下文相关？

最后，请给出以下结论:
1. 满足度评分 (1-10分)
2. 是否满足用户需求 (是/否)
3. 如果不满足，请提供具体的改进建议，以便我们能重新规划查询策略

请以JSON格式输出你的评估结果:
```json
{{
  "relevance_score": 评分,
  "completeness_score": 评分,
  "accuracy_score": 评分,
  "context_match_score": 评分,
  "overall_satisfaction": 总评分,
  "is_satisfactory": true/false,
  "improvement_suggestions": "如果不满足，在这里提供具体改进建议"
}}
```
        """
        
        try:
            # 调用LLM进行评估
            response = self.llm.invoke(prompt)
            
            # 解析LLM的响应
            import re
            import json
            
            # 尝试从响应中提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content
                
            # 尝试解析JSON
            result = json.loads(json_str)
            
            # 提取评估结果
            is_satisfactory = result.get("is_satisfactory", False)
            improvement_suggestions = result.get("improvement_suggestions", "")
            
            return is_satisfactory, improvement_suggestions if not is_satisfactory else None
            
        except Exception as e:
            print(f"评估RAG结果出错: {e}")
            
            # 如果评估过程出错，默认认为结果满足需求
            return True, None
    
    def reflect_and_improve(
        self, 
        user_query: str,
        intent_result: IntentClassificationResult,
        rag_result: Dict[str, Any],
        context: Dict[str, Any],
        iteration: int = 0
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        反思并改进RAG检索结果
        
        Args:
            user_query: 用户的原始查询
            intent_result: 意图分类结果
            rag_result: RAG检索结果
            context: 上下文信息
            iteration: 当前迭代次数
            
        Returns:
            Tuple[bool, Optional[str], Dict[str, Any]]: 
                (是否满足需求, 改进建议, 包含评估信息的RAG结果)
        """
        # 评估RAG结果
        is_satisfactory, improvement_suggestions = self.evaluate_rag_result(
            user_query, intent_result, rag_result, context, iteration
        )
        
        # 添加评估信息到RAG结果
        rag_result_with_reflection = rag_result.copy()
        rag_result_with_reflection["reflection"] = {
            "is_satisfactory": is_satisfactory,
            "improvement_suggestions": improvement_suggestions,
            "iteration": iteration
        }
        
        return is_satisfactory, improvement_suggestions, rag_result_with_reflection 