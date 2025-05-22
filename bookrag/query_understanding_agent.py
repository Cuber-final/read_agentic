"""
查询理解与规划Agent (QueryUnderstandingAgent)

这是实现Planning范式的核心模块。它接收用户原始查询、意图分类结果和上下文信息，
然后通过调用一系列专门的工具，完成对查询的深度理解与RAG策略规划，最终生成优化后的查询。

主要任务包括：
- 槽位填充与关键词提取
- 上下文分析
- RAG策略规划
- 优化查询生成
"""

from typing import Dict, List, Optional, Any, Tuple

from autochain.agent.openai_functions_agent.openai_functions_agent import OpenAIFunctionsAgent
from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI
from autochain.tools.base import Tool

from bookrag.query_planning_tools import get_query_planning_tools
from bookrag.intent_classifier import IntentClassificationResult


class QueryUnderstandingAgent:
    """
    查询理解与规划Agent类
    
    基于AutoChain的OpenAIFunctionsAgent实现，使用Function Calling机制来调用
    一系列内部规划工具，完成查询理解与优化任务。
    """
    
    def __init__(self, llm=None, tools=None):
        """
        初始化查询理解与规划Agent
        
        Args:
            llm: 用于Agent规划的语言模型，默认为None，将使用默认模型
            tools: Agent可以调用的工具列表，默认为None，将使用标准的查询规划工具
        """
        # 如果没有传入LLM，则使用功能强大的模型
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0.1)
        
        # 如果没有传入工具，则使用标准的查询规划工具
        self.tools = tools or get_query_planning_tools()
        
        # 创建OpenAIFunctionsAgent
        self.agent = OpenAIFunctionsAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            prompt=self._get_agent_prompt()
        )
        
        # 创建Chain，用于执行Agent
        self.memory = BufferMemory()
        self.chain = Chain(agent=self.agent, memory=self.memory)
    
    def _get_agent_prompt(self) -> str:
        """
        获取Agent的系统提示
        
        Returns:
            str: Agent的系统提示
        """
        return """
你是书籍问答系统的查询理解与规划核心Agent。
你的目标是根据用户的输入、意图和提供的上下文，生成最优化的一个或多个查询语句，以供后续的RAG系统进行高效检索。

可用工具：
- extract_query_slots: 从用户输入中提取关键信息和槽位。
- rewrite_and_expand_query: 基于上下文优化和丰富用户查询。
- plan_rag_retrieval_strategy: 为优化后的查询规划详细的RAG检索策略。
- generate_final_queries_for_rag: 根据策略生成最终的RAG查询语句。

工作流程建议：
1. 首先使用 `extract_query_slots` 理解用户查询的核心。
2. 然后使用 `rewrite_and_expand_query` 结合上下文优化查询。
3. 接着使用 `plan_rag_retrieval_strategy` 制定检索方案。
4. 最后使用 `generate_final_queries_for_rag` 生成给RAG系统的具体查询。
5. 确保最终输出的是可以直接用于RAG检索的查询列表。
"""
    
    def reset(self):
        """重置Agent的内存状态"""
        self.memory = BufferMemory()
        self.chain = Chain(agent=self.agent, memory=self.memory)
    
    def _prepare_context_summary(self, context: Dict[str, Any]) -> str:
        """
        准备上下文摘要，用于传递给工具
        
        Args:
            context: 上下文字典
            
        Returns:
            str: 上下文摘要文本
        """
        summary_parts = []
        
        # 处理选中文本上下文
        if "selected_text" in context:
            selected_text = context["selected_text"]
            summary_parts.append(f"选中文本: '{selected_text.get('text', '')}'")
            if selected_text.get('previous_paragraphs'):
                summary_parts.append(f"前文内容: '{selected_text['previous_paragraphs'][-1] if selected_text['previous_paragraphs'] else ''}'")
            if selected_text.get('following_paragraphs'):
                summary_parts.append(f"后文内容: '{selected_text['following_paragraphs'][0] if selected_text['following_paragraphs'] else ''}'")
            if selected_text.get('section_title'):
                summary_parts.append(f"所在小节: '{selected_text.get('section_title', '')}'")
        
        # 处理书籍上下文
        if "book" in context:
            book = context["book"]
            summary_parts.append(f"书籍: '{book.get('title', '')}'")
            summary_parts.append(f"当前章节: '{book.get('current_chapter_title', '')}'")
        
        # 处理对话上下文
        if "conversation" in context:
            conv = context["conversation"]
            if conv.get("last_question") and conv.get("last_answer"):
                summary_parts.append(f"上一问题: '{conv.get('last_question', '')}'")
                summary_parts.append(f"上一回答: '{conv.get('last_answer', '')}'")
        
        # 合并摘要
        return "\n".join(summary_parts)
    
    def _format_metadata_schema(self, context: Dict[str, Any]) -> Dict[str, str]:
        """
        从上下文信息格式化书籍元数据模式
        
        Args:
            context: 上下文字典
            
        Returns:
            Dict[str, str]: 元数据模式字典
        """
        # 这是一个示例实现，实际应用中可能需要从数据库或配置中获取元数据模式
        metadata_schema = {
            "chapter_id": "string",
            "chapter_title": "string",
            "section_title": "string",
            "paragraph_id": "string",
            "is_dialogue": "boolean",
            "contains_quote": "boolean",
            "sentiment": "string (positive/negative/neutral)",
            "entities_mentioned": "string[]",
            "timestamp": "number",
            "location": "string"
        }
        
        return metadata_schema
    
    def _get_selected_text_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        从上下文获取选中文本信息
        
        Args:
            context: 上下文字典
            
        Returns:
            Dict[str, Any]: 选中文本信息
        """
        if "selected_text" not in context:
            return {}
        
        selected_text = context["selected_text"]
        return {
            "text": selected_text.get("text", ""),
            "chapter_id": selected_text.get("chapter_id", ""),
            "paragraph_id": selected_text.get("paragraph_id", ""),
            "exists": bool(selected_text.get("text", ""))
        }
    
    def plan_query(
        self, 
        user_query: str, 
        intent_result: IntentClassificationResult, 
        context: Dict[str, Any],
        reflection_feedback: str = None
    ) -> Dict[str, Any]:
        """
        规划并生成优化的RAG查询
        
        Args:
            user_query: 用户的原始查询
            intent_result: 意图分类结果
            context: 上下文信息
            reflection_feedback: 反思Agent的反馈，用于迭代优化
            
        Returns:
            Dict[str, Any]: 包含最终查询语句和其他规划信息的字典
        """
        # 重置Agent状态
        self.reset()
        
        # 准备初始消息
        initial_message = f"""
用户查询: "{user_query}"
用户意图: {intent_result.intent.name}
上下文信息概述:
{self._prepare_context_summary(context)}
"""
        
        if reflection_feedback:
            initial_message += f"\nRAG反思反馈: {reflection_feedback}"
        
        # 运行Agent链，让它根据需要调用各种工具
        try:
            result = self.chain.run(initial_message)
            
            # 分析结果，找到最终的查询规划
            # 通常是 generate_final_queries_for_rag 工具的最后一次调用结果
            final_result = None
            agent_steps = self.memory.get_memory().get("history", [])
            
            for step in reversed(agent_steps):
                if hasattr(step, "action") and step.action.tool == "generate_final_queries_for_rag":
                    final_result = step.observation
                    break
            
            if not final_result:
                # 如果没有找到最终结果，可能是因为Agent没有调用 generate_final_queries_for_rag
                # 返回一个默认结果
                return {
                    "final_queries": [
                        {
                            "query_text": user_query,
                            "metadata_filters": [],
                            "purpose": "原始查询"
                        }
                    ],
                    "execution_plan": "直接使用原始查询",
                    "planning_process": "简化处理 - Agent未生成完整规划"
                }
            
            # 添加规划过程信息
            planning_process = {}
            for step in agent_steps:
                if hasattr(step, "action") and hasattr(step, "observation"):
                    tool_name = step.action.tool
                    if tool_name not in planning_process:
                        planning_process[tool_name] = step.observation
            
            # 构建并返回最终结果
            result_with_process = final_result.copy() if isinstance(final_result, dict) else {}
            result_with_process["planning_process"] = planning_process
            
            return result_with_process
            
        except Exception as e:
            print(f"查询规划出错: {e}")
            
            # 返回默认结果
            return {
                "final_queries": [
                    {
                        "query_text": user_query,
                        "metadata_filters": [],
                        "purpose": "原始查询 (出错后回退)"
                    }
                ],
                "execution_plan": "直接使用原始查询",
                "error": str(e)
            } 