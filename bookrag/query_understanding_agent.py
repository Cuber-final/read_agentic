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

from typing import Dict, List, Optional, Any, Union

from autochain.agent.conversational_agent.conversational_agent import ConversationalAgent
from autochain.agent.message import ChatMessageHistory
from autochain.agent.structs import AgentAction, AgentFinish
from autochain.models.base import BaseLanguageModel
from autochain.tools.base import Tool

from bookrag.query_planning_tools import get_query_planning_tools
from bookrag.intent_classifier import IntentClassificationResult

QUERY_UNDERSTANDING_PROMPT = """你是书籍问答系统的查询理解与规划核心Agent。
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

请注意：
1. 每次只能执行一个工具调用
2. 根据工具执行结果决定下一步行动
3. 当完成所有必要步骤后，返回最终优化的查询列表
"""

class QueryUnderstandingAgent(ConversationalAgent):
    """
    查询理解与规划Agent类
    
    继承自ConversationalAgent，利用其对话式工具调用机制，
    完成查询理解与优化任务。
    """
    
    def __init__(self, llm=None, tools=None):
        """
        初始化查询理解与规划Agent
        
        Args:
            llm: 用于Agent规划的语言模型，默认为None，将使用默认模型
            tools: Agent可以调用的工具列表，默认为None，将使用标准的查询规划工具
        """
        # 调用父类初始化方法
        super().__init__(
            llm=llm,
            tools=tools or get_query_planning_tools(),
            prompt=QUERY_UNDERSTANDING_PROMPT
        )
        
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
        self.memory.clear()
        
        # 构建初始消息
        history = ChatMessageHistory()
        history.add_user_message(
            f"""用户查询: "{user_query}"
意图: {intent_result.intent.name}
上下文信息:
{self._prepare_context_summary(context)}
{f'RAG反思反馈: {reflection_feedback}' if reflection_feedback else ''}"""
        )
        
        # 执行规划过程
        intermediate_steps = []
        try:
            # 添加最大步骤数限制，防止无限循环
            max_steps = 10
            step_count = 0
            
            while step_count < max_steps:
                step_count += 1
                
                # 获取下一步行动
                next_step = self.plan(
                    history=history,
                    intermediate_steps=intermediate_steps
                )
                
                # 如果是完成状态，检查是否有最终查询
                if isinstance(next_step, AgentFinish):
                    # 尝试从中间步骤中找到最终查询
                    final_queries = self._extract_final_queries(intermediate_steps)
                    if final_queries:
                        return {
                            "final_queries": final_queries,
                            "execution_plan": "成功完成查询规划",
                            "planning_process": self._extract_planning_process(intermediate_steps)
                        }
                    else:
                        return {
                            "final_queries": [{
                                "query_text": user_query,
                                "metadata_filters": [],
                                "purpose": "原始查询"
                            }],
                            "execution_plan": "使用原始查询",
                            "planning_process": self._extract_planning_process(intermediate_steps)
                        }
                
                # 执行工具调用
                if isinstance(next_step, AgentAction):
                    # 确保有工具定义
                    if next_step.tool not in self.allowed_tools:
                        history.add_system_message(
                            f"错误: 工具 '{next_step.tool}' 未定义或不可用。请选择有效的工具。"
                        )
                        continue
                    
                    # 执行工具并获取结果
                    tool = self.allowed_tools[next_step.tool]
                    try:
                        # 传入工具输入
                        tool_input = next_step.tool_input
                        tool_output = tool.run(tool_input)
                        
                        # 更新Agent操作的tool_output
                        next_step.tool_output = tool_output
                        
                        # 添加到中间步骤
                        intermediate_steps.append(next_step)
                        
                        # 将工具输出添加到历史记录
                        history.add_assistant_message(
                            f"执行工具 {next_step.tool}: {next_step.tool_input}"
                        )
                        history.add_system_message(
                            f"工具输出: {tool_output}"
                        )
                    except Exception as e:
                        error_message = f"执行工具 {next_step.tool} 时出错: {str(e)}"
                        history.add_system_message(error_message)
                        print(error_message)
            
            # 如果达到最大步骤数但没有得到最终查询，返回默认结果
            print(f"达到最大步骤数 {max_steps}，但未完成规划")
            final_queries = self._extract_final_queries(intermediate_steps)
            if final_queries:
                return {
                    "final_queries": final_queries,
                    "execution_plan": "部分完成查询规划(达到最大步骤数)",
                    "planning_process": self._extract_planning_process(intermediate_steps)
                }
            else:
                return {
                    "final_queries": [{
                        "query_text": user_query,
                        "metadata_filters": [],
                        "purpose": "原始查询(达到最大步骤数)"
                    }],
                    "execution_plan": "使用原始查询",
                    "planning_process": self._extract_planning_process(intermediate_steps)
                }
                
        except Exception as e:
            print(f"查询规划出错: {e}")
            return {
                "final_queries": [{
                    "query_text": user_query,
                    "metadata_filters": [],
                    "purpose": "原始查询 (出错后回退)"
                }],
                "execution_plan": "直接使用原始查询",
                "error": str(e),
                "planning_process": self._extract_planning_process(intermediate_steps)
            }
    
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
        
        return "\n".join(summary_parts)
    
    def _extract_final_queries(self, intermediate_steps: List[AgentAction]) -> Optional[List[Dict[str, Any]]]:
        """
        从中间步骤中提取最终的查询列表
        
        Args:
            intermediate_steps: 中间步骤列表
            
        Returns:
            Optional[List[Dict[str, Any]]]: 最终查询列表，如果没有找到则返回None
        """
        # 从后向前遍历，找到最近的final_queries生成结果
        for step in reversed(intermediate_steps):
            if step.tool == "generate_final_queries_for_rag":
                # 检查工具输出是否有效
                if isinstance(step.tool_output, dict) and "final_queries" in step.tool_output:
                    return step.tool_output.get("final_queries", [])
                
                # 如果输出格式不对，尝试修复
                if isinstance(step.tool_output, str):
                    try:
                        import json
                        import re
                        
                        # 尝试从字符串中提取JSON
                        json_match = re.search(r'```json\s*(.*?)\s*```', step.tool_output, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = step.tool_output
                            
                        # 尝试解析JSON
                        parsed_output = json.loads(json_str)
                        if "final_queries" in parsed_output:
                            return parsed_output.get("final_queries", [])
                    except:
                        pass
        
        # 如果没有找到最终查询或解析失败，则尝试从rewrite_and_expand_query工具中提取查询
        for step in reversed(intermediate_steps):
            if step.tool == "rewrite_and_expand_query":
                if isinstance(step.tool_output, dict):
                    optimized_query = step.tool_output.get("optimized_main_query")
                    if optimized_query:
                        return [{
                            "query_text": optimized_query,
                            "metadata_filters": [],
                            "purpose": "优化后的查询"
                        }]
        
        return None
    
    def _extract_planning_process(self, intermediate_steps: List[AgentAction]) -> Dict[str, Any]:
        """
        从中间步骤中提取规划过程信息
        
        Args:
            intermediate_steps: 中间步骤列表
            
        Returns:
            Dict[str, Any]: 规划过程信息
        """
        planning_process = {}
        for step in intermediate_steps:
            # 检查是否有工具和工具输出
            if not hasattr(step, 'tool') or not hasattr(step, 'tool_output'):
                continue
                
            # 将每个工具的结果添加到规划过程中
            if step.tool not in planning_process:
                planning_process[step.tool] = {
                    "input": step.tool_input,
                    "output": step.tool_output
                }
            
        return planning_process 