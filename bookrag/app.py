"""
书籍RAG系统应用 (BookRAG App)

这是书籍RAG系统的主应用程序文件，整合了各个模块，实现了完整的书籍问答功能。
该应用采用了基于Agent的增强型RAG系统设计，包含意图分类、查询规划、RAG检索和回复生成等核心功能。
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

from autochain.models.chat_openai import ChatOpenAI
# TODO 设计兼容openai compatible的llm ，比如还可以支持deepseek api等
from bookrag.user_input_handler import UserInputHandler, StructuredUserInput
from bookrag.intent_classifier import IntentClassifier, IntentType, IntentClassificationResult
from bookrag.context_manager import ContextManager
from bookrag.query_understanding_agent import QueryUnderstandingAgent
from bookrag.rag_interface import RAGInterface
from bookrag.reflection_agent import ReflectionAgent
from bookrag.response_synthesizer import ResponseSynthesizer


class BookRAGApp:
    """
    书籍RAG系统应用类
    
    整合了所有模块，提供完整的书籍问答功能。
    """
    
    def __init__(
        self, 
        book_content_provider=None, 
        rag_engine=None,
        intent_llm=None,
        query_agent_llm=None,
        reflection_llm=None,
        response_llm=None
    ):
        """
        初始化书籍RAG应用
        
        Args:
            book_content_provider: 提供书籍内容的对象
            rag_engine: RAG检索引擎
            intent_llm: 用于意图分类的LLM
            query_agent_llm: 用于查询规划的LLM
            reflection_llm: 用于结果反思的LLM
            response_llm: 用于生成回复的LLM
        """
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("BookRAGApp")
        
        # 初始化各个模块
        self.user_input_handler = UserInputHandler()
        self.intent_classifier = IntentClassifier(llm=intent_llm)
        self.context_manager = ContextManager(book_content_provider=book_content_provider)
        self.query_agent = QueryUnderstandingAgent(llm=query_agent_llm)
        self.rag_interface = RAGInterface(rag_engine=rag_engine)
        self.reflection_agent = ReflectionAgent(llm=reflection_llm)
        self.response_synthesizer = ResponseSynthesizer(llm=response_llm)
        
        self.logger.info("BookRAG应用初始化完成")
    
    def process_query(
        self,
        query: str,
        book_metadata: Dict[str, Any],
        selected_text: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        处理用户查询，返回回复和相关信息
        
        Args:
            query: 用户的查询文本
            book_metadata: 书籍元数据
            selected_text: 用户选中的文本信息
            conversation_history: 对话历史
            max_iterations: 最大迭代次数
            
        Returns:
            Dict[str, Any]: 包含回复和处理过程信息的字典
        """
        # 记录开始处理
        self.logger.info(f"开始处理用户查询: {query}")
        
        # 步骤1: 处理用户输入
        user_input = self.user_input_handler.process_raw_input(
            query=query,
            book_metadata=book_metadata,
            selected_text=selected_text,
            conversation_history=conversation_history
        )
        self.logger.info("用户输入处理完成")
        
        # 步骤2: 意图分类
        intent_result = self.intent_classifier.classify(user_input)
        self.logger.info(f"意图分类结果: {intent_result.intent.name}, 置信度: {intent_result.confidence.value}")
        
        # 对于某些意图，可以采取特殊处理
        if intent_result.intent == IntentType.CHIT_CHAT:
            # 闲聊直接调用LLM生成回复，跳过RAG流程
            self.logger.info("检测到闲聊意图，跳过RAG流程")
            return {
                "response": self._handle_chitchat(user_input),
                "intent": intent_result.intent.name,
                "process_info": {
                    "rag_used": False,
                    "intent_confidence": intent_result.confidence.value
                }
            }
        
        # 步骤3: 收集上下文
        context = self.context_manager.get_structured_context(user_input, intent_result)
        context_dict = self.context_manager.get_context_as_dict(context)
        self.logger.info("上下文收集完成")
        
        # 步骤4-7: 迭代查询规划、RAG检索和结果反思
        final_rag_result, planning_info = self._iterative_rag_process(
            user_input.query, 
            intent_result, 
            context_dict, 
            max_iterations
        )
        
        # 步骤8: 生成回复
        response = self.response_synthesizer.generate_response(
            user_input.query,
            intent_result,
            final_rag_result,
            context_dict
        )
        self.logger.info("回复生成完成")
        
        # 返回结果
        return {
            "response": response,
            "intent": intent_result.intent.name,
            "process_info": {
                "rag_used": True,
                "intent_confidence": intent_result.confidence.value,
                "planning_info": planning_info,
                "rag_reflection": final_rag_result.get("reflection", {})
            }
        }
    
    def _handle_chitchat(self, user_input: StructuredUserInput) -> str:
        """
        处理闲聊类型的查询
        
        Args:
            user_input: 结构化的用户输入
            
        Returns:
            str: 生成的回复
        """
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        prompt = f"""
你是一个友好的书籍阅读助手。用户发送了一条与当前书籍内容无关的消息，请给予适当的回应。
不要假装你可以做书籍内容之外的事情，如果用户询问与书籍无关的专业知识，可以礼貌地建议他们回到书籍内容。

用户消息: "{user_input.query}"

请直接回复，无需任何前缀。
        """
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"生成闲聊回复出错: {e}")
            return "抱歉，我现在无法回答这个问题。我是一个书籍阅读助手，可以帮您解答关于正在阅读的书籍的问题。"
    
    def _iterative_rag_process(
        self,
        query: str,
        intent_result: IntentClassificationResult,
        context: Dict[str, Any],
        max_iterations: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        迭代执行查询规划、RAG检索和结果反思
        
        Args:
            query: 用户的查询文本
            intent_result: 意图分类结果
            context: 上下文信息
            max_iterations: 最大迭代次数
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: (最终RAG结果, 规划信息)
        """
        # 记录规划信息
        planning_info = {
            "iterations": [],
            "final_iteration": 0
        }
        
        # 初始化反馈为None
        reflection_feedback = None
        
        # 迭代执行查询规划、RAG检索和结果反思
        for iteration in range(max_iterations):
            self.logger.info(f"开始第{iteration+1}次迭代")
            
            # 步骤4: 查询规划
            query_plan = self.query_agent.plan_query(
                query, intent_result, context, reflection_feedback
            )
            
            # 检查查询规划是否有最终查询
            final_queries = query_plan.get("final_queries", [])
            if not final_queries:
                self.logger.warning("查询规划未返回有效查询，使用默认查询")
                final_queries = [{
                    "query_text": query,
                    "metadata_filters": [],
                    "purpose": "默认查询"
                }]
                query_plan["final_queries"] = final_queries
            
            self.logger.info(f"查询规划完成，生成了{len(final_queries)}个查询")
            
            # 步骤5: RAG检索
            rag_result = self.rag_interface.execute_query_plan(query_plan)
            self.logger.info(f"RAG检索完成，获取了{len(rag_result.get('combined_chunks', []))}个文档片段")
            
            # 步骤6: 反思结果
            is_satisfactory, improvement_suggestions, rag_result_with_reflection = self.reflection_agent.reflect_and_improve(
                query, intent_result, rag_result, context, iteration
            )
            
            # 记录本次迭代信息
            planning_info["iterations"].append({
                "iteration": iteration + 1,
                "query_plan_summary": {
                    "query_count": len(final_queries),
                    "execution_plan": query_plan.get("execution_plan", ""),
                    "planning_process": query_plan.get("planning_process", {})
                },
                "rag_result_summary": {
                    "chunk_count": len(rag_result.get("combined_chunks", [])),
                },
                "reflection_summary": {
                    "is_satisfactory": is_satisfactory,
                    "has_improvement_suggestions": improvement_suggestions is not None
                }
            })
            
            # 步骤7: 决定是否需要继续迭代
            if is_satisfactory:
                self.logger.info(f"第{iteration+1}次迭代结果令人满意，停止迭代")
                planning_info["final_iteration"] = iteration + 1
                return rag_result_with_reflection, planning_info
            
            # 如果不满意但已达到最大迭代次数，也停止迭代
            if iteration == max_iterations - 1:
                self.logger.info(f"已达到最大迭代次数{max_iterations}，停止迭代")
                planning_info["final_iteration"] = iteration + 1
                return rag_result_with_reflection, planning_info
            
            # 更新反馈，进行下一次迭代
            reflection_feedback = improvement_suggestions
            self.logger.info(f"第{iteration+1}次迭代结果不满意，准备下一次迭代")
        
        # 这里实际上不会执行到，因为在循环中已经有返回条件
        return rag_result_with_reflection, planning_info 