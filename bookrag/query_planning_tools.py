"""
查询理解与规划工具 (Query Planning Tools)

这个模块提供了一系列工具，用于实现QueryUnderstandingAgent的功能。
每个工具负责查询理解与规划过程中的一个特定子任务，如槽位提取、查询改写等。
这些工具遵循AutoChain中Tool的接口设计，可以被Agent调用。
"""

from typing import Dict, List, Optional, Any, Callable
import json

from autochain.tools.base import Tool
from autochain.models.chat_openai import ChatOpenAI


def create_slot_extractor_tool(llm=None) -> Tool:
    """
    创建槽位提取工具
    
    这个工具从用户查询和相关上下文中提取关键信息、实体、问题焦点等。
    
    Args:
        llm: 用于执行槽位提取的语言模型，默认为None，将使用默认模型
        
    Returns:
        Tool: 配置好的槽位提取工具
    """
    # 如果没有传入LLM，则使用默认的轻量级模型
    llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    def extract_query_slots(
        user_query: str,
        selected_text: str = "",
        conversation_history_summary: str = ""
    ) -> Dict[str, Any]:
        """
        从用户原始查询、选中文本和对话历史中提取关键实体、问题焦点、意图等槽位信息，
        用于深入理解用户具体诉求。
        
        Args:
            user_query: 用户的原始文本输入
            selected_text: 用户在阅读器中选中的文本，可能为空
            conversation_history_summary: 与当前查询相关的对话历史摘要
            
        Returns:
            Dict[str, Any]: 提取出的槽位信息
        """
        # 构建Prompt
        prompt = f"""
你是一个专业的文本分析助手。请从以下用户查询和上下文中提取关键信息（槽位）。

用户查询: "{user_query}"
选中文本: "{selected_text}"
对话历史摘要: "{conversation_history_summary}"

请提取以下信息:
1. 核心问题类型（如定义解释、因果关系、比较对比、观点摘取等）
2. 关键实体（如书中提到的人物、概念、事件等）
3. 问题焦点（用户最关心的核心信息点）
4. 时间/位置信息（如果有）
5. 用户可能的隐含意图（潜在需求）

输出格式要求：
```json
{{
  "question_type": "问题类型",
  "key_entities": ["实体1", "实体2", ...],
  "question_focus": "问题焦点",
  "temporal_spatial_info": "时间位置信息",
  "implicit_intent": "隐含意图"
}}
```

请确保输出是有效的JSON格式。对于不适用或没有找到的字段，请使用空字符串或空数组。
        """
        
        try:
            # 调用LLM进行槽位提取
            response = llm.invoke(prompt)
            
            # 从响应中提取JSON
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content
                
            # 尝试解析JSON
            slots = json.loads(json_str)
            return slots
            
        except Exception as e:
            print(f"槽位提取出错: {e}")
            # 返回默认空槽位
            return {
                "question_type": "",
                "key_entities": [],
                "question_focus": "",
                "temporal_spatial_info": "",
                "implicit_intent": ""
            }
    
    # 创建并返回Tool
    return Tool(
        name="extract_query_slots",
        func=extract_query_slots,
        description="从用户原始查询、选中文本和对话历史中提取关键实体、问题焦点、意图等槽位信息，用于深入理解用户具体诉求。",
        arg_description={
            "user_query": "用户的原始文本输入。",
            "selected_text": "用户在阅读器中选中的文本，可能为空。",
            "conversation_history_summary": "与当前查询相关的对话历史摘要。"
        }
    )


def create_query_rewriter_tool(llm=None) -> Tool:
    """
    创建查询改写工具
    
    这个工具根据已提取的槽位和上下文，对原始查询进行改写或扩展。
    
    Args:
        llm: 用于执行查询改写的语言模型，默认为None，将使用默认模型
        
    Returns:
        Tool: 配置好的查询改写工具
    """
    # 如果没有传入LLM，则使用功能更强大的模型
    llm = llm or ChatOpenAI(model="gpt-4", temperature=0.2)
    
    def rewrite_and_expand_query(
        original_query: str,
        intent: str,
        extracted_slots: Dict[str, Any],
        full_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        根据已提取的槽位、用户意图及完整上下文，对原始查询进行改写或扩展，
        生成更清晰、信息更丰富的查询版本，以提高RAG检索效果。
        
        Args:
            original_query: 用户的原始查询文本
            intent: 用户的意图分类结果
            extracted_slots: 由extract_query_slots工具输出的槽位信息
            full_context: 包含选中文本及其周边、当前章节/书籍信息、对话历史的完整上下文摘要
            
        Returns:
            Dict[str, Any]: 包含优化查询及改写说明的字典
        """
        # 构建Prompt
        # 格式化上下文信息，以便在Prompt中使用
        selected_text_info = ""
        if "selected_text" in full_context:
            selected_text = full_context.get("selected_text", {})
            text = selected_text.get("text", "")
            section_title = selected_text.get("section_title", "")
            selected_text_info = f"选中文本: '{text}'\n选中文本所在小节: '{section_title}'"
        
        book_info = ""
        if "book" in full_context:
            book = full_context.get("book", {})
            book_info = f"书籍: '{book.get('title', '')}'\n当前章节: '{book.get('current_chapter_title', '')}'"
        
        # 转换槽位信息为文本
        slots_text = ""
        if extracted_slots:
            slots_text = f"""
提取的槽位信息:
- 问题类型: {extracted_slots.get('question_type', '')}
- 关键实体: {', '.join(extracted_slots.get('key_entities', []))}
- 问题焦点: {extracted_slots.get('question_focus', '')}
- 时间/位置信息: {extracted_slots.get('temporal_spatial_info', '')}
- 隐含意图: {extracted_slots.get('implicit_intent', '')}
"""
        
        prompt = f"""
你是一个专业的查询优化助手。请根据提供的信息，对用户的原始查询进行改写和扩展，使其更适合RAG系统检索。

原始查询: "{original_query}"
用户意图: {intent}

{slots_text}

上下文信息:
{selected_text_info}
{book_info}

请执行以下优化:
1. 消除歧义和模糊表达
2. 补充关键上下文信息
3. 添加同义词或相关术语
4. 使查询更具体、明确

输出要求:
1. 提供一个优化后的主查询
2. 可选地提供1-2个补充查询，以覆盖不同角度
3. 简要说明你做出的改写/扩展及其原因

输出格式:
```json
{{
  "optimized_main_query": "优化后的主查询",
  "supplementary_queries": ["补充查询1", "补充查询2"],
  "rewrite_explanation": "改写和扩展的说明"
}}
```
        """
        
        try:
            # 调用LLM进行查询改写
            response = llm.invoke(prompt)
            
            # 从响应中提取JSON
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content
                
            # 尝试解析JSON
            result = json.loads(json_str)
            return result
            
        except Exception as e:
            print(f"查询改写出错: {e}")
            # 返回原始查询
            return {
                "optimized_main_query": original_query,
                "supplementary_queries": [],
                "rewrite_explanation": "查询改写过程中出错，返回原始查询。"
            }
    
    # 创建并返回Tool
    return Tool(
        name="rewrite_and_expand_query",
        func=rewrite_and_expand_query,
        description="根据已提取的槽位、用户意图及完整上下文，对原始查询进行改写或扩展，生成更清晰、信息更丰富的查询版本，以提高RAG检索效果。例如补充上下文关键词、同义词替换、消除歧义等。",
        arg_description={
            "original_query": "用户的原始查询文本。",
            "intent": "用户的意图分类结果 (例如 RAG_SPECIFIC_TEXT, RAG_BOOK_GENERAL)。",
            "extracted_slots": "由extract_query_slots工具输出的槽位信息。",
            "full_context": "包含选中文本及其周边、当前章节/书籍信息、对话历史的完整上下文摘要。"
        }
    )


def create_rag_strategy_planner_tool(llm=None) -> Tool:
    """
    创建RAG策略规划工具
    
    这个工具为优化后的查询规划RAG检索策略。
    
    Args:
        llm: 用于执行策略规划的语言模型，默认为None，将使用默认模型
        
    Returns:
        Tool: 配置好的RAG策略规划工具
    """
    # 如果没有传入LLM，则使用功能更强大的模型
    llm = llm or ChatOpenAI(model="gpt-4", temperature=0.1)
    
    def plan_rag_retrieval_strategy(
        optimized_query: str,
        intent: str,
        selected_text_info: Dict[str, Any],
        book_metadata_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        为优化后的查询规划RAG检索策略。
        
        Args:
            optimized_query: 优化后的查询文本
            intent: 用户意图
            selected_text_info: 关于选中文本的详细信息
            book_metadata_schema: 描述当前书籍可用的元数据字段及其类型
            
        Returns:
            Dict[str, Any]: RAG检索策略
        """
        # 构建Prompt
        has_selected_text = bool(selected_text_info and selected_text_info.get("text"))
        
        # 格式化元数据模式，以便在Prompt中使用
        metadata_fields = ""
        if book_metadata_schema:
            metadata_fields = "可用的书籍元数据字段:\n"
            for field, field_type in book_metadata_schema.items():
                metadata_fields += f"- {field}: {field_type}\n"
        
        prompt = f"""
你是一个RAG检索策略专家。请为以下查询规划最佳的检索策略。

优化后的查询: "{optimized_query}"
用户意图: {intent}
是否有选中文本: {"是" if has_selected_text else "否"}
{metadata_fields}

请规划以下检索策略:
1. 检索范围: 应该在哪个范围内检索？(如选中文本附近，当前章节，全书)
2. 子查询分解: 是否需要将查询分解为多个子查询？如果是，应如何分解？
3. 元数据过滤: 应使用哪些元数据字段进行过滤？
4. 检索参数: 建议的top_k值、相似度阈值等参数。
5. 特殊处理: 是否有需要特殊处理的情况？(如需要精确匹配的关键词)

输出格式:
```json
{{
  "retrieval_scope": "检索范围",
  "decompose_into_subqueries": true/false,
  "subqueries": ["子查询1", "子查询2"],
  "metadata_filters": [
    {{"field": "字段名", "value": "过滤值", "operator": "等于/包含/大于"}}
  ],
  "retrieval_params": {{
    "top_k": 5,
    "similarity_threshold": 0.7
  }},
  "special_handling": "特殊处理说明"
}}
```
        """
        
        try:
            # 调用LLM进行策略规划
            response = llm.invoke(prompt)
            
            # 从响应中提取JSON
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content
                
            # 尝试解析JSON
            strategy = json.loads(json_str)
            return strategy
            
        except Exception as e:
            print(f"RAG策略规划出错: {e}")
            # 返回默认策略
            return {
                "retrieval_scope": "current_chapter" if has_selected_text else "whole_book",
                "decompose_into_subqueries": False,
                "subqueries": [],
                "metadata_filters": [],
                "retrieval_params": {
                    "top_k": 5,
                    "similarity_threshold": 0.7
                },
                "special_handling": ""
            }
    
    # 创建并返回Tool
    return Tool(
        name="plan_rag_retrieval_strategy",
        func=plan_rag_retrieval_strategy,
        description="为优化后的查询规划RAG检索策略。包括决定检索范围（如仅限选中文本附近、当前章节、全书）、是否需要分解为子查询、以及如何利用书籍元数据（如章节标题、标签）进行过滤。",
        arg_description={
            "optimized_query": "由rewrite_and_expand_query工具输出的优化查询。",
            "intent": "用户意图。",
            "selected_text_info": "关于选中文本的详细信息（是否存在、内容、位置）。",
            "book_metadata_schema": "描述当前书籍可用的元数据字段及其类型，供规划时参考。"
        }
    )


def create_final_query_generator_tool(llm=None) -> Tool:
    """
    创建最终查询生成工具
    
    这个工具基于规划好的RAG检索策略和优化后的用户查询，生成最终用于提交给RAG系统的具体查询语句。
    
    Args:
        llm: 用于执行最终查询生成的语言模型，默认为None，将使用默认模型
        
    Returns:
        Tool: 配置好的最终查询生成工具
    """
    # 如果没有传入LLM，则使用合适的模型
    llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    def generate_final_queries_for_rag(
        optimized_query: str,
        rag_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        基于规划好的RAG检索策略和优化后的用户查询，生成最终用于提交给RAG系统的具体查询语句。
        
        Args:
            optimized_query: 优化后的核心查询内容
            rag_strategy: 由plan_rag_retrieval_strategy工具输出的策略对象
            
        Returns:
            Dict[str, Any]: 包含最终查询语句和执行计划的字典
        """
        # 格式化RAG策略，以便在Prompt中使用
        retrieval_scope = rag_strategy.get("retrieval_scope", "")
        decompose_into_subqueries = rag_strategy.get("decompose_into_subqueries", False)
        subqueries = rag_strategy.get("subqueries", [])
        metadata_filters = rag_strategy.get("metadata_filters", [])
        special_handling = rag_strategy.get("special_handling", "")
        
        # 构建Prompt
        prompt = f"""
你是一个RAG查询生成专家。请根据提供的查询和RAG策略，生成最终要提交给RAG系统的查询。

优化后的查询: "{optimized_query}"

RAG策略:
- 检索范围: {retrieval_scope}
- 是否分解为子查询: {"是" if decompose_into_subqueries else "否"}
- 子查询列表: {", ".join(subqueries) if subqueries else "无"}
- 元数据过滤: {metadata_filters if metadata_filters else "无"}
- 特殊处理: {special_handling if special_handling else "无"}

你的任务是:
1. 如果策略要求分解为子查询，则生成每个子查询的最终形式
2. 如果不需要分解，则生成单个最终查询
3. 为每个查询添加必要的关键词和修饰词，使其最适合向量检索

输出格式:
```json
{{
  "final_queries": [
    {{
      "query_text": "最终查询文本1",
      "metadata_filters": [
        {{"field": "字段名", "value": "过滤值", "operator": "等于/包含/大于"}}
      ],
      "purpose": "这个查询的目的"
    }},
    {{
      "query_text": "最终查询文本2",
      "metadata_filters": [...],
      "purpose": "这个查询的目的"
    }}
  ],
  "execution_plan": "执行这些查询的顺序和逻辑说明"
}}
```
        """
        
        try:
            # 调用LLM生成最终查询
            response = llm.invoke(prompt)
            
            # 从响应中提取JSON
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.content
                
            # 尝试解析JSON
            final_queries = json.loads(json_str)
            return final_queries
            
        except Exception as e:
            print(f"最终查询生成出错: {e}")
            # 返回简单的默认查询
            return {
                "final_queries": [
                    {
                        "query_text": optimized_query,
                        "metadata_filters": [],
                        "purpose": "主查询"
                    }
                ],
                "execution_plan": "直接执行主查询"
            }
    
    # 创建并返回Tool
    return Tool(
        name="generate_final_queries_for_rag",
        func=generate_final_queries_for_rag,
        description="基于规划好的RAG检索策略和优化后的用户查询，生成一个或多个最终用于提交给RAG系统的具体查询语句。",
        arg_description={
            "optimized_query": "优化后的核心查询内容。",
            "rag_strategy": "由plan_rag_retrieval_strategy工具输出的策略对象。"
        }
    )


def get_query_planning_tools(llm_config=None) -> List[Tool]:
    """
    获取所有查询规划工具的列表
    
    Args:
        llm_config: 不同工具使用的LLM配置，格式为{工具名称: LLM实例}
        
    Returns:
        List[Tool]: 查询规划工具列表
    """
    llm_config = llm_config or {}
    
    # 创建工具列表
    tools = [
        create_slot_extractor_tool(llm=llm_config.get("slot_extractor")),
        create_query_rewriter_tool(llm=llm_config.get("query_rewriter")),
        create_rag_strategy_planner_tool(llm=llm_config.get("rag_strategy_planner")),
        create_final_query_generator_tool(llm=llm_config.get("final_query_generator"))
    ]
    
    return tools 