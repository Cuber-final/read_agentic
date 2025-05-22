"""
RAG接口模块 (RAGInterface)

封装与底层RAG系统（如LlamaIndex）的交互。
它接收由QueryUnderstandingAgent生成的优化查询，
调用RAG系统执行检索，并返回检索到的相关文档片段（chunks）。
"""

from typing import Dict, List, Any, Optional
import json


class RAGResult:
    """
    RAG检索结果类
    
    封装RAG系统返回的检索结果，包含检索到的文档片段及其元数据。
    """
    
    def __init__(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化RAG检索结果
        
        Args:
            query: 执行的查询语句
            chunks: 检索到的文档片段列表
            metadata: 结果元数据
        """
        self.query = query
        self.chunks = chunks
        self.metadata = metadata or {}
    
    def get_combined_text(self) -> str:
        """
        获取所有检索到的文档片段的组合文本
        
        Returns:
            str: 组合的文本内容
        """
        texts = [chunk.get("text", "") for chunk in self.chunks]
        return "\n\n".join(texts)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将RAG结果转换为字典形式
        
        Returns:
            Dict[str, Any]: 字典形式的RAG结果
        """
        return {
            "query": self.query,
            "chunks": self.chunks,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGResult':
        """
        从字典创建RAG结果实例
        
        Args:
            data: 字典形式的RAG结果数据
            
        Returns:
            RAGResult: 创建的RAG结果实例
        """
        return cls(
            query=data.get("query", ""),
            chunks=data.get("chunks", []),
            metadata=data.get("metadata", {})
        )


class RAGInterface:
    """
    RAG接口类
    
    封装与底层RAG系统的交互，提供统一的查询接口。
    """
    
    def __init__(self, rag_engine=None, default_top_k=5):
        """
        初始化RAG接口
        
        Args:
            rag_engine: 底层RAG引擎，如LlamaIndex查询引擎
            default_top_k: 默认返回的最大结果数
        """
        self.rag_engine = rag_engine
        self.default_top_k = default_top_k
    
    def execute_query(self, query: str, metadata_filters: List[Dict[str, Any]] = None, top_k: int = None) -> RAGResult:
        """
        执行单个RAG查询
        
        Args:
            query: 查询文本
            metadata_filters: 元数据过滤条件
            top_k: 返回的最大结果数
            
        Returns:
            RAGResult: RAG检索结果
        """
        top_k = top_k or self.default_top_k
        metadata_filters = metadata_filters or []
        
        # 如果没有设置rag_engine，返回模拟数据
        if not self.rag_engine:
            return self._mock_rag_result(query, metadata_filters, top_k)
        
        # 实际应用中，这里应该调用真实的RAG引擎
        # 示例: 调用LlamaIndex查询引擎
        try:
            # 根据metadata_filters构建过滤条件
            filter_dict = {}
            for filter_item in metadata_filters:
                field = filter_item.get("field")
                value = filter_item.get("value")
                operator = filter_item.get("operator", "eq")  # 默认为等于
                
                if field and value is not None:
                    # 实际实现应根据具体RAG引擎的API调整
                    filter_dict[field] = {"value": value, "operator": operator}
            
            # 调用rag_engine执行查询
            # 示例实现，实际代码应根据使用的RAG引擎调整
            # response = self.rag_engine.query(
            #     query,
            #     filter_dict=filter_dict if filter_dict else None,
            #     top_k=top_k
            # )
            
            # 将响应转换为RAGResult格式
            # chunks = [
            #     {
            #         "text": node.text,
            #         "metadata": node.metadata,
            #         "score": node.score
            #     }
            #     for node in response.nodes
            # ]
            
            # return RAGResult(
            #     query=query,
            #     chunks=chunks,
            #     metadata={"total_chunks_found": len(chunks)}
            # )
            
            # 临时使用模拟数据
            return self._mock_rag_result(query, metadata_filters, top_k)
            
        except Exception as e:
            print(f"RAG查询执行出错: {e}")
            # 返回空结果
            return RAGResult(
                query=query,
                chunks=[],
                metadata={"error": str(e)}
            )
    
    def execute_queries(self, queries: List[Dict[str, Any]]) -> List[RAGResult]:
        """
        执行多个RAG查询
        
        Args:
            queries: 查询列表，每个查询是一个包含query_text、metadata_filters等键的字典
            
        Returns:
            List[RAGResult]: RAG检索结果列表
        """
        results = []
        
        for query_item in queries:
            query_text = query_item.get("query_text", "")
            metadata_filters = query_item.get("metadata_filters", [])
            top_k = query_item.get("top_k", self.default_top_k)
            
            result = self.execute_query(query_text, metadata_filters, top_k)
            results.append(result)
        
        return results
    
    def execute_query_plan(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行查询规划
        
        Args:
            query_plan: 由QueryUnderstandingAgent生成的查询规划
            
        Returns:
            Dict[str, Any]: 包含所有查询结果的字典
        """
        # 获取查询列表
        final_queries = query_plan.get("final_queries", [])
        
        if not final_queries:
            # 如果没有查询，返回空结果
            return {
                "results": [],
                "combined_chunks": [],
                "execution_plan": query_plan.get("execution_plan", "未提供执行计划")
            }
        
        # 执行所有查询
        results = []
        for query_item in final_queries:
            query_text = query_item.get("query_text", "")
            metadata_filters = query_item.get("metadata_filters", [])
            purpose = query_item.get("purpose", "未指定目的")
            
            result = self.execute_query(query_text, metadata_filters)
            results.append({
                "query": query_text,
                "purpose": purpose,
                "result": result.to_dict()
            })
        
        # 合并所有结果的chunks
        all_chunks = []
        for result_item in results:
            chunks = result_item.get("result", {}).get("chunks", [])
            # 添加来源查询信息
            for chunk in chunks:
                if "source_query" not in chunk:
                    chunk["source_query"] = result_item.get("query", "")
                    chunk["query_purpose"] = result_item.get("purpose", "")
            all_chunks.extend(chunks)
        
        # 返回结果
        return {
            "results": results,
            "combined_chunks": all_chunks,
            "execution_plan": query_plan.get("execution_plan", "未提供执行计划")
        }
    
    def _mock_rag_result(self, query: str, metadata_filters: List[Dict[str, Any]], top_k: int) -> RAGResult:
        """
        生成模拟的RAG结果，用于测试
        
        Args:
            query: 查询文本
            metadata_filters: 元数据过滤条件
            top_k: 返回的最大结果数
            
        Returns:
            RAGResult: 模拟的RAG检索结果
        """
        # 生成一些模拟的文档片段
        mock_chunks = []
        for i in range(min(3, top_k)):  # 最多生成3个或top_k个片段
            mock_chunks.append({
                "text": f"这是关于'{query}'的模拟文档片段 #{i+1}。这里包含了一些与查询相关的内容，可能会提到书中的一些概念、人物或事件。",
                "metadata": {
                    "chapter_id": f"chapter_{i+1}",
                    "chapter_title": f"第{i+1}章 模拟章节",
                    "paragraph_id": f"p_{i*10+5}",
                    "score": 0.95 - (i * 0.1)  # 模拟相关性分数
                }
            })
        
        # 如果有元数据过滤条件，添加一些相关信息
        if metadata_filters:
            filter_info = []
            for filter_item in metadata_filters:
                field = filter_item.get("field", "")
                value = filter_item.get("value", "")
                filter_info.append(f"{field}={value}")
            
            filter_text = "，".join(filter_info)
            for chunk in mock_chunks:
                chunk["text"] += f" 应用了以下过滤条件：{filter_text}。"
        
        return RAGResult(
            query=query,
            chunks=mock_chunks,
            metadata={
                "total_chunks_found": len(mock_chunks),
                "is_mock_data": True
            }
        ) 