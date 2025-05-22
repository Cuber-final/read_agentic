"""
BookRAG系统简单示例

这个脚本展示了如何使用BookRAG系统进行书籍问答。
为了便于演示，这里使用了模拟数据，没有实际连接到书籍数据库或RAG引擎。
"""

import sys
import os
import json
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from bookrag.app import BookRAGApp


def create_mock_book_metadata() -> Dict[str, Any]:
    """创建模拟的书籍元数据"""
    return {
        "book_id": "book_123",
        "book_title": "人工智能：现代方法",
        "current_chapter_id": "chapter_4",
        "current_chapter_title": "第4章：机器学习基础"
    }


def create_mock_selected_text() -> Dict[str, Any]:
    """创建模拟的选中文本"""
    return {
        "text": "机器学习是人工智能的一个子领域，主要研究如何使计算机系统从数据中学习。",
        "chapter_id": "chapter_4",
        "paragraph_id": "p_4_12",
        "start_index": 0,
        "end_index": 31
    }


def create_mock_conversation_history() -> List[Dict[str, Any]]:
    """创建模拟的对话历史"""
    return [
        {
            "role": "user",
            "content": "机器学习和深度学习有什么区别？",
            "timestamp": 1630000000.0
        },
        {
            "role": "assistant",
            "content": "机器学习是一个广泛的领域，涵盖了各种让计算机从数据中学习的方法。而深度学习是机器学习的一个子领域，特别关注使用多层神经网络（深度神经网络）进行学习。深度学习模型能够自动从原始数据中提取特征，而传统机器学习方法通常需要人工设计特征。",
            "timestamp": 1630000010.0
        }
    ]


def main():
    """主函数"""
    # 创建BookRAG应用实例
    app = BookRAGApp()
    
    # 创建模拟数据
    book_metadata = create_mock_book_metadata()
    selected_text = create_mock_selected_text()
    conversation_history = create_mock_conversation_history()
    
    # 准备演示查询
    print("\n===== BookRAG系统演示 =====\n")
    
    # 示例1：选中文本相关查询
    query1 = "这句话中提到的机器学习主要研究什么？"
    print(f"查询1（选中文本相关）: {query1}")
    print(f"选中文本: {selected_text['text']}")
    
    result1 = app.process_query(
        query=query1,
        book_metadata=book_metadata,
        selected_text=selected_text,
        conversation_history=conversation_history
    )
    
    print(f"\n回复: {result1['response']}")
    print(f"识别的意图: {result1['intent']}")
    print("\n" + "-" * 50 + "\n")
    
    # 示例2：书籍一般性查询
    query2 = "这本书的第4章主要讲了什么内容？"
    print(f"查询2（书籍一般性）: {query2}")
    
    result2 = app.process_query(
        query=query2,
        book_metadata=book_metadata,
        conversation_history=conversation_history
    )
    
    print(f"\n回复: {result2['response']}")
    print(f"识别的意图: {result2['intent']}")
    print("\n" + "-" * 50 + "\n")
    
    # 示例3：闲聊查询
    query3 = "你是谁开发的？"
    print(f"查询3（闲聊）: {query3}")
    
    result3 = app.process_query(
        query=query3,
        book_metadata=book_metadata,
        conversation_history=conversation_history
    )
    
    print(f"\n回复: {result3['response']}")
    print(f"识别的意图: {result3['intent']}")
    
    print("\n===== 演示结束 =====\n")


if __name__ == "__main__":
    main() 