# BookRAG - 增强的书籍问答系统

BookRAG是一个基于Agent的增强型RAG系统，专为书籍问答设计。系统可以处理用户对书籍内容的查询，通过智能决策和RAG优化提供精准答案。

## 系统架构

BookRAG系统由以下核心模块组成：

1. **UserInputHandler**: 处理来自前端的原始用户输入，将其结构化。
2. **IntentClassifier**: 分析用户查询的意图，确定处理策略。
3. **ContextManager**: 收集与用户查询相关的上下文信息。
4. **QueryUnderstandingAgent**: 基于Function Calling实现的Agent，负责深度理解查询并规划RAG策略。
5. **RAGInterface**: 封装与底层RAG系统的交互，执行检索。
6. **ReflectionAgent**: 评估RAG结果质量，决定是否需要迭代优化。
7. **ResponseSynthesizer**: 生成最终回复。

系统遵循PRAI (Planning-Retrieval-Analysis-Integration) 范式，流程如下：

```
用户输入 → 意图分类 → 上下文收集 → 查询规划 → RAG检索 → 结果反思 → [迭代优化] → 回复生成
```

## 特点

- **智能意图识别**：能够识别多种查询意图，包括选中文本相关查询、书籍一般性问题、术语解释等。
- **上下文感知**：考虑用户选中的文本、当前阅读章节、对话历史等上下文信息。
- **迭代优化RAG**：通过反思机制评估检索结果，必要时调整查询策略。
- **Function Calling实现**：利用OpenAI的Function Calling特性，实现更精确的Agent工具调用。

## 使用方法

### 安装依赖

```bash
pip install autochain
```

### 基本用法

```python
from bookrag.app import BookRAGApp

# 创建应用实例
app = BookRAGApp()

# 处理用户查询
result = app.process_query(
    query="这本书的核心观点是什么？",
    book_metadata={
        "book_id": "book_123",
        "book_title": "人工智能：现代方法",
        "current_chapter_id": "chapter_4",
        "current_chapter_title": "第4章：机器学习基础"
    }
)

# 输出回复
print(result["response"])
```

### 带选中文本的查询

```python
result = app.process_query(
    query="这段文字在讲什么？",
    book_metadata={...},
    selected_text={
        "text": "机器学习是人工智能的一个子领域，主要研究如何使计算机系统从数据中学习。",
        "chapter_id": "chapter_4",
        "paragraph_id": "p_4_12"
    }
)
```

## 运行示例

```bash
python -m bookrag.examples.simple_demo
```

## 自定义配置

可以通过传入自定义的LLM模型和RAG引擎来配置系统：

```python
from autochain.models.chat_openai import ChatOpenAI

app = BookRAGApp(
    intent_llm=ChatOpenAI(model="gpt-3.5-turbo"),
    query_agent_llm=ChatOpenAI(model="gpt-4"),
    rag_engine=your_custom_rag_engine
)
``` 