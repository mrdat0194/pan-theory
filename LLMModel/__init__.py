"""
LLMModel: RAG (Retrieval-Augmented Generation) with VectorDB.
Input: document files (PDF, TXT, MD). Output: context for chatbot.
"""

from .rag import RAGPipeline

try:
    from .rag_langchain import RAGPipelineLangChain
    __all__ = ["RAGPipeline", "RAGPipelineLangChain"]
except ImportError:
    RAGPipelineLangChain = None  # type: ignore
    __all__ = ["RAGPipeline"]
