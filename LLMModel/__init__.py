"""
LLMModel: RAG (Retrieval-Augmented Generation) with VectorDB.
Input: document files (PDF, TXT, MD). Output: context for chatbot.
"""

from .rag import RAGPipeline
from .chatbot import RAGChatbot

try:
    from .rag_langchain import RAGPipelineLangChain
    __all__ = ["RAGPipeline", "RAGPipelineLangChain", "RAGChatbot"]
except ImportError:
    RAGPipelineLangChain = None  # type: ignore
    __all__ = ["RAGPipeline", "RAGChatbot"]
