"""
Chatbot that uses RAG context. Consumes RAGPipeline output and optional LLM for replies.
"""

from typing import Callable, Optional

from .rag import RAGPipeline


# Default prompt template: inject context and user question for an LLM
DEFAULT_PROMPT_TEMPLATE = """You are an objective AI assistant. Use the following context to answer the question. 
If the context mentions specific company names like 'VNA' or 'Vortex' in a way that describes their specific implementation, 
generalize the answer to refer to 'the user', 'the enterprise', or 'the client' whenever appropriate. 
The goal is to provide a white-label, objective answer that applies to any user, 
unless the question specifically asks about a named entity.

Context:
{context}

Question: {query}

Answer:"""


class RAGChatbot:
    """
    Chatbot backed by RAG: each user message is used to retrieve context from VectorDB;
    output is a prompt (context + question) ready for your LLM, or you can plug a generator.
    """

    def __init__(
        self,
        rag: RAGPipeline,
        n_retrieve: int = 5,
        prompt_template: Optional[str] = None,
        llm_callback: Optional[Callable[[str], str]] = None,
    ):
        self.rag = rag
        self.n_retrieve = n_retrieve
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.llm_callback = llm_callback

    def get_context(self, user_message: str) -> dict:
        """
        Get RAG output for the user message (context + chunks). Use this to build your own reply.
        """
        return self.rag.query(user_message, n_results=self.n_retrieve, include_documents=True)

    def get_prompt(self, user_message: str) -> str:
        """
        Build the full prompt string (context + question) to send to an LLM.
        """
        out = self.get_context(user_message)
        return self.prompt_template.format(
            context=out["context"] or "(No relevant context found.)",
            query=out["query"],
        )

    def reply(self, user_message: str) -> str:
        """
        If llm_callback was set, return LLM reply; otherwise return the prompt (for manual use).
        """
        prompt = self.get_prompt(user_message)
        if self.llm_callback:
            return self.llm_callback(prompt)
        return prompt
