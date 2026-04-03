"""
RAG pipeline using LangChain: GoogleGenerativeAIEmbeddings + langchain_chroma.Chroma + Document.
Same API as RAGPipeline: index_documents(doc_paths), query(question) -> {context, chunks, query}.
"""

from pathlib import Path
from typing import List, Optional

from .document_loader import load_and_chunk_documents


def _get_langchain_documents(chunks: List[str], metadatas: List[dict]):
    """Convert (chunks, metadatas) to LangChain Document list."""
    from langchain_core.documents import Document
    return [
        Document(page_content=c, metadata=m or {})
        for c, m in zip(chunks, metadatas)
    ]


class RAGPipelineLangChain:
    """
    RAG using LangChain stack: GoogleGenerativeAIEmbeddings + langchain_chroma.Chroma + Document.
    Matches the example: Gemini embeddings, Chroma.from_documents, similarity_search.
    Same interface as RAGPipeline: index_documents(doc_paths), query(question).
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model_name: str = "models/gemini-embedding-001",
        embedding_api_key: Optional[str] = None,
        collection_name: str = "rag_docs",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.persist_directory = persist_directory or "./chroma_langchain_db"
        self.embedding_model_name = embedding_model_name
        self.embedding_api_key = embedding_api_key
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embeddings = None
        self._vector_store = None

    def _get_embeddings(self):
        if self._embeddings is None:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
            except ImportError:
                raise ImportError(
                    "LangChain Gemini support requires: pip install langchain-google-genai"
                )
            import os
            api_key = self.embedding_api_key or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Set GOOGLE_API_KEY or pass embedding_api_key")
            self._embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model_name)
        return self._embeddings

    def _get_vector_store(self, create_if_missing: bool = True):
        """Get or create Chroma vector store (LangChain)."""
        try:
            from langchain_chroma import Chroma
        except ImportError:
            raise ImportError(
                "LangChain Chroma requires: pip install langchain-chroma"
            )
        if self._vector_store is not None:
            return self._vector_store
        if create_if_missing:
            # Create empty store that we'll replace in index_documents, or load existing
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self._get_embeddings(),
            )
            return self._vector_store
        return Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self._get_embeddings(),
        )

    def index_documents(
        self,
        doc_paths: List[Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """
        Load documents, chunk, embed with Gemini (LangChain), store in LangChain Chroma.
        Returns number of chunks indexed.
        """
        doc_paths = [Path(p) for p in doc_paths]
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        chunks, metadatas = load_and_chunk_documents(
            doc_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not chunks:
            return 0

        documents = _get_langchain_documents(chunks, metadatas)
        embeddings = self._get_embeddings()

        try:
            from langchain_chroma import Chroma
        except ImportError:
            raise ImportError("pip install langchain-chroma")

        self._vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        return len(chunks)

    def index_langchain_documents(self, documents: List) -> int:
        """
        Index a list of LangChain Document objects (e.g. from your own chunking).
        Same as: Chroma.from_documents(documents, embedding=gemini_embeddings).
        """
        if not documents:
            return 0
        try:
            from langchain_chroma import Chroma
        except ImportError:
            raise ImportError("pip install langchain-chroma")
        embeddings = self._get_embeddings()
        self._vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        return len(documents)

    def query(
        self,
        question: str,
        n_results: int = 5,
        include_documents: bool = True,
    ) -> dict:
        """
        Semantic search via LangChain Chroma. Returns same shape as RAGPipeline:
        { "context", "chunks" (text, metadata, distance), "query" }.
        """
        if not question or not question.strip():
            return {"context": "", "chunks": [], "query": question}

        vs = self._get_vector_store(create_if_missing=True)
        # similarity_search_with_score returns (Document, distance)
        results = vs.similarity_search_with_score(question.strip(), k=n_results)

        docs = []
        chunks_out = []
        for doc, score in results:
            docs.append(doc.page_content)
            chunks_out.append({
                "text": doc.page_content,
                "metadata": doc.metadata or {},
                "distance": float(score),
            })

        context = "\n\n---\n\n".join(docs) if docs else ""

        return {
            "query": question.strip(),
            "context": context,
            "chunks": chunks_out if include_documents else [],
        }

    def clear(self) -> None:
        """Remove this collection from the vector store."""
        self._vector_store = None
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(name=self.collection_name)
        except Exception:
            pass
