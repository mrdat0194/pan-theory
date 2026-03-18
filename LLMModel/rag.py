"""
RAG pipeline: index documents in VectorDB, query for context usable by a chatbot.
"""

from pathlib import Path
from typing import List, Optional

from .document_loader import load_and_chunk_documents
from .embeddings import get_embedding_fn


class RAGPipeline:
    """
    Basic RAG using a vector DB (Chroma). Input: doc files. Output: retrieved context for chatbot.
    Embeddings: open-source (sentence_transformers), Gemini API, or OpenAI — set embedding_provider.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_provider: str = "sentence_transformers",
        embedding_model_name: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        collection_name: str = "rag_docs",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.persist_directory = persist_directory
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        self.embedding_api_key = embedding_api_key
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_fn = None
        self._client = None
        self._collection = None

    def _get_embedding_fn(self):
        if self._embedding_fn is None:
            self._embedding_fn = get_embedding_fn(
                self.embedding_provider,
                model_name=self.embedding_model_name,
                api_key=self.embedding_api_key,
            )
        return self._embedding_fn

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError("chromadb is required. Install with: pip install chromadb")
            if self.persist_directory:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                self._client = chromadb.EphemeralClient()
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document chunks"},
            )
        return self._collection

    def index_documents(
        self,
        doc_paths: List[Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """
        Load documents, chunk, embed, and store in VectorDB.
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

        embed_fn = self._get_embedding_fn()
        embeddings = embed_fn(chunks)

        collection = self._get_collection()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
        return len(chunks)

    def query(
        self,
        question: str,
        n_results: int = 5,
        include_documents: bool = True,
    ) -> dict:
        """
        Retrieve relevant chunks for a question. Output is ready for chatbot use.

        Returns dict with:
          - "context": concatenated retrieved text (for LLM prompt)
          - "chunks": list of { "text", "metadata", "distance" } if include_documents
          - "query": original question
        """
        if not question or not question.strip():
            return {"context": "", "chunks": [], "query": question}

        embed_fn = self._get_embedding_fn()
        query_embedding = embed_fn([question.strip()], task_type="retrieval_query")

        collection = self._get_collection()
        count = collection.count()
        if count == 0:
            return {"context": "", "chunks": [], "query": question.strip()}

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results.get("distances") else []

        chunks = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
            chunks.append({"text": doc, "metadata": meta or {}, "distance": float(dist)})

        context = "\n\n---\n\n".join(docs) if docs else ""

        out = {
            "query": question.strip(),
            "context": context,
            "chunks": chunks if include_documents else [],
        }
        return out

    def clear(self) -> None:
        """Remove all documents from the collection (reset index)."""
        client = self._get_client()
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = None
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document chunks"},
        )
