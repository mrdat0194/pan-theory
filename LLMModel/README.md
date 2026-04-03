# LLMModel: RAG System with VectorDB

A basic **Retrieval-Augmented Generation (RAG)** pipeline that indexes your documents in a vector database and exposes retrieved context for use in a **chatbot**.

## Overview


|            |                                                                |
| ---------- | -------------------------------------------------------------- |
| **Input**  | Document files (PDF, TXT, MD)                                  |
| **Output** | Retrieved context + optional prompt ready for an LLM / chatbot |


Flow:

1. **Index**: Load docs → chunk text → embed (open-source, Gemini, or OpenAI) → store in **ChromaDB** (VectorDB).
2. **Query**: User question → embed → similarity search → return top-k chunks as **context**.
3. **Chatbot**: Context + question are combined into a prompt you can send to any LLM.

## Setup

From the project root (or from `LLMModel`):

```bash
pip install -r LLMModel/requirements.txt
```

Dependencies:

- **chromadb** – in-memory or persistent vector store
- **sentence-transformers** – default embeddings (open-source, local)
- **PyPDF2** – PDF text extraction (optional if you only use .txt/.md)
- For **Gemini** embeddings: `pip install google-generativeai` and set `GEMINI_API_KEY`
- For **OpenAI** embeddings: `pip install openai` and set `OPENAI_API_KEY`

## Quick Start

### 1. Index documents

```python
from pathlib import Path
from LLMModel import RAGPipeline

rag = RAGPipeline(persist_directory="./db/local")  # or None for in-memory only

doc_paths = [
    Path("path/to/manual.pdf"),
    Path("path/to/notes.txt"),
]
num_chunks = rag.index_documents(doc_paths)
print(f"Indexed {num_chunks} chunks")
```

### 2. Query for chatbot context

```python
# Get context + metadata for a user question
out = rag.query("What is the refund policy?", n_results=5)

# Ready for your LLM:
print(out["context"])   # Concatenated relevant chunks
print(out["query"])     # Original question
print(out["chunks"])    # List of { "text", "metadata", "distance" }
```


## PDF Build/Query Scripts (Current Default)

For `questionandanswer.pdf`, two ready scripts are included:

- `build_questionandanswer_vector_index.py` (index/build step)
- `query_questionandanswer_vector_db.py` (query step)

These scripts currently default to:

- `--backend gemini` (Gemini embeddings via `RAGPipelineLangChain`)
- `--brain gemini` (Gemini answer generation in query script)
- Local fallback model arg: `Xenova/multilingual-e5-small` (mapped to `intfloat/multilingual-e5-small` in Python)

Set environment variable first:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
```

Build index:

```bash
python -m LLMModel.build_questionandanswer_vector_index --rebuild
```

Query:

```bash
python -m LLMModel.query_questionandanswer_vector_db --question "your question"
```

Interactive query mode:

```bash
python -m LLMModel.query_questionandanswer_vector_db
```

Use local embeddings instead:

```bash
python -m LLMModel.build_questionandanswer_vector_index --backend local --local-embedding-model Xenova/multilingual-e5-small --rebuild
python -m LLMModel.query_questionandanswer_vector_db --backend local --brain gemini --question "your question"
```

## Embeddings: Open-source vs Gemini vs OpenAI

The default **does not** use Gemini or Google knowledge. It uses **sentence-transformers** (e.g. `all-MiniLM-L6-v2`): open-source weights, runs locally, no API key.


| Option                                    | What it uses                                                                                                                           | Best when                                                                               |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Open-source** (`sentence_transformers`) | Local model (e.g. MiniLM, all-mpnet). No external API.                                                                                 | Free, private, offline, good enough for many RAG apps.                                  |
| **Gemini API**                            | Google’s embedding models (e.g. `text-embedding-004`). Uses Google’s weights/knowledge only for *embedding* (not search over the web). | You want high-quality embeddings and are fine with API cost and sending text to Google. |
| **OpenAI**                                | OpenAI embedding models (e.g. `text-embedding-3-small`).                                                                               | You already use OpenAI; same tradeoffs as Gemini (cost, data sent to provider).         |


**Which is better?**

- **Use open-source** when: you want no API cost, full data privacy, or offline use; quality is often sufficient for in-doc RAG.
- **Use Gemini (or OpenAI)** when: you need the best retrieval quality, handle many languages, or want to align with an API-based stack; you accept per-token cost and sending data to the provider.

You can switch by setting `embedding_provider` (and optionally `embedding_model_name` / `embedding_api_key`):

```python
# Default: open-source, local (no Gemini/Google knowledge)
rag = RAGPipeline(persist_directory="./db/local")

# Gemini API (set GEMINI_API_KEY or pass embedding_api_key)
rag = RAGPipeline(
    persist_directory="./db/gemini",
    embedding_provider="gemini",
    embedding_model_name="models/text-embedding-004",
)

# OpenAI
rag = RAGPipeline(
    persist_directory="./db/openai",
    embedding_provider="openai",
    embedding_model_name="text-embedding-3-small",
)
```

## LangChain version (Gemini + Chroma + Document)

To match the **LangChain** example (e.g. `GoogleGenerativeAIEmbeddings`, `langchain_chroma.Chroma`, `Document`), use `**RAGPipelineLangChain`**. Same API as `RAGPipeline`: `index_documents(doc_paths)` and `query(question)` → `{context, chunks, query}`.

**Install:**

```bash
pip install langchain-google-genai langchain-chroma langchain-core
```

**From file paths (like current RAG):**

```python
from pathlib import Path
from LLMModel import RAGPipelineLangChain

# os.environ["GEMINI_API_KEY"] = "your-api-key-here"

rag = RAGPipelineLangChain(
    persist_directory="./db/gemini",
    embedding_model_name="models/text-embedding-004",
)
rag.index_documents([Path("path/to/doc.pdf")])

out = rag.query("What happens to my unused vacation time?", n_results=1)
print(out["context"])
print(out["chunks"][0]["metadata"])
```

**From LangChain `Document` list (matches your example):**

```python
from langchain_core.documents import Document
from LLMModel import RAGPipelineLangChain

rag = RAGPipelineLangChain(persist_directory="./db/gemini")

chunks = [
    Document(page_content="Employees get 15 days of paid time off per year.", metadata={"page": 1}),
    Document(page_content="Vacation days do not roll over to the next year.", metadata={"page": 2}),
    Document(page_content="The company covers 80% of health insurance premiums.", metadata={"page": 3}),
]
rag.index_langchain_documents(chunks)

results = rag.query("What happens to my unused vacation time?", n_results=1)
print(results["chunks"][0]["text"])
print(results["chunks"][0]["metadata"]["page"])
```

**What changed vs current LLMModel:**


| Your LangChain example                                                 | LLMModel adaptation                                                                       |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")`      | Used inside `RAGPipelineLangChain` (same model)                                           |
| `Chroma.from_documents(documents=chunks, embedding=gemini_embeddings)` | `index_documents(doc_paths)` or `index_langchain_documents(chunks)`                       |
| `vector_store.similarity_search(user_question, k=1)`                   | `rag.query(question, n_results=1)` → same content in `out["context"]` and `out["chunks"]` |
| `Document(page_content=..., metadata=...)`                             | Supported via `index_langchain_documents()`; from files we build `Document` internally    |



## Configuration

- **RAGPipeline**
  - `persist_directory`: Directory for ChromaDB persistence; `None` = ephemeral.
  - `embedding_provider`: `"sentence_transformers"` (default), `"gemini"`, or `"openai"`.
  - `embedding_model_name`: Model name for the chosen provider (optional; defaults per provider).
  - `embedding_api_key`: API key for Gemini/OpenAI (or use env `GEMINI_API_KEY` / `OPENAI_API_KEY`).
  - `collection_name`: ChromaDB collection name.
  - `chunk_size` / `chunk_overlap`: Document chunking (default 500 / 100 chars).
- **RAGPipelineLangChain** (optional; requires langchain-google-genai, langchain-chroma, langchain-core)
  - Same as above where applicable; uses `GoogleGenerativeAIEmbeddings` and `langchain_chroma.Chroma`.
  - `index_langchain_documents(documents)` to index a list of LangChain `Document` objects.

## File Layout

```
LLMModel/
├── README.md           # This intro
├── RAG_SYSTEM.md       # Comprehensive Architecture & Deployment Guide
├── requirements.txt    # chromadb, sentence-transformers, PyPDF2; optional LangChain
├── __init__.py         # RAGPipeline, RAGPipelineLangChain (if deps)
├── build_questionandanswer_vector_index.py  # Build/index questionandanswer.pdf
├── query_questionandanswer_vector_db.py     # Query persisted VectorDB
├── document_loader.py  # Load PDF/TXT/MD and chunk
├── embeddings.py       # Embedding backends: sentence_transformers, gemini, openai
├── rag.py              # RAG pipeline (index + query)
├── rag_langchain.py    # LangChain: Gemini + Chroma + Document (same API)
└── rag_langchain.py    # LangChain: Gemini + Chroma + Document (same API)
```

## Summary

- **Input**: Docs (PDF, TXT, MD) → indexed into VectorDB (ChromaDB).
- **Output**: For each user question you get **context** (retrieved chunks) and a **prompt** string, so you can plug any LLM and use the result in your chatbot.

