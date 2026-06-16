"""
EDB Data Scientist Assessment — Part B
RAG Q&A Endpoint

Your task: complete the TODOs below to build a working FastAPI app
that answers questions grounded in the provided EDB policy documents.

Run with:  uvicorn app:app --reload
Test with: curl -X POST http://localhost:8000/ask \
               -H "Content-Type: application/json" \
               -d '{"question": "What is the Enterprise Development Grant?"}'
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# TODO (B1): import LangChain / embedding / vector store dependencies
# e.g. from langchain_community.document_loaders import TextLoader
#      from langchain.text_splitter import RecursiveCharacterTextSplitter
#      from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#      from langchain_community.vectorstores import FAISS

app = FastAPI(title="EDB Policy Q&A", version="1.0")


# ── Request / Response schemas ─────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


# ── B1: Document ingestion ─────────────────────────────────────────────────────
# Load, chunk, embed, and store the 3 policy documents in docs/
# This should run ONCE at startup (not on every request).

def build_vectorstore():
    """
    TODO:
    1. Load all .txt files from the docs/ directory
    2. Split into chunks (choose your chunk_size and overlap — document your choice in README.md)
    3. Generate embeddings using OpenAIEmbeddings or a local model
    4. Store in a FAISS or Chroma in-memory vector store
    5. Return the retriever
    """
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    # ... your code here ...
    raise NotImplementedError("Complete build_vectorstore()")


# Initialise at startup
try:
    retriever = build_vectorstore()
    print("✓ Vector store ready")
except NotImplementedError:
    retriever = None
    print("⚠  Vector store not yet implemented")


# ── B2: /ask endpoint ──────────────────────────────────────────────────────────

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """
    TODO:
    1. Retrieve the top-k most relevant chunks using the retriever
    2. Construct a prompt that:
       - Provides the retrieved chunks as context
       - Instructs the model to cite sources (filename + chunk preview)
       - Instructs the model to say "I don't know" for out-of-scope questions
    3. Call the LLM (ChatOpenAI or equivalent)
    4. Return the answer and a list of source references

    The response MUST follow this format:
      {"answer": "...", "sources": ["doc1.txt — chunk preview...", ...]}
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised")

    # ... your code here ...
    raise NotImplementedError("Complete the /ask endpoint")


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "vectorstore_ready": retriever is not None}


# ── B3: Manual test queries (run this block in a separate terminal) ────────────
# After starting the server, test these two queries and paste the output in README.md:
#
# ANSWERABLE (should cite a source):
#   {"question": "What support does EDB offer for workforce development?"}
#
# OUT-OF-SCOPE (model should decline):
#   {"question": "What is the current Singapore prime minister's salary?"}
