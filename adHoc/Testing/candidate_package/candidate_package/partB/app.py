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
from dotenv import load_dotenv

# B1: import LangChain / embedding / vector store dependencies
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

load_dotenv()

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

    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    # 1. Load all .txt files
    loader = DirectoryLoader(docs_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # 2. Split into chunks
    # I chose chunk_size=500 with overlap=50 to balance context preservation and precision.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 3. Generate embeddings
    embeddings = OpenAIEmbeddings()

    # 4. Store in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Return the retriever
    top_k = int(os.getenv("TOP_K", 4))
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


# Initialise at startup
try:
    retriever = build_vectorstore()
    print("✓ Vector store ready")
except Exception as e:
    retriever = None
    print(f"⚠  Vector store not initialised: {e}")


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

    # 1. Retrieve the top-k most relevant chunks
    docs = retriever.invoke(request.question)

    # 2. Construct a prompt
    context_text = ""
    sources = []
    for doc in docs:
        filename = os.path.basename(doc.metadata.get("source", "Unknown"))
        content_preview = doc.page_content[:100].replace("\n", " ") + "..."
        source_str = f"{filename} — {content_preview}"
        sources.append(source_str)
        context_text += f"\n---\nSOURCE: {filename}\nCONTENT: {doc.page_content}\n"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an EDB policy expert. Answer the user's question using ONLY the provided context. "
            "If the answer is not in the context, say 'I don't know'. "
            "When answering, you must cite the filename of the source you are using. "
            "\n\nContext:\n{context}"
        )),
        ("user", "{question}")
    ])

    # 3. Call the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt_template | llm

    try:
        response = chain.invoke({"context": context_text, "question": request.question})
        answer = response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    # 4. Return the answer and sources
    return AnswerResponse(answer=answer, sources=sources)


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
