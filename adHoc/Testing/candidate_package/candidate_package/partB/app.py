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

# Load environment variables from .env file
load_dotenv()

# B1: LangChain / embedding / vector store dependencies
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

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
    1. Load all .txt files from the docs/ directory
    2. Split into chunks (chunk_size=500, overlap=50)
    3. Generate embeddings using OpenAIEmbeddings (text-embedding-3-small)
    4. Store in a FAISS in-memory vector store
    5. Return the retriever
    """
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")

    # Load documents
    loader = DirectoryLoader(docs_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings and store in FAISS
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": int(os.getenv("TOP_K", 4))})


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
    1. Retrieve relevant chunks using the retriever
    2. Construct a prompt with context and instructions
    3. Call the LLM (gpt-4o)
    4. Return answer and source references
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised")

    # Define LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Define prompt template
    system_prompt = (
        "You are an expert assistant for EDB policy questions. "
        "Use the following pieces of retrieved context to answer the question. "
        "For every claim you make, you must cite the source document filename. "
        "If the answer is not in the provided context, say 'I don't know'. "
        "Keep the answer concise and professional.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoke chain
    response = rag_chain.invoke({"input": request.question})

    # Prepare sources
    sources = []
    for doc in response["context"]:
        filename = os.path.basename(doc.metadata.get("source", "unknown"))
        preview = doc.page_content[:100].replace("\n", " ")
        sources.append(f"{filename} — {preview}...")

    return AnswerResponse(
        answer=response["answer"],
        sources=list(set(sources))  # Unique sources
    )


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
