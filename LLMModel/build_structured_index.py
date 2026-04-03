import os
import sys
from pathlib import Path
import pdfplumber

# Add the project root to Python's path so absolute imports work
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from LLMModel.rag import RAGPipeline
except ImportError:
    from rag import RAGPipeline

def clean_text(text):
    if not text:
        return ""
    # Remove excessive newlines and spaces often found in PDF table extraction
    return " ".join(text.split())

def main():
    pdf_path = Path("LLMModel/questionandanswer.pdf")
    if not pdf_path.exists():
        print(f"Error: Document not found at {pdf_path}")
        return

    print(f"Extracting structured Q&A from {pdf_path}...")
    
    qa_pairs = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    # Based on the image:
                    # Col 1: Category (Nhóm câu hỏi) -> row[0]
                    # Col 2: Question (Anh Thành hỏi / Chị Trang hỏi) -> row[1]
                    # Col 3: Answer (Gợi ý trả lời) -> row[2]
                    
                    if len(row) < 3:
                        continue
                        
                    question = clean_text(row[1])
                    answer = clean_text(row[2])
                    
                    # Skip header rows or empty rows
                    if not question or not answer:
                        continue
                    if "Anh Thành hỏi" in question or "Gợi ý trả lời" in answer:
                        continue
                    if "Chị Trang hỏi" in question:
                        continue

                    # Construct a structured document for the vector DB
                    # We store the question and answer together so the vector represents the relationship
                    doc_content = f"Question: {question}\nAnswer: {answer}"
                    qa_pairs.append({
                        "content": doc_content,
                        "metadata": {
                            "question": question,
                            "answer": answer,
                            "source": str(pdf_path),
                            "type": "qa_pair"
                        }
                    })

    if not qa_pairs:
        print("No Q&A pairs found in the table.")
        return

    print(f"Found {len(qa_pairs)} Q&A pairs. Initializing RAG pipeline...")
    
    # We'll use a specific collection/directory for structured data to avoid mixing
    rag = RAGPipeline(
        persist_directory="LLMModel/db/structured",
        collection_name="structured_qa"
    )
    
    # Clear existing data in this specific collection to ensure fresh start
    print("Clearing 'structured_qa' collection...")
    rag.clear()

    print("Indexing documents...")
    
    # RAGPipeline.index_documents normally takes file paths, but we have raw strings.
    # Let's bypass the loader and add directly if possible, or create temp files.
    # Looking at rag.py, it uses load_and_chunk_documents.
    
    # Since we want EXACT matching for these pairs, we'll use the collection directly 
    # or expose a method in RAGPipeline.
    
    collection = rag._get_collection()
    embed_fn = rag._get_embedding_fn()
    
    documents = [p["content"] for p in qa_pairs]
    metadatas = [p["metadata"] for p in qa_pairs]
    ids = [f"qa_{i}" for i in range(len(qa_pairs))]
    
    embeddings = embed_fn(documents)
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Successfully indexed {len(qa_pairs)} Q&A pairs into 'structured_qa' collection.")

if __name__ == "__main__":
    main()
