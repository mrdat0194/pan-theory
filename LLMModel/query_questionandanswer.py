import sys
from pathlib import Path

# Add the project root to Python's path so absolute imports work
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from LLMModel.rag import RAGPipeline
except ImportError:
    from rag import RAGPipeline

def main():
    print("Initializing RAG pipeline from persisted database...")
    # Initialize the RAG pipeline pointing to the same persist directory
    rag = RAGPipeline(
        persist_directory="LLMModel/db/structured",
        collection_name="structured_qa"
    )
    
    # Check if a question was passed as a command-line argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Tại sao vẫn cần các App partner?"
    
    print(f"Query: '{question}'\n")
    
    # Query the RAG pipeline with n_results=1 for exact matching
    try:
        results = rag.query(question, n_results=1)
        
        chunks = results.get("chunks", [])
        if chunks:
            best_match = chunks[0]
            # Use metadata to get the clean answer if available
            answer = best_match.get("metadata", {}).get("answer") or best_match.get("text", "")
            
            print("-" * 40)
            print("Best Answer:")
            print("-" * 40)
            print(answer)
        else:
            print("-" * 40)
            print("No matching answer found. Ensure the document is indexed first.")
        print("-" * 40)
        
    except Exception as e:
        print(f"Error querying the database: {e}")
        print("Make sure to run 'python LLMModel/build_structured_index.py' first to initialize the database.")

if __name__ == "__main__":
    main()
