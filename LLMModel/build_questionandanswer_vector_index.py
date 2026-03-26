"""
Build (index) a VectorDB for `questionandanswer.pdf` from `LLMModel/`.

This is the "index/build" step only: it loads the PDF, chunks it, embeds it,
and stores vectors in ChromaDB.

Run:
  python -m LLMModel.build_questionandanswer_vector_index --backend local

Gemini (embeddings) variant:
  set GOOGLE_API_KEY=...
  python -m LLMModel.build_questionandanswer_vector_index --backend gemini

By default, it uses:
  - backend: local embeddings (sentence-transformers)
  - persist dir:
      LLMModel/rag_db_local   (local)
      LLMModel/rag_db_gemini  (gemini)
  - collection name: questionandanswer_full
"""

from __future__ import annotations

import argparse
from pathlib import Path

from LLMModel.rag import RAGPipeline


def get_pdf_and_paths() -> tuple[Path, Path, str]:
    base_dir = Path(__file__).resolve().parent
    pdf_path = base_dir / "questionandanswer.pdf"
    collection_name = "questionandanswer_full"
    return pdf_path, base_dir, collection_name


def make_rag(backend: str, persist_dir: Path, collection_name: str) -> object:
    if backend == "local":
        return RAGPipeline(
            persist_directory=str(persist_dir),
            embedding_provider="sentence_transformers",
            collection_name=collection_name,
        )

    if backend == "gemini":
        # Optional path: only works when LangChain Gemini dependencies are installed.
        from LLMModel.rag_langchain import RAGPipelineLangChain

        return RAGPipelineLangChain(
            persist_directory=str(persist_dir),
            collection_name=collection_name,
            embedding_model_name="models/text-embedding-004",
        )

    raise ValueError(f"Unknown backend: {backend}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["local", "gemini"],
        default="local",
        help="Embedding backend to use for indexing.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index (clear the Chroma collection) before indexing.",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    pdf_path, base_dir, collection_name = get_pdf_and_paths()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    persist_dir = base_dir / ("rag_db_gemini" if args.backend == "gemini" else "rag_db_local")
    marker = persist_dir / f"indexed_questionandanswer_{args.backend}_{collection_name}.marker"
    persist_dir.mkdir(parents=True, exist_ok=True)

    if marker.exists() and not args.rebuild:
        print(f"Index already exists (marker found): {marker}")
        print("Use --rebuild to force rebuilding.")
        return

    rag = make_rag(args.backend, persist_dir, collection_name)

    if args.rebuild:
        print("Clearing existing vectors/collection...")
        try:
            rag.clear()  # both pipelines expose clear()
        except Exception:
            pass

    print(f"Indexing {pdf_path.name} into: {persist_dir}")
    n = rag.index_documents([pdf_path], chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"Indexed {n} chunks.")

    marker.write_text("ok", encoding="utf-8")
    print(f"Wrote marker: {marker}")


if __name__ == "__main__":
    main()

