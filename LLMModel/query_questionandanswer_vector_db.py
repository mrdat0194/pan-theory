"""
Query the VectorDB built from `questionandanswer.pdf`.

This is the "query/run" step: it loads your persisted ChromaDB and retrieves
top-k relevant chunks for a question.

Run:
  python -m LLMModel.query_questionandanswer_vector_db --backend gemini --brain gemini --question "your question"

Optional interactive mode:
  python -m LLMModel.query_questionandanswer_vector_db --backend gemini
  (then type questions)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from LLMModel.rag import RAGPipeline

# Windows console sometimes uses a legacy code page (cp1258), which can crash
# `print()` for Vietnamese characters. Force UTF-8 for stdout/stderr if possible.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _normalize_local_embedding_model(model_name: str) -> str:
    if model_name == "Xenova/multilingual-e5-small":
        return "intfloat/multilingual-e5-small"
    return model_name


def make_rag(backend: str, persist_dir: Path, collection_name: str, local_embedding_model: str) -> object:
    if backend == "local":
        return RAGPipeline(
            persist_directory=str(persist_dir),
            embedding_provider="sentence_transformers",
            embedding_model_name=_normalize_local_embedding_model(local_embedding_model),
            collection_name=collection_name,
        )

    if backend == "gemini":
        from LLMModel.rag_langchain import RAGPipelineLangChain

        return RAGPipelineLangChain(
            persist_directory=str(persist_dir),
            collection_name=collection_name,
            embedding_model_name="models/text-embedding-004",
        )

    raise ValueError(f"Unknown backend: {backend}")


def gemini_answer(question: str, context: str, model_name: str) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return "Gemini disabled: set GOOGLE_API_KEY in your environment."

    try:
        import google.generativeai as genai
    except ImportError:
        return "Gemini disabled: install package `google-generativeai`."

    genai.configure(api_key=api_key)
    prompt = (
        "Answer the user's question using only the provided context. "
        "If context is insufficient, say you do not have enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    try:
        response = genai.GenerativeModel(model_name).generate_content(prompt)
        return (response.text or "").strip() if response else ""
    except Exception as e:
        # Keep retrieval usable even if Gemini fails (bad key, model not found, etc.)
        return f"Gemini error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["local", "gemini"],
        default="gemini",
        help="Embedding backend to use for query-time embeddings.",
    )
    parser.add_argument(
        "--brain",
        choices=["none", "gemini"],
        default="gemini",
        help="Optional answer generation model on top of retrieved context.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model used as answer generator.",
    )
    parser.add_argument(
        "--local-embedding-model",
        type=str,
        default="Xenova/multilingual-e5-small",
        help="Local sentence-transformers embedding model.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="How many chunks to retrieve.")
    parser.add_argument("--question", type=str, default="", help="Question to ask. If empty, runs interactive mode.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    collection_name = "questionandanswer_full"
    persist_dir = base_dir / ("rag_db_gemini" if args.backend == "gemini" else "rag_db_local")

    rag = make_rag(
        args.backend,
        persist_dir,
        collection_name,
        local_embedding_model=args.local_embedding_model,
    )

    def run_one(q: str):
        out = rag.query(q, n_results=args.top_k, include_documents=True)
        context = out.get("context", "") or ""
        chunks = out.get("chunks", []) or []

        print("\n" + "=" * 80)
        print(f"Question: {q}")

        if not chunks or not context.strip():
            print("No matches found. Run the build script first:")
            print("  python -m LLMModel.build_questionandanswer_vector_index --backend", args.backend)
            return

        best = chunks[0]
        print("\nAnswer (best retrieved chunk):")
        print(best.get("text", "").strip())

        if args.brain == "gemini":
            final_answer = gemini_answer(q, context, model_name=args.gemini_model)
            print("\nFinal answer (Gemini):")
            print(final_answer or "(No text returned by Gemini.)")

        print("\nRetrieved chunks (top-k):")
        for i, c in enumerate(chunks, 1):
            dist = c.get("distance", None)
            meta = c.get("metadata", {}) or {}
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "(no metadata)"
            print(f"\n[{i}] distance={dist} | {meta_str}")
            print(c.get("text", "").strip()[:600] + ("..." if len(c.get("text", "")) > 600 else ""))

    if args.question.strip():
        run_one(args.question.strip())
    else:
        while True:
            q = input("\nAsk a question (Enter to quit): ").strip()
            if not q:
                break
            run_one(q)


if __name__ == "__main__":
    main()

