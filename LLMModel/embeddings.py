"""
Embedding backends: open-source (sentence-transformers), Gemini API, or OpenAI.
Same interface: encode(texts) -> list of vectors (list of float lists).
"""

from typing import List, Union


def encode_sentence_transformers(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """Open-source, local. No API key. No Google/OpenAI knowledge."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts).tolist()


def encode_gemini(
    texts: List[str],
    model_name: str = "models/gemini-embedding-001",
    api_key: str = None,
    task_type: str = "retrieval_document",
) -> List[List[float]]:
    """Google Gemini embedding API. task_type: 'retrieval_document' for index, 'retrieval_query' for search."""
    import google.generativeai as genai
    key = api_key or _get_env("GEMINI_API_KEY") or _get_env("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Gemini requires GOOGLE_API_KEY or embedding_api_key")
    genai.configure(api_key=key)
    out = []
    for t in texts:
        result = genai.embed_content(model=model_name, content=t, task_type=task_type)
        out.append(result["embedding"])
    return out


def encode_openai(texts: List[str], model_name: str = "text-embedding-3-small", api_key: str = None) -> List[List[float]]:
    """OpenAI embedding API. Requires API key."""
    from openai import OpenAI
    key = api_key or _get_env("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI requires OPENAI_API_KEY or embedding_api_key")
    client = OpenAI(api_key=key)
    r = client.embeddings.create(input=texts, model=model_name)
    order = {e.index: e.embedding for e in r.data}
    return [order[i] for i in range(len(texts))]


def _get_env(name: str) -> str:
    import os
    return os.environ.get(name) or os.environ.get(name.replace("_", "-"), "")


# Provider registry: "sentence_transformers" | "gemini" | "openai"
def get_embedding_fn(provider: str, model_name: str = None, api_key: str = None):
    """Return a callable: (texts: List[str]) -> List[List[float]]."""
    model_name = model_name or {
        "sentence_transformers": "all-MiniLM-L6-v2",
        "gemini": "models/gemini-embedding-001",
        "openai": "text-embedding-3-small",
    }.get(provider)
    if provider == "sentence_transformers":
        def fn(texts, task_type=None):
            return encode_sentence_transformers(texts, model_name=model_name)
        return fn
    if provider == "gemini":
        def fn(texts, task_type="retrieval_document"):
            return encode_gemini(texts, model_name=model_name, api_key=api_key, task_type=task_type)
        return fn
    if provider == "openai":
        def fn(texts, task_type=None):
            return encode_openai(texts, model_name=model_name, api_key=api_key)
        return fn
    raise ValueError(f"Unknown embedding provider: {provider}. Use sentence_transformers, gemini, or openai.")
