"""
Load and chunk documents from files (PDF, TXT, MD) for RAG indexing.
"""

from pathlib import Path
from typing import List, Tuple

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


def load_text_file(path: Path) -> str:
    """Load plain text or markdown file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_pdf(path: Path) -> str:
    """Extract text from PDF."""
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)


def load_document(path: Path) -> str:
    """Load a single document by extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".txt", ".md", ".markdown"):
        return load_text_file(path)
    raise ValueError(f"Unsupported format: {suffix}. Use .pdf, .txt, or .md")


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:
    """Split text into overlapping chunks (by character count)."""
    if not text or not text.strip():
        return []
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if not chunk.strip():
            start = end
            continue
        chunks.append(chunk.strip())
        start = end - chunk_overlap
    return chunks


def load_and_chunk_documents(
    paths: List[Path],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Tuple[List[str], List[dict]]:
    """
    Load multiple documents and return chunk texts plus metadata (source path, chunk index).
    Returns (chunks, metadatas).
    """
    all_chunks = []
    all_metadatas = []
    for path in paths:
        path = Path(path)
        try:
            raw = load_document(path)
            chunks = chunk_text(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for i, c in enumerate(chunks):
                all_chunks.append(c)
                all_metadatas.append({"source": str(path), "chunk_index": i})
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e
    return all_chunks, all_metadatas
