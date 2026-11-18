"""
Vector store: FAISS index + a JSON file with the chunk text and metadata.

Files saved in data/vector_store/:
    index.faiss  - the vectors
    chunks.json  - text, source file and page of every vector (same order)
"""

import json  # saves chunk metadata
from pathlib import Path

import faiss  # vector search library

from app import config
from app.rag.chunker import chunk_pages
from app.rag.embedder import embed_texts
from app.rag.loader import load_document

INDEX_FILE = config.VECTOR_STORE_DIR / "index.faiss"
CHUNKS_FILE = config.VECTOR_STORE_DIR / "chunks.json"


def load_store():
    """Return (index, chunks), or (None, []) if nothing has been indexed yet."""
    if not INDEX_FILE.is_file() or not CHUNKS_FILE.is_file():
        return None, []
    index = faiss.read_index(str(INDEX_FILE))
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return index, chunks


def save_store(index, chunks: list[dict]) -> None:
    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_store() -> None:
    """Delete the whole index (used before a full rebuild)."""
    INDEX_FILE.unlink(missing_ok=True)
    CHUNKS_FILE.unlink(missing_ok=True)


def remove_source(source: str) -> int:
    """Remove every chunk of one file. Returns how many chunks were removed."""
    index, chunks = load_store()
    if index is None:
        return 0
    keep = [i for i, c in enumerate(chunks) if c["source"] != source]  # positions to keep
    removed = len(chunks) - len(keep)
    if removed == 0:
        return 0
    if not keep:  # nothing left
        reset_store()
        return removed
    vectors = index.reconstruct_n(0, index.ntotal)  # read all stored vectors back
    new_index = faiss.IndexFlatIP(index.d)  # inner-product index (= cosine here)
    new_index.add(vectors[keep])  # add only the vectors we keep
    save_store(new_index, [chunks[i] for i in keep])
    return removed


def add_file(path) -> dict:
    """Load -> chunk -> embed -> add one file to the index (replaces older copy)."""
    path = Path(path)
    pages = load_document(path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(f"No text chunks created from {path.name}")
    vectors = embed_texts([c["text"] for c in chunks])  # embed before touching the store

    remove_source(path.name)  # re-uploading the same file replaces it
    index, existing = load_store()
    if index is None:  # first file: create a new index
        index, existing = faiss.IndexFlatIP(vectors.shape[1]), []
    elif index.d != vectors.shape[1]:  # embedding model was changed
        raise ValueError("Embedding model changed. Rebuild with: python -m app.rag.ingest")
    index.add(vectors)
    save_store(index, existing + chunks)
    return {"source": path.name, "pages": len(pages), "chunks": len(chunks)}


def list_sources() -> dict:
    """Return {file name: number of chunks} for everything in the index."""
    _, chunks = load_store()
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
    return counts