"""
Retriever: question -> best matching chunks.

Steps:
 1. Convert the question to a vector (embedding).
 2. Search FAISS for the most similar chunks (semantic search).
 3. Drop chunks whose similarity is below SIMILARITY_THRESHOLD.
 4. Re-rank the rest by mixing the vector score with a BM25 keyword score
    (hybrid search).
 5. Return the top-k chunks with their scores.

If nothing passes the threshold, found=False. The chatbot must then say it has
no information instead of guessing (first part of the input guardrail).
"""

import re  # splits text into words

from rank_bm25 import BM25Okapi  # keyword scoring

from app import config
from app.rag.embedder import embed_texts
from app.rag.vector_store import load_store

VECTOR_WEIGHT = 0.7  # how much the meaning-based score counts
KEYWORD_WEIGHT = 0.3  # how much the exact-word score counts


def _tokens(text: str) -> list[str]:  # "Admission, Process!" -> ["admission", "process"]
    return re.findall(r"\w+", text.lower())


def retrieve(query: str, top_k: int | None = None, source: str | None = None) -> dict:
    """Return {"query", "found", "results": [...]} for one question."""
    top_k = top_k or config.TOP_K
    query = (query or "").strip()
    empty = {"query": query, "found": False, "results": []}
    if not query:
        return empty

    index, chunks = load_store()
    if index is None:  # nothing indexed yet
        return {**empty, "message": "The knowledge base is empty. Run: python -m app.rag.ingest"}

    # 1-2. embed the question and search a larger pool than we need
    pool = min(index.ntotal, max(top_k * 4, 20))
    scores, ids = index.search(embed_texts([query]), pool)

    # 3. keep only chunks that are similar enough (and match the file filter)
    candidates = []  # list of (position in chunks, vector score)
    for score, i in zip(scores[0], ids[0]):
        if i < 0:  # FAISS returns -1 when it has fewer results than asked
            continue
        if source and chunks[i]["source"] != source:  # optional metadata filter
            continue
        if score < config.SIMILARITY_THRESHOLD:
            continue
        candidates.append((int(i), float(score)))
    if not candidates:
        return empty

    # 4. keyword scores over all chunks, then combine
    bm25 = BM25Okapi([_tokens(c["text"]) for c in chunks])
    keyword = bm25.get_scores(_tokens(query))
    best_keyword = max(max(float(keyword[i]), 0.0) for i, _ in candidates)  # for 0-1 scaling

    results = []
    for i, vector_score in candidates:
        keyword_score = max(float(keyword[i]), 0.0) / best_keyword if best_keyword > 0 else 0.0
        final = VECTOR_WEIGHT * vector_score + KEYWORD_WEIGHT * keyword_score
        results.append({
            "source": chunks[i]["source"],
            "page": chunks[i]["page"],
            "text": chunks[i]["text"],
            "vector_score": round(vector_score, 4),
            "keyword_score": round(keyword_score, 4),
            "final_score": round(final, 4),
        })

    # 5. best first, keep top_k
    results.sort(key=lambda r: r["final_score"], reverse=True)
    for rank, item in enumerate(results[:top_k], start=1):
        item["rank"] = rank
    return {"query": query, "found": True, "results": results[:top_k]}