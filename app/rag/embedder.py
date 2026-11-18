"""
Embeddings: turn text into vectors (lists of numbers) with a Hugging Face model.

Vectors are normalized, so the inner product of two vectors equals their
cosine similarity (a score between about 0 and 1).
"""

import numpy as np  # arrays of numbers
from sentence_transformers import SentenceTransformer  # Hugging Face embedding models

from app import config  # EMBEDDING_MODEL

_model = None  # the model is loaded once and reused


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:  # first call: download (once) and load the model
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return a float32 array with one normalized vector per text."""
    vectors = get_model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype="float32")  # FAISS needs float32