"""Semantic check: cosine similarity between an item and candidate passages."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    vecs = _model(model_name).encode(texts, batch_size=batch_size, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between matching rows of ``a`` and ``b``."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.sum(a * b, axis=-1)
