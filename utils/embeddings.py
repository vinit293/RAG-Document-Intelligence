"""
Embeddings Module (embeddings.py)
=================================
Manages text embeddings using SentenceTransformer (all-MiniLM-L6-v2, 384 dimensions).
Supports batched encoding and normalization for cosine similarity search.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Loads and returns the SentenceTransformer embedding model."""
    return SentenceTransformer(model_name)


def generate_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generates normalized dense embeddings for a list of text strings in batches.
    
    Returns:
        numpy.ndarray of shape (len(texts), EMBEDDING_DIMENSION)
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION))

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return embeddings


def embed_query(
    query: str,
    model: SentenceTransformer,
    normalize: bool = True,
) -> List[float]:
    """Generates a normalized embedding vector for a single query string."""
    vector = model.encode(query, normalize_embeddings=normalize)
    return vector.tolist()
