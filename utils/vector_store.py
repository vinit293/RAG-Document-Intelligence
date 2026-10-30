"""
Vector Store Module (vector_store.py)
=====================================
Manages Qdrant vector database operations:
- Dual-mode support (Cloud Qdrant with seamless Local Embedded fallback).
- Multi-instance lock conflict handling (process-scoped storage or memory fallback).
- Collection management and payload indexing (source, file_hash).
- Duplicate prevention via SHA-256 fingerprint matching.
- Paginated listing of stored documents.
- Batched vector upserting with deterministic UUIDs.
"""

import os
import uuid
from typing import List, Dict, Any, Set, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

from utils.embeddings import generate_embeddings, EMBEDDING_DIMENSION

DEFAULT_COLLECTION = "rag_documents"
DEFAULT_LOCAL_PATH = "./qdrant_storage"


def _create_local_client(local_path: str = DEFAULT_LOCAL_PATH) -> Tuple[QdrantClient, str]:
    """Attempts local disk storage, falling back to process-scoped or memory client if locked."""
    try:
        client = QdrantClient(path=local_path)
        return client, "Local Embedded (Offline)"
    except Exception as exc:
        err_msg = str(exc).lower()
        if "already accessed" in err_msg or "locked" in err_msg:
            try:
                pid_path = f"{local_path}_{os.getpid()}"
                client = QdrantClient(path=pid_path)
                return client, f"Local Embedded ({pid_path})"
            except Exception:
                client = QdrantClient(location=":memory:")
                return client, "Local Embedded (:memory: fallback)"
        raise exc


def init_vector_client(
    storage_mode: str = "Auto",
    local_path: str = DEFAULT_LOCAL_PATH,
) -> Tuple[QdrantClient, str]:
    """
    Initializes Qdrant client based on requested mode:
    - 'Local Embedded (Offline)': Runs embedded on disk (local_path).
    - 'Cloud Qdrant': Connects strictly to cloud cluster.
    - 'Auto': Tries Cloud cluster first; if unreachable (e.g. WinError 10054),
              seamlessly falls back to Local Embedded without failing.
    
    Returns:
        (client, mode_description_string)
    """
    if storage_mode == "Local Embedded (Offline)":
        return _create_local_client(local_path)

    def _get_val(k: str) -> Optional[str]:
        v = os.getenv(k)
        if not v:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and k in st.secrets:
                    v = st.secrets[k]
            except Exception:
                pass
        return v

    url = _get_val("QDRANT_URL")
    api_key = _get_val("QDRANT_API_KEY")

    if storage_mode == "Cloud Qdrant":
        if not url or not api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing in environment or Streamlit Secrets.")
        client = QdrantClient(url=url, api_key=api_key, timeout=10)
        client.get_collections()
        return client, "Cloud Qdrant"

    # Default: Auto with fallback
    if url and api_key:
        try:
            client = QdrantClient(url=url, api_key=api_key, timeout=5)
            client.get_collections()
            return client, "Cloud Qdrant"
        except Exception as exc:
            client, mode_lbl = _create_local_client(local_path)
            return client, f"{mode_lbl} (Cloud error: {type(exc).__name__})"

    return _create_local_client(local_path)


def ensure_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION,
    dimension: int = EMBEDDING_DIMENSION,
):
    """Creates the collection and keyword payload indexes if they do not exist."""
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
    except Exception as exc:
        raise ConnectionError(f"Failed to verify or create collection '{collection_name}': {exc}")

    # Safe payload indexing for filtering & duplicate detection
    for field in ["source", "file_hash"]:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception:
            pass


def is_file_already_indexed(
    client: QdrantClient,
    file_hash: str,
    collection_name: str = DEFAULT_COLLECTION,
) -> bool:
    """Checks whether points with the given SHA-256 hash exist in Qdrant."""
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="file_hash", match=MatchValue(value=file_hash))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(points) > 0
    except Exception:
        return False


def delete_file_by_hash(
    client: QdrantClient,
    file_hash: str,
    collection_name: str = DEFAULT_COLLECTION,
):
    """Deletes existing points associated with a file_hash for clean re-indexing."""
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="file_hash", match=MatchValue(value=file_hash))]
            ),
            wait=True,
        )
    except Exception:
        pass


def store_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: List[Dict[str, Any]],
    file_hash: str,
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = 64,
) -> int:
    """
    Encodes chunk texts and upserts PointStruct records to Qdrant in batches.
    Uses deterministic UUID5 IDs constructed from file_hash and global chunk index.
    """
    if not chunks:
        return 0

    total_stored = 0

    for start_idx in range(0, len(chunks), batch_size):
        batch = chunks[start_idx : start_idx + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = generate_embeddings(texts, model, batch_size=batch_size, normalize=True)

        points = []
        for i, (chunk, emb) in enumerate(zip(batch, embeddings)):
            idx = start_idx + i
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_hash}:{idx}"))

            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload={
                        **chunk,
                        "file_hash": file_hash,
                        "chunk_index": idx,
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)
        total_stored += len(points)

    return total_stored


def get_all_indexed_files(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[str]:
    """Retrieves all unique document names using scroll pagination (no 100-limit cutoff)."""
    unique_files: Set[str] = set()
    offset = None

    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for pt in points:
                if pt.payload and "source" in pt.payload:
                    unique_files.add(pt.payload["source"])

            if next_offset is None or not points:
                break
            offset = next_offset
        except Exception:
            break

    return sorted(unique_files)
