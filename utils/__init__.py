"""
Utils Package for RAG Document Intelligence
===========================================
Modular architecture components:
- ocr_processor: PyMuPDF page rendering & Tesseract OCR
- pdf_extractor: Primary pypdf extraction with OCR fallback
- chunking: Overlapping character chunking with metadata
- embeddings: SentenceTransformer vector generation
- vector_store: Qdrant client, indexing, deduplication, batching
- retrieval: Semantic search & grounded generation with citations
"""

from utils.ocr_processor import (
    configure_tesseract,
    is_ocr_available,
    perform_ocr_on_page,
    perform_ocr_on_image_bytes,
)
from utils.pdf_extractor import extract_pages_with_fallback
from utils.chunking import create_chunks
from utils.embeddings import load_embedding_model, generate_embeddings, embed_query
from utils.vector_store import (
    init_vector_client,
    ensure_collection,
    is_file_already_indexed,
    delete_file_by_hash,
    store_chunks,
    get_all_indexed_files,
    DEFAULT_COLLECTION,
)
from utils.retrieval import search_documents, generate_grounded_answer

__all__ = [
    "configure_tesseract",
    "is_ocr_available",
    "perform_ocr_on_page",
    "extract_pages_with_fallback",
    "create_chunks",
    "load_embedding_model",
    "generate_embeddings",
    "embed_query",
    "init_vector_client",
    "ensure_collection",
    "is_file_already_indexed",
    "delete_file_by_hash",
    "store_chunks",
    "get_all_indexed_files",
    "DEFAULT_COLLECTION",
    "search_documents",
    "generate_grounded_answer",
]
