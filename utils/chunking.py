"""
Chunking Module (chunking.py)
=============================
Implements character-based chunking with configurable overlap.
Preserves page-level metadata, extraction method (pypdf vs OCR), and source tracking.
"""

from typing import List, Dict, Any


def create_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Divides page text into overlapping chunks of chunk_size characters with overlap.
    
    Args:
        pages: List of dictionaries containing page text and metadata.
        chunk_size: Maximum characters per chunk (default: 800).
        overlap: Character overlap between consecutive chunks (default: 100).
        
    Returns:
        List of chunk dictionaries with metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    step = chunk_size - overlap
    global_chunk_idx = 0

    for page in pages:
        text = page["text"]
        page_num = page["page"]
        source = page["source"]
        method = page["extraction_method"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page_num,
                    "extraction_method": method,
                    "chunk_index": global_chunk_idx,
                })
                global_chunk_idx += 1

            if end >= len(text):
                break
            start += step

    return chunks
