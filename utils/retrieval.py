"""
Retrieval & Generation Module (retrieval.py)
============================================
Handles semantic vector search, Top-K retrieval with document filtering,
Groq LLM answer generation, and strict citation validation.
"""

import os
import re
from typing import List, Dict, Any, Optional, Set
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sentence_transformers import SentenceTransformer
from groq import Groq

from utils.embeddings import embed_query
from utils.vector_store import DEFAULT_COLLECTION

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def search_documents(
    client: QdrantClient,
    model: SentenceTransformer,
    question: str,
    top_k: int = 3,
    selected_files: Optional[List[str]] = None,
    similarity_threshold: float = 0.0,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[Any]:
    """
    Performs cosine similarity search for the question vector in Qdrant.
    Supports source filtering and minimum similarity threshold.
    """
    query_vector = embed_query(question, model, normalize=True)

    search_args = {
        "collection_name": collection_name,
        "query": query_vector,
        "limit": top_k,
        "with_payload": True,
    }

    if selected_files:
        search_args["query_filter"] = Filter(
            must=[FieldCondition(key="source", match=MatchAny(any=selected_files))]
        )

    results = client.query_points(**search_args).points

    if similarity_threshold > 0.0:
        results = [r for r in results if getattr(r, "score", 0.0) >= similarity_threshold]

    return results


def generate_grounded_answer(
    question: str,
    results: List[Any],
    groq_api_key: Optional[str] = None,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> Dict[str, Any]:
    """
    Constructs context with source citations [S1], [S2] and generates a strictly
    grounded answer via Groq LLM. Validates all cited IDs against retrieved sources.
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing in environment. Please configure your Groq key.")

    if not results:
        return {
            "answer": "Uploaded documents mein relevant information nahi mili (No matching document chunks found).",
            "citations": [],
            "valid_cited_ids": set(),
            "invalid_cited_ids": set(),
        }

    context_parts = []
    valid_sources = []
    valid_ids_map = {}

    for i, result in enumerate(results, start=1):
        payload = result.payload or {}
        text = payload.get("text", "").strip()
        if not text:
            continue

        cid = f"S{i}"
        source = payload.get("source", "Unknown")
        page = payload.get("page", "Unknown")
        method = payload.get("extraction_method", "pypdf")
        score = getattr(result, "score", 0.0)

        context_parts.append(
            f"[{cid}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Extraction: {method}\n"
            f"Text: {text}"
        )

        src_dict = {
            "id": cid,
            "source": source,
            "page": page,
            "extraction_method": method,
            "score": score,
            "text": text,
        }
        valid_sources.append(src_dict)
        valid_ids_map[cid] = src_dict

    if not context_parts:
        return {
            "answer": "Uploaded documents mein relevant information nahi mili.",
            "citations": [],
            "valid_cited_ids": set(),
            "invalid_cited_ids": set(),
        }

    context = "\n\n".join(context_parts)
    groq_client = Groq(api_key=api_key)

    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and strict document QA assistant. "
                        "Answer the question using ONLY the supplied sources. "
                        "If the sources do not contain the answer, state clearly that "
                        "the information was not found in the uploaded documents. "
                        "Do not invent facts or extrapolate beyond the provided text. "
                        "For every factual claim, cite the supporting source using its exact ID, "
                        "such as [S1] or [S2]. "
                        "Only cite a source if its text directly supports that claim. "
                        "At the end, write a line: 'Cited sources:' followed by the IDs you used. "
                        "If no source supports the answer, do not cite any source."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Available sources:\n{context}"
                    ),
                },
            ],
            temperature=0,
        )
        answer = response.choices[0].message.content or ""
    except Exception as exc:
        raise RuntimeError(f"Groq LLM generation error: {exc}")

    # Citation Validation
    cited_matches = set(re.findall(r"\[\s*(S\d+)\s*\]|【\s*(S\d+)\s*】", answer, re.IGNORECASE))
    all_cited_ids = set()
    for m1, m2 in cited_matches:
        cid = (m1 or m2).upper()
        if cid:
            all_cited_ids.add(cid)

    valid_cited_ids = {cid for cid in all_cited_ids if cid in valid_ids_map}
    invalid_cited_ids = {cid for cid in all_cited_ids if cid not in valid_ids_map}

    return {
        "answer": answer,
        "citations": valid_sources,
        "valid_cited_ids": valid_cited_ids,
        "invalid_cited_ids": invalid_cited_ids,
    }
