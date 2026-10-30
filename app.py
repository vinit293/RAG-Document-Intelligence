"""
RAG Document Intelligence (app.py)
==================================
Streamlit Application for Multimodal OCR-Enabled Document Question Answering.
Modular architecture importing from utils package.
"""

import os
import hashlib
from typing import Tuple, Any, List, Dict, Optional
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from utils.ocr_processor import is_ocr_available, TESSERACT_PATH
from utils.pdf_extractor import extract_pages_with_fallback
from utils.chunking import create_chunks
from utils.embeddings import load_embedding_model
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
from rag_evalu import evaluate_live_query

# ------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ------------------------------------------------------------
load_dotenv(find_dotenv(usecwd=True))

st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📄",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { color: #6c757d; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .badge-ocr { background-color: #ffc107; color: #212529; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .badge-pypdf { background-color: #0d6efd; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">📄 RAG Document Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">OCR-Enabled Multimodal PDF Question Answering with Semantic Search & Grounded Citations</div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 2. CACHED RESOURCE INITIALIZERS
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_cached_model():
    return load_embedding_model()


@st.cache_resource(show_spinner="Initializing vector database...")
def get_cached_client(mode: str) -> Tuple[Any, str]:
    return init_vector_client(storage_mode=mode)


# ------------------------------------------------------------
# 3. SIDEBAR CONFIGURATION & HEALTH
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status & Settings")

    # OCR Engine Status
    if is_ocr_available():
        st.success(f"OCR Engine: Active\n\n`{TESSERACT_PATH}`")
    else:
        st.warning("OCR Engine: Tesseract not detected in standard paths. Text-based PDFs work via pypdf; scanned PDFs require Tesseract.")

    # Storage Mode Selector
    storage_choice = st.selectbox(
        "Vector Database Storage",
        options=["Auto (Cloud with Local Fallback)", "Local Embedded (Offline)", "Cloud Qdrant"],
        index=0,
        help="Local Embedded runs entirely on your disk (offline). Auto tries Cloud first and falls back to Local if unreachable.",
    )

    try:
        client, mode_label = get_cached_client(storage_choice)
        ensure_collection(client)
        if "Fallback" in mode_label or "Local" in mode_label:
            st.info(f"Qdrant: {mode_label}\n\nRunning locally in `./qdrant_storage`.")
        else:
            st.success(f"Qdrant: {mode_label}")
    except Exception as e:
        client = None
        st.error(f"Qdrant Connection Error: {e}")

    # Model Status
    try:
        model = get_cached_model()
        st.success("Embedding Model: all-MiniLM-L6-v2")
    except Exception as e:
        model = None
        st.error(f"Embedding Model Error: {e}")

    st.divider()
    st.subheader("Retrieval Parameters")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=3)
    similarity_thresh = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filter chunks below this cosine similarity score.",
    )

    st.divider()
    chunk_size = st.number_input("Chunk Size (characters)", value=800, min_value=100, max_value=4000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (characters)", value=100, min_value=0, max_value=500, step=50)

    overwrite_existing = st.checkbox(
        "Re-index existing files (overwrite duplicates)",
        value=False,
        help="If checked, duplicate files with identical SHA-256 hashes will be replaced in Qdrant.",
    )


# ------------------------------------------------------------
# 4. SECTION 1: INGESTION & OCR PROCESSING
# ------------------------------------------------------------
st.subheader("1. Ingest Documents")
uploaded_files = st.file_uploader(
    "Choose PDF or Image files (supports text PDFs, scanned PDFs, PNG, JPG, JPEG)",
    type=["pdf", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) selected for processing.")

    if st.button("Add Files to Knowledge Base", type="primary"):
        if not client or not model:
            st.error("Cannot proceed: Qdrant client or embedding model is unavailable. Check sidebar.")
        else:
            total_stored = 0
            skipped_duplicates = 0

            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name

                with st.status(f"Processing '{file_name}'...", expanded=True) as status_box:
                    try:
                        file_bytes = uploaded_file.getvalue()
                        file_hash = hashlib.sha256(file_bytes).hexdigest()

                        # Step 1: Duplicate Prevention Check
                        if is_file_already_indexed(client, file_hash) and not overwrite_existing:
                            st.warning(
                                f"Duplicate Detected: '{file_name}' (SHA-256: `{file_hash[:12]}...`) "
                                "is already indexed in Qdrant. Skipping to prevent redundant chunks."
                            )
                            skipped_duplicates += 1
                            status_box.update(label=f"Skipped duplicate: {file_name}", state="complete")
                            continue
                        elif is_file_already_indexed(client, file_hash) and overwrite_existing:
                            st.info(f"Re-indexing enabled: Removing existing points for '{file_name}'...")
                            delete_file_by_hash(client, file_hash)

                        # Step 2: Extraction with OCR Fallback
                        st.write("Extracting pages (pypdf + PyMuPDF/Tesseract OCR fallback)...")
                        pages, stats = extract_pages_with_fallback(file_bytes, file_name, ocr_threshold_chars=30)

                        if stats["warnings"]:
                            for warn in stats["warnings"]:
                                st.caption(f"⚠️ {warn}")

                        if not pages:
                            st.warning(f"No extractable text found in '{file_name}'.")
                            status_box.update(label=f"No text extracted: {file_name}", state="error")
                            continue

                        # Step 3: Overlapping Chunking
                        chunks = create_chunks(pages, chunk_size=chunk_size, overlap=chunk_overlap)
                        st.write(
                            f"Extracted {len(chunks)} chunks across {stats['total_pages']} page(s) "
                            f"(Text: {stats['pypdf_pages']}, OCR: {stats['ocr_pages']})"
                        )

                        # Step 4: Batched Embeddings & Qdrant Upsert
                        st.write("Generating dense embeddings and upserting into Qdrant...")
                        count = store_chunks(client, model, chunks, file_hash, batch_size=64)
                        total_stored += count

                        status_box.update(
                            label=f"Successfully indexed '{file_name}' ({count} chunks stored)",
                            state="complete",
                        )

                        # Preview Expander
                        with st.expander(f"Preview Chunks: {file_name} (First 5 chunks)"):
                            for i, chunk in enumerate(chunks[:5], start=1):
                                badge_class = "badge-ocr" if chunk["extraction_method"] == "OCR" else "badge-pypdf"
                                st.markdown(
                                    f"**Chunk {i} — Page {chunk['page']}** "
                                    f"<span class='{badge_class}'>{chunk['extraction_method']}</span>",
                                    unsafe_allow_html=True,
                                )
                                st.write(chunk["text"])

                    except ValueError as val_err:
                        st.error(f"Validation Error: {val_err}")
                        status_box.update(label=f"Failed: {file_name}", state="error")
                    except Exception as exc:
                        st.error(f"Processing error: {exc}")
                        status_box.update(label=f"Failed: {file_name}", state="error")

            st.success(
                f"Ingestion finished! Total chunks stored: {total_stored} | "
                f"Duplicates skipped: {skipped_duplicates}"
            )


# ------------------------------------------------------------
# 5. SECTION 2: QUESTION ANSWERING & CITATION GROUNDING
# ------------------------------------------------------------
st.divider()
st.subheader("2. Ask Questions About Your PDFs")

if client:
    available_files = get_all_indexed_files(client)
else:
    available_files = []

col_filter, col_query = st.columns([1, 2])

with col_filter:
    if available_files:
        selected_files = st.multiselect(
            "Filter by Document(s):",
            options=available_files,
            default=available_files,
            help="Select one or more documents. Leave all selected to query entire database.",
        )
    else:
        selected_files = []
        st.info("No documents are currently indexed in Qdrant. Upload a PDF above to begin.")

with col_query:
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g. What are the key skills or policies in the uploaded document?",
    )
    ask_button = st.button("Search & Answer", type="primary", disabled=not available_files)

if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not client or not model:
        st.error("System is not ready: Qdrant client or embedding model is not initialized.")
    else:
        try:
            with st.spinner("Retrieving relevant passages and generating grounded answer..."):
                results = search_documents(
                    client=client,
                    model=model,
                    question=question,
                    top_k=top_k,
                    selected_files=selected_files,
                    similarity_threshold=similarity_thresh,
                )

                output = generate_grounded_answer(question, results)

            # Display Answer
            st.markdown("### Answer")
            st.write(output["answer"])

            # Citations Display & Validation
            st.markdown("### Cited Sources")
            citations = output.get("citations", [])
            valid_cited_ids = output.get("valid_cited_ids", set())
            invalid_cited_ids = output.get("invalid_cited_ids", set())

            if citations:
                displayed_keys = set()
                found_valid_citations = False

                for src in citations:
                    if src["id"] in valid_cited_ids:
                        found_valid_citations = True
                        key = (src["source"], src["page"])
                        if key not in displayed_keys:
                            displayed_keys.add(key)
                            badge = "badge-ocr" if src["extraction_method"] == "OCR" else "badge-pypdf"
                            st.markdown(
                                f"- **[{src['id']}]** `{src['source']}` — Page {src['page']} "
                                f"<span class='{badge}'>{src['extraction_method']}</span> "
                                f"(Similarity: {src['score']:.3f})",
                                unsafe_allow_html=True,
                            )

                if not found_valid_citations:
                    ans_lower = output["answer"].lower()
                    if (
                        "not found" in ans_lower
                        or "relevant information nahi mili" in ans_lower
                        or "information was not found" in ans_lower
                    ):
                        st.info("The requested information was not found in the selected documents.")
                    else:
                        st.info("No specific source IDs were cited in the answer.")

                # Flag hallucinated citation IDs
                if invalid_cited_ids:
                    st.warning(
                        f"Citation Validation Alert: The model cited ID(s) {sorted(list(invalid_cited_ids))} "
                        "which were not in the retrieved passages."
                    )
            else:
                st.info("No supporting sources were retrieved for this query.")

            # Retrieved Chunks Inspection Expander
            st.markdown("### Retrieved Chunks Inspection")
            if not results:
                st.info("No matching chunks met the search criteria.")
            else:
                for idx, res in enumerate(results, start=1):
                    p = res.payload or {}
                    method = p.get("extraction_method", "pypdf")
                    badge = "badge-ocr" if method == "OCR" else "badge-pypdf"

                    with st.expander(
                        f"Chunk [S{idx}] — {p.get('source', 'Unknown')} (Page {p.get('page', '?')}) | Score: {res.score:.3f}"
                    ):
                        st.markdown(
                            f"**Method:** <span class='{badge}'>{method}</span> | **Chunk Index:** {p.get('chunk_index', '?')}",
                            unsafe_allow_html=True,
                        )
                        st.text(p.get("text", ""))

            # ------------------------------------------------------------
            # LIVE 5-METRIC RAG EVALUATION (FOR ANY CUSTOM PDF)
            # ------------------------------------------------------------
            st.divider()
            with st.expander("📊 Live RAG Evaluation (Score This Query on Any PDF)"):
                st.markdown(
                    "Evaluate retrieval quality and answer grounding directly for **this specific PDF and query** "
                    "using the LLM-as-a-judge system."
                )
                custom_gt = st.text_input(
                    "Optional Ground Truth Answer (Provide to score Context Recall & Answer Correctness):",
                    placeholder="e.g. The exact policy or fact from the document",
                    key=f"gt_{abs(hash(question))}",
                )

                if st.button("Run Real-Time Evaluation", key=f"eval_btn_{abs(hash(question))}"):
                    try:
                        from groq import Groq
                        key_val = os.getenv("GROQ_API_KEY")
                        if not key_val and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                            key_val = st.secrets["GROQ_API_KEY"]
                        groq_client_eval = Groq(api_key=key_val)

                        with st.spinner("LLM Judge is assessing Context Precision, Faithfulness & Relevancy..."):
                            eval_res = evaluate_live_query(
                                groq_client=groq_client_eval,
                                question=question,
                                retrieved_chunks=results,
                                answer=output["answer"],
                                ground_truth=custom_gt.strip() if custom_gt else None,
                            )

                        st.markdown("#### Evaluation Results")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Context Precision", f"{eval_res['context_precision']:.2f}")
                            st.caption("Proportion of retrieved chunks relevant to query.")
                        with c2:
                            st.metric("Faithfulness", f"{eval_res['faithfulness']:.2f}")
                            st.caption(eval_res.get("faithfulness_reason", "Answer supported by context."))
                        with c3:
                            st.metric("Answer Relevancy", f"{eval_res['answer_relevancy']:.2f}")
                            st.caption(eval_res.get("relevancy_reason", "Directness of response."))

                        if "context_recall" in eval_res and "answer_correctness" in eval_res:
                            c4, c5 = st.columns(2)
                            with c4:
                                st.metric("Context Recall", f"{eval_res['context_recall']:.2f}")
                                st.caption(eval_res.get("recall_reason", "Information coverage."))
                            with c5:
                                st.metric("Answer Correctness", f"{eval_res['answer_correctness']:.2f}")
                                st.caption(eval_res.get("correctness_reason", "Ground truth accuracy."))

                    except Exception as eval_err:
                        st.error(f"Evaluation failed: {eval_err}")

        except Exception as error:
            st.error(f"Question answering failed: {error}")
