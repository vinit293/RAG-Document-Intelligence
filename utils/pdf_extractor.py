"""
PDF Extractor Module (pdf_extractor.py)
=======================================
Extracts text from uploaded PDF files page by page.
Performs primary extraction using pypdf, and triggers OCR fallback
via ocr_processor for scanned or image-based pages.
"""

import io
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
import fitz  # PyMuPDF

from utils.ocr_processor import perform_ocr_on_page


def extract_pages_with_fallback(
    pdf_bytes: bytes,
    filename: str,
    ocr_threshold_chars: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts text page-by-page from PDF bytes.
    - Uses pypdf for selectable text.
    - If a page has < ocr_threshold_chars, triggers PyMuPDF + Tesseract OCR fallback.
    
    Returns:
        (pages_list, summary_stats)
    """
    if not pdf_bytes:
        raise ValueError(f"Uploaded file '{filename}' is empty.")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted PDF file '{filename}': {exc}")

    if reader.is_encrypted:
        raise ValueError(
            f"'{filename}' is password-protected or encrypted. "
            "Please remove the password before uploading."
        )

    # Open PyMuPDF document for rendering OCR fallback pages
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        pdf_document = None

    extracted_pages = []
    stats = {
        "filename": filename,
        "total_pages": len(reader.pages),
        "pypdf_pages": 0,
        "ocr_pages": 0,
        "empty_pages": 0,
        "warnings": [],
    }

    try:
        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1
            raw_text = page.extract_text() or ""
            text = " ".join(raw_text.split())
            method = "pypdf"

            # Check if OCR fallback is needed (scanned or image page)
            if len(text.strip()) < ocr_threshold_chars and pdf_document is not None:
                ocr_text, warn = perform_ocr_on_page(pdf_document, page_index)
                if warn:
                    stats["warnings"].append(warn)

                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    method = "OCR"
                    stats["ocr_pages"] += 1
                else:
                    if text.strip():
                        stats["pypdf_pages"] += 1
            else:
                if text.strip():
                    stats["pypdf_pages"] += 1

            if not text.strip():
                stats["empty_pages"] += 1
                continue

            extracted_pages.append({
                "page": page_number,
                "text": text,
                "extraction_method": method,
                "source": filename,
            })

    finally:
        if pdf_document is not None:
            pdf_document.close()

    return extracted_pages, stats
