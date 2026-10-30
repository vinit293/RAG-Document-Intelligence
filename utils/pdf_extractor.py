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

from utils.ocr_processor import perform_ocr_on_page, perform_ocr_on_image_bytes


def extract_pages_with_fallback(
    pdf_bytes: bytes,
    filename: str,
    ocr_threshold_chars: int = 30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts text from PDF or Image bytes:
    - If image (PNG, JPG, JPEG): performs direct Tesseract OCR.
    - If PDF: uses pypdf for selectable text with PyMuPDF + Tesseract OCR fallback for scanned pages.
    
    Returns:
        (pages_list, summary_stats)
    """
    if not pdf_bytes:
        raise ValueError(f"Uploaded file '{filename}' is empty.")

    # Direct Image File Handling (PNG, JPG, JPEG, etc.)
    image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
    if filename.lower().endswith(image_exts):
        ocr_text, warn = perform_ocr_on_image_bytes(pdf_bytes)
        stats = {
            "filename": filename,
            "total_pages": 1,
            "pypdf_pages": 0,
            "ocr_pages": 1 if ocr_text else 0,
            "empty_pages": 0 if ocr_text else 1,
            "warnings": [warn] if warn else [],
        }
        if not ocr_text.strip():
            return [], stats

        return [{
            "page": 1,
            "text": ocr_text,
            "extraction_method": "OCR",
            "source": filename,
        }], stats

    # PDF File Handling
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
