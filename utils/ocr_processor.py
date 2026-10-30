"""
OCR Processor Module (ocr_processor.py)
=======================================
Handles Tesseract OCR configuration, high-resolution rendering of PDF pages
using PyMuPDF (fitz), and Optical Character Recognition with pytesseract.
"""

import io
import os
from typing import Optional, Tuple
import fitz  # PyMuPDF
from PIL import Image
import pytesseract


def configure_tesseract() -> Optional[str]:
    """Auto-detect Tesseract OCR executable in standard Windows & custom paths."""
    custom_path = os.getenv("TESSERACT_PATH")
    candidates = [
        custom_path,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
    return None


TESSERACT_PATH = configure_tesseract()


def is_ocr_available() -> bool:
    """Checks if Tesseract OCR is configured and usable."""
    return TESSERACT_PATH is not None and os.path.exists(TESSERACT_PATH)


def perform_ocr_on_page(
    pdf_document: fitz.Document,
    page_index: int,
    zoom_factor: float = 2.0,
    language: str = "eng",
) -> Tuple[str, Optional[str]]:
    """
    Renders a specific PDF page as a high-resolution image and extracts text using Tesseract OCR.
    
    Returns:
        (ocr_text, warning_or_error_message)
    """
    try:
        pdf_page = pdf_document.load_page(page_index)
        # Render at 2x resolution (matrix) for clear character recognition
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        pix = pdf_page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        
        ocr_text = pytesseract.image_to_string(image, lang=language)
        cleaned_text = " ".join(ocr_text.split())
        return cleaned_text, None

    except pytesseract.TesseractNotFoundError:
        return "", (
            f"Page {page_index + 1}: Tesseract OCR executable not found. "
            "Please install Tesseract-OCR or configure TESSERACT_PATH in .env."
        )
    except Exception as exc:
        return "", f"Page {page_index + 1} OCR processing error: {exc}"
