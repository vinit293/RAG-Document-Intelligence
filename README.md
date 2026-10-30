# 📄 RAG Document Intelligence with OCR & Evaluation

An end-to-end, production-grade **Retrieval-Augmented Generation (RAG)** system built with Python, Streamlit, Qdrant, SentenceTransformers, and Groq LLM (`openai/gpt-oss-120b`).

Supports **text-based, scanned, and mixed PDFs** via an automated **OCR fallback pipeline**, eliminates duplicate uploads using **SHA-256 fingerprinting**, validates **source citations**, and provides a structured **5-metric evaluation suite** using an LLM-as-a-judge approach.

---

## 🏗️ Project Architecture & Directory Structure

```text
RAG-Document-Intelligence/
│
├── app.py                  # Main Streamlit web application
├── evalu.py                # Standalone evaluation runner
├── rag_evalu.py            # 5-metric LLM-as-a-judge evaluation engine
├── requirements.txt        # Production dependency specifications
├── .env.example            # Environment configuration template
├── .gitignore              # Protects secrets & build artifacts
│
├── utils/                  # Modular Single-Responsibility Engine
│   ├── __init__.py         # Package exports
│   ├── pdf_extractor.py    # Primary pypdf extraction with OCR routing
│   ├── ocr_processor.py    # PyMuPDF (fitz) page rendering & Tesseract OCR
│   ├── chunking.py         # Overlapping character chunking (800 chars, 100 overlap)
│   ├── embeddings.py       # all-MiniLM-L6-v2 (384-d) dense embedding generation
│   ├── vector_store.py     # Qdrant client, indexing, deduplication, batching
│   └── retrieval.py        # Semantic vector search & Groq LLM grounded answering
│
└── README.md               # Technical documentation & interview guide
```

---

## ⚡ Key Features & Engineering Highlights

| Feature | Implementation Details |
| :--- | :--- |
| **PDF Extraction** | Primary extraction via `pypdf`. Fast and handles selectable text layers. |
| **OCR Fallback** | Automated routing: if page text `< 30` characters, renders page at 2x resolution via `PyMuPDF` (`fitz`) and recognizes text using `pytesseract`. |
| **Deduplication** | Computes SHA-256 hash of uploaded bytes. Queries Qdrant before indexing to skip duplicate re-processing. Supports optional overwrite re-indexing. |
| **Batching** | Memory-safe batched embedding generation and upserting (batch size: 64) to prevent out-of-memory errors on large multi-page PDFs. |
| **Complete File Listing** | Implements scroll pagination (`offset=next_page_offset`) rather than `limit=100` to ensure all stored documents appear in the filter dropdown. |
| **Vector Storage** | Qdrant Cloud with seamless **Local Embedded Fallback** (`./qdrant_storage`). Offline testing never fails even if cloud clusters sleep. |
| **Strict Citations** | Validates `[S1]`, `[S2]` citation IDs parsed from LLM output against retrieved chunks to catch hallucinated references. |
| **Evaluation Suite** | Evaluates **Context Precision, Context Recall, Faithfulness, Answer Relevancy, and Answer Correctness**. |

---

## 📊 RAG Evaluation (5 Core Metrics)

The system includes both **Benchmark Regression Evaluation** and **Live Real-Time Query Evaluation**:

```
                                  RAG Pipeline
  Question ───────► Retrieval (Qdrant) ───────► LLM Generation (Groq)
                         │                                 │
                         ▼                                 ▼
                 Retrieved Context                  Generated Answer
                         │                                 │
                         └───────────────┬─────────────────┘
                                         ▼
                                5 Evaluation Metrics
       ┌───────────────────────────────┬───────────────────────────────┐
       ▼                               ▼                               ▼
Context Precision               Faithfulness                    Answer Relevancy
(Relevance of Chunks)         (Context Grounding)             (Directness to Prompt)
       │                               │                               │
       └───────────────────────────────┼───────────────────────────────┘
                                       ▼ (With Ground Truth)
                        Context Recall & Correctness
```

### 1. Context Precision (Reference-Free)
$$\text{Context Precision} = \frac{\text{Relevant Retrieved Chunks}}{\text{Total Retrieved Chunks}}$$
- Evaluates whether retrieved passages are actually useful for answering the query.

### 2. Context Recall (Reference-Based)
- Measures how much of the ground truth reference information is present in the retrieved context.

### 3. Faithfulness (Reference-Free)
- Verifies that every factual claim in the generated answer is directly grounded in the retrieved context, penalizing hallucinations.

### 4. Answer Relevancy (Reference-Free)
- Assesses whether the model directly addressed the user's question without digression.

### 5. Answer Correctness (Reference-Based)
- Compares the generated answer factually against the reference ground truth answer.

> **Interview Highlight**: When a user uploads a new custom document, **Context Precision, Faithfulness, and Answer Relevancy** can be evaluated in real-time **without any ground truth**! Context Recall and Correctness are used when ground truth answers are provided.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.11+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (optional for scanned PDFs; text PDFs work out-of-the-box).

### 2. Environment Configuration
Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=https://your-cluster-id.region.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key_here
# Optional: Tesseract executable path if not in system PATH
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 3. Install Dependencies
```powershell
uv add -r requirements.txt
# OR
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```powershell
uv run streamlit run app.py
```
Visit `http://localhost:8501` in your browser.

### 5. Run the Evaluation Suite
```powershell
uv run python evalu.py
```

---

## 🎤 Interview Cheatsheet (STAR Format)

- **Situation**: Large or scanned PDF documents make keyword search ineffective, and standard LLMs hallucinate without grounded source evidence.
- **Task**: Build an interactive, production-ready RAG application that supports scanned PDFs, prevents redundant vector indexing, enforces strict source citations, and provides quantifiable quality evaluation.
- **Action**: Refactored the architecture into modular components (`utils/`). Integrated PyMuPDF + Tesseract OCR fallback for scanned pages, SHA-256 duplicate detection, 384-dimensional dense embeddings (`all-MiniLM-L6-v2`), Qdrant vector storage with cloud/local fallback, Groq-hosted LLM generation (`openai/gpt-oss-120b`), and an LLM-as-a-judge 5-metric evaluation pipeline.
- **Result**: Robust document processing for both text and scanned PDFs, zero duplicate embedding overhead, verified citations, and instant real-time quality scoring across retrieval and generation stages.
