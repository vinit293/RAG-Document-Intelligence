"""
RAG Evaluation Module (rag_evalu.py)
===================================
Evaluates RAG retrieval and generation performance using an LLM-as-a-judge approach across 5 core metrics:
1. Context Precision  — Proportion of retrieved chunks that are relevant to the query.
2. Context Recall     — Coverage of ground truth information within the retrieved context.
3. Faithfulness       — Grounding of generated answer claims in the retrieved context.
4. Answer Relevancy   — Alignment and directness of the generated answer with respect to the user question.
5. Answer Correctness — Factual alignment of the generated answer against the reference ground truth.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# 1. ENVIRONMENT & CONFIGURATION
# ============================================================
load_dotenv(find_dotenv(usecwd=True))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EVAL_COLLECTION_NAME = "rag_eval"
EMBEDDING_SIZE = 384
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_JUDGE_MODEL = "openai/gpt-oss-120b"


# ============================================================
# 2. CLIENT & RESOURCE INITIALIZERS
# ============================================================
def get_qdrant_client() -> QdrantClient:
    """Creates a validated Qdrant client with automatic local fallback."""
    if QDRANT_URL and QDRANT_API_KEY:
        try:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5)
            client.get_collections()
            return client
        except Exception:
            print("Note: Qdrant Cloud connection failed ([WinError 10054] / unreachable).")
            print("Falling back to local embedded Qdrant storage (./qdrant_storage).")

    return QdrantClient(path="./qdrant_storage")


def get_groq_client() -> Groq:
    """Creates a validated Groq client."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing from environment. Please check your .env file.")
    return Groq(api_key=GROQ_API_KEY)


def get_embedding_model() -> SentenceTransformer:
    """Loads the sentence embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# 3. KNOWLEDGE INGESTION FOR EVALUATION BENCHMARK
# ============================================================
def setup_eval_collection(client: QdrantClient, model: SentenceTransformer, knowledge_path: str = "knowledge.json"):
    """
    Ensures the eval collection exists in Qdrant and loads knowledge.json
    only if not already indexed (idempotent setup).
    """
    if not client.collection_exists(EVAL_COLLECTION_NAME):
        client.create_collection(
            collection_name=EVAL_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection: '{EVAL_COLLECTION_NAME}'")

    # Safe payload index creation
    try:
        client.create_payload_index(
            collection_name=EVAL_COLLECTION_NAME,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
    except Exception:
        pass

    # Check if points already exist
    collection_info = client.get_collection(collection_name=EVAL_COLLECTION_NAME)
    if collection_info.points_count and collection_info.points_count > 0:
        print(f"Collection '{EVAL_COLLECTION_NAME}' already contains {collection_info.points_count} points. Reusing.")
        return

    # Load knowledge documents
    if not os.path.exists(knowledge_path):
        # Fallback to local directory if relative path differs
        local_candidate = os.path.join(os.path.dirname(__file__), knowledge_path)
        if os.path.exists(local_candidate):
            knowledge_path = local_candidate
        else:
            raise FileNotFoundError(f"Knowledge file '{knowledge_path}' not found.")

    with open(knowledge_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Indexing {len(documents)} knowledge items into '{EVAL_COLLECTION_NAME}'...")
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, normalize_embeddings=True)

    points = [
        PointStruct(
            id=i + 1,
            vector=embeddings[i].tolist(),
            payload=documents[i],
        )
        for i in range(len(documents))
    ]

    client.upsert(collection_name=EVAL_COLLECTION_NAME, points=points, wait=True)
    print(f"Uploaded {len(points)} benchmark documents to Qdrant successfully.")


# ============================================================
# 4. RETRIEVAL & ANSWER GENERATION PIPELINE
# ============================================================
def search_eval_collection(client: QdrantClient, model: SentenceTransformer, query: str, top_k: int = 3) -> List[Any]:
    """Retrieves top_k relevant points from the evaluation collection."""
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=EVAL_COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return results


def ask_rag_pipeline(groq_client: Groq, question: str, context: str) -> str:
    """Generates an answer from Groq LLM strictly grounded in the retrieved context."""
    prompt = f"""You are a helpful and strict QA assistant. Answer the question using ONLY the information provided in the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say exactly:
"I don't know based on the provided information."
Do not invent any facts."""

    response = groq_client.chat.completions.create(
        model=GROQ_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


# ============================================================
# 5. ROBUST LLM JUDGE WITH JSON PARSING SAFEGUARDS
# ============================================================
def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Safely parses JSON even if the LLM wraps it in markdown backticks
    or includes additional introductory/closing comments.
    """
    cleaned = text.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback: find first JSON object matching {...}
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    return {"score": 0.0, "reason": "Failed to parse JSON response from LLM judge."}


def llm_judge(groq_client: Groq, prompt: str) -> Dict[str, Any]:
    """Invokes Groq LLM in JSON mode with temperature=0 for deterministic evaluation."""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        return safe_parse_json(content)
    except Exception as exc:
        return {"score": 0.0, "reason": f"Judge invocation failed: {exc}"}


# ============================================================
# 6. FIVE RAG EVALUATION METRICS
# ============================================================

# Metric 1: Context Precision
def evaluate_context_precision(groq_client: Groq, question: str, retrieved_docs: List[Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Context Precision = (Relevant Retrieved Chunks) / (Total Retrieved Chunks)
    The LLM judge evaluates each chunk independently for relevance to the question.
    """
    if not retrieved_docs:
        return 0.0, []

    relevant_count = 0
    judgments = []

    for i, doc in enumerate(retrieved_docs, start=1):
        chunk_text = doc.payload.get("text", "")
        prompt = f"""You are evaluating the retrieval quality of a RAG system.

Question:
{question}

Retrieved Chunk:
{chunk_text}

Is this chunk relevant to answering the question?
Return ONLY JSON:
{{
    "relevant": true,
    "reason": "short explanation"
}}

Return true if the chunk contains information useful for answering the question.
Return false if it is unrelated."""

        result = llm_judge(groq_client, prompt)
        is_rel = bool(result.get("relevant", False))
        if is_rel:
            relevant_count += 1
        judgments.append({"chunk_index": i, "relevant": is_rel, "reason": result.get("reason", "")})

    precision_score = relevant_count / len(retrieved_docs)
    return precision_score, judgments


# Metric 2: Context Recall
def evaluate_context_recall(groq_client: Groq, question: str, context: str, ground_truth: str) -> Dict[str, Any]:
    """
    Context Recall measures how much of the required ground-truth information is captured in the context.
    Rubric: 1.0 (Full), 0.7 (Most), 0.5 (Partial), 0.0 (Absent).
    """
    prompt = f"""You are evaluating the retrieval quality of a RAG system.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context}

Does the retrieved context contain enough information to produce the ground truth answer?
Return ONLY JSON:
{{
    "score": 0.0,
    "reason": "short explanation"
}}

Scoring Rubric:
1.0 = All important information needed for the answer is present.
0.7 = Most important information is present, but minor details are missing.
0.5 = Some important information is present, but significant parts are missing.
0.0 = The required information is completely absent."""

    res = llm_judge(groq_client, prompt)
    try:
        res["score"] = float(res.get("score", 0.0))
    except (ValueError, TypeError):
        res["score"] = 0.0
    return res


# Metric 3: Faithfulness
def evaluate_faithfulness(groq_client: Groq, question: str, context: str, answer: str) -> Dict[str, Any]:
    """
    Faithfulness measures whether the claims in the generated answer are grounded in the retrieved context.
    Rubric: 1.0 (Fully grounded), 0.7 (Mostly supported), 0.5 (Partially supported), 0.0 (Unsupported/hallucinated).
    """
    prompt = f"""You are evaluating the faithfulness of a RAG system's response.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Determine whether the claims in the generated answer are directly supported by the retrieved context.
Return ONLY JSON:
{{
    "score": 0.0,
    "reason": "short explanation"
}}

Scoring Rubric:
1.0 = All claims are supported by the retrieved context.
0.7 = Mostly supported with minor unsupported statements.
0.5 = Some claims are supported, but others are unsupported.
0.0 = Completely unsupported or contradictory to the context."""

    res = llm_judge(groq_client, prompt)
    try:
        res["score"] = float(res.get("score", 0.0))
    except (ValueError, TypeError):
        res["score"] = 0.0
    return res


# Metric 4: Answer Relevancy
def evaluate_answer_relevancy(groq_client: Groq, question: str, answer: str) -> Dict[str, Any]:
    """
    Answer Relevancy evaluates whether the answer directly addresses the user query without being off-topic.
    Rubric: 1.0 (Direct), 0.7 (Mostly relevant), 0.5 (Partially), 0.0 (Off-topic).
    """
    prompt = f"""You are evaluating the answer relevancy of a RAG system.

Question:
{question}

Generated Answer:
{answer}

Does the generated answer directly address the question asked?
Return ONLY JSON:
{{
    "score": 0.0,
    "reason": "short explanation"
}}

Scoring Rubric:
1.0 = Directly and clearly answers the question.
0.7 = Mostly answers the question, with minor digressions.
0.5 = Partially answers the question or answers vaguely.
0.0 = Completely off-topic or fails to address the question."""

    res = llm_judge(groq_client, prompt)
    try:
        res["score"] = float(res.get("score", 0.0))
    except (ValueError, TypeError):
        res["score"] = 0.0
    return res


# Metric 5: Answer Correctness
def evaluate_answer_correctness(groq_client: Groq, answer: str, ground_truth: str) -> Dict[str, Any]:
    """
    Answer Correctness measures factual alignment against the ground truth reference answer.
    Rubric: 1.0 (Completely correct), 0.7 (Mostly correct), 0.5 (Partially correct), 0.0 (Incorrect).
    """
    prompt = f"""You are evaluating the answer correctness of a RAG system.

Generated Answer:
{answer}

Ground Truth Answer:
{ground_truth}

Determine whether the generated answer is factually correct compared with the ground truth.
Return ONLY JSON:
{{
    "score": 0.0,
    "reason": "short explanation"
}}

Scoring Rubric:
1.0 = Completely factually correct compared to the ground truth.
0.7 = Mostly correct with minor omissions or slight differences.
0.5 = Partially correct, but misses major factual points.
0.0 = Incorrect, contradictory, or completely wrong."""

    res = llm_judge(groq_client, prompt)
    try:
        res["score"] = float(res.get("score", 0.0))
    except (ValueError, TypeError):
        res["score"] = 0.0
    return res


# ============================================================
# 7. DYNAMIC / LIVE QUERY EVALUATION (FOR ANY NEW PDF)
# ============================================================
def evaluate_live_query(
    groq_client: Groq,
    question: str,
    retrieved_chunks: List[Any],
    answer: str,
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates ANY arbitrary query on ANY new uploaded PDF:
    - Reference-Free (No Ground Truth needed): Context Precision, Faithfulness, Answer Relevancy
    - Reference-Based (If Ground Truth provided): Context Recall, Answer Correctness
    """
    context = "\n\n".join([
        (r.payload.get("text", "") if hasattr(r, "payload") and r.payload else str(r))
        for r in retrieved_chunks
    ])

    # 1. Context Precision
    precision, precision_details = evaluate_context_precision(groq_client, question, retrieved_chunks)

    # 2. Faithfulness (Grounding in retrieved context)
    faith_res = evaluate_faithfulness(groq_client, question, context, answer)

    # 3. Answer Relevancy (Directness to question)
    rel_res = evaluate_answer_relevancy(groq_client, question, answer)

    results = {
        "context_precision": round(precision, 3),
        "faithfulness": round(faith_res["score"], 3),
        "faithfulness_reason": faith_res.get("reason", ""),
        "answer_relevancy": round(rel_res["score"], 3),
        "relevancy_reason": rel_res.get("reason", ""),
        "precision_details": precision_details,
    }

    # If ground truth is provided, evaluate recall and correctness as well
    if ground_truth and ground_truth.strip():
        rec_res = evaluate_context_recall(groq_client, question, context, ground_truth)
        corr_res = evaluate_answer_correctness(groq_client, answer, ground_truth)
        results["context_recall"] = round(rec_res["score"], 3)
        results["recall_reason"] = rec_res.get("reason", "")
        results["answer_correctness"] = round(corr_res["score"], 3)
        results["correctness_reason"] = corr_res.get("reason", "")

    return results


# ============================================================
# 8. GOLDEN DATASET (HR & POLICIES BENCHMARK)
# ============================================================
GOLDEN_DATASET = [
    {
        "id": 1,
        "question": "How many vacation days do I get?",
        "ground_truth": "Employees in India receive 24 days of paid leave per year and 12 public holidays.",
        "expected_information": "Employees receive 24 days of paid leave per year in India.",
    },
    {
        "id": 2,
        "question": "What is the work from home policy?",
        "ground_truth": "Akamai follows a hybrid work model where employees work from office three days every week on Tuesday, Wednesday and Thursday.",
        "expected_information": "Hybrid model with 3 office days on Tuesday, Wednesday, and Thursday.",
    },
    {
        "id": 3,
        "question": "How much internet reimbursement do employees receive?",
        "ground_truth": "Employees can claim Rs 2000 per month for home internet reimbursement using ExpenseIt before the 5th of every month.",
        "expected_information": "Rs 2000 per month home internet reimbursement.",
    },
    {
        "id": 4,
        "question": "When do promotions happen?",
        "ground_truth": "Promotion cycles happen twice every year during March and September, usually after spending at least two years at the current level.",
        "expected_information": "Promotion cycles happen twice a year in March and September.",
    },
    {
        "id": 5,
        "question": "What is the gym benefit?",
        "ground_truth": "Employees receive Rs 3000 every month as a wellness benefit for gym memberships or fitness app subscriptions.",
        "expected_information": "Rs 3000 monthly wellness benefit for gym memberships or fitness apps.",
    },
]


# ============================================================
# 8. COMPLETE EVALUATION EXECUTION & REPORTING
# ============================================================
def run_evaluation(output_json_path: Optional[str] = "eval_results.json") -> Dict[str, Any]:
    """
    Executes the end-to-end evaluation pipeline across all golden dataset questions.
    Generates detailed per-question diagnostics and aggregate summary metrics.
    """
    print("\n" + "=" * 75)
    print("                RAG EVALUATION PIPELINE (5 CORE METRICS)")
    print("=" * 75)

    client = get_qdrant_client()
    groq_client = get_groq_client()
    model = get_embedding_model()

    # Step 1: Initialize collection with benchmark knowledge
    setup_eval_collection(client, model)

    all_scores = {
        "precision": [],
        "recall": [],
        "faithfulness": [],
        "relevancy": [],
        "correctness": [],
    }

    per_question_results = []

    # Step 2: Run each test question
    for test in GOLDEN_DATASET:
        qid = test["id"]
        question = test["question"]
        ground_truth = test["ground_truth"]

        print("\n" + "-" * 75)
        print(f"TEST CASE #{qid}: {question}")
        print(f"Ground Truth: {ground_truth}")
        print("-" * 75)

        # Retrieval
        retrieved_results = search_eval_collection(client, model, question, top_k=3)
        context = "\n".join([r.payload.get("text", "") for r in retrieved_results])

        print(f"Retrieved {len(retrieved_results)} chunk(s). Top score: {retrieved_results[0].score:.3f}" if retrieved_results else "No chunks retrieved.")

        # Generation
        answer = ask_rag_pipeline(groq_client, question, context)
        print(f"Generated Answer:\n  \"{answer}\"")

        # 1. Context Precision
        precision, precision_details = evaluate_context_precision(groq_client, question, retrieved_results)

        # 2. Context Recall
        recall_res = evaluate_context_recall(groq_client, question, context, ground_truth)
        recall = recall_res["score"]

        # 3. Faithfulness
        faith_res = evaluate_faithfulness(groq_client, question, context, answer)
        faithfulness = faith_res["score"]

        # 4. Answer Relevancy
        rel_res = evaluate_answer_relevancy(groq_client, question, answer)
        relevancy = rel_res["score"]

        # 5. Answer Correctness
        corr_res = evaluate_answer_correctness(groq_client, answer, ground_truth)
        correctness = corr_res["score"]

        # Record scores
        all_scores["precision"].append(precision)
        all_scores["recall"].append(recall)
        all_scores["faithfulness"].append(faithfulness)
        all_scores["relevancy"].append(relevancy)
        all_scores["correctness"].append(correctness)

        # Per-question printout
        print("\nScores for this query:")
        print(f"  • Context Precision : {precision:.2f}")
        print(f"  • Context Recall    : {recall:.2f} ({recall_res.get('reason', '')})")
        print(f"  • Faithfulness      : {faithfulness:.2f} ({faith_res.get('reason', '')})")
        print(f"  • Answer Relevancy  : {relevancy:.2f} ({rel_res.get('reason', '')})")
        print(f"  • Answer Correctness: {correctness:.2f} ({corr_res.get('reason', '')})")

        # Diagnostic flags
        diagnostics = []
        if precision < 0.7:
            diagnostics.append("⚠️ Low Precision: Retrieval returned irrelevant chunks.")
        if recall < 0.7:
            diagnostics.append("⚠️ Low Recall: Missing required information in retrieved context.")
        if faithfulness < 0.7:
            diagnostics.append("⚠️ Low Faithfulness: Answer contains ungrounded claims.")
        if relevancy < 0.7:
            diagnostics.append("⚠️ Low Relevancy: Answer is not directly addressing the prompt.")
        if correctness < 0.7:
            diagnostics.append("⚠️ Low Correctness: Answer contradicts ground truth.")

        if not diagnostics:
            print("  Status: ✅ PASSED — High quality grounded response.")
        else:
            for diag in diagnostics:
                print(f"  Status: {diag}")

        per_question_results.append({
            "id": qid,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "scores": {
                "precision": precision,
                "recall": recall,
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "correctness": correctness,
            },
            "diagnostics": diagnostics,
        })

    # Summary Aggregates
    print("\n" + "=" * 75)
    print("                          FINAL SUMMARY")
    print("=" * 75)

    summary_averages = {}
    for metric, values in all_scores.items():
        avg = sum(values) / len(values) if values else 0.0
        summary_averages[metric] = round(avg, 3)
        print(f"  • Average {metric.capitalize():18}: {avg:.2f}")

    final_report = {
        "summary": summary_averages,
        "questions": per_question_results,
    }

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)
        print(f"\nDetailed evaluation results saved to '{output_json_path}'.")

    print("=" * 75)
    return final_report


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run_evaluation()