"""
evalu.py - Evaluation Runner and Interface
=========================================
Runs RAG evaluation pipeline for the 5 core metrics:
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy
- Answer Correctness

Supports both:
1. Benchmark Golden Dataset mode (for regression testing)
2. Live Query Evaluation mode (for any custom uploaded PDF)
"""

from rag_evalu import (
    run_evaluation,
    evaluate_live_query,
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_faithfulness,
    evaluate_answer_relevancy,
    evaluate_answer_correctness,
    GOLDEN_DATASET,
)

if __name__ == "__main__":
    run_evaluation(output_json_path="eval_results.json")
