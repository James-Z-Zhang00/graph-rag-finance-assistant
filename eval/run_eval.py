"""
Eval runner — two experiments:

  Experiment A (graphrag_vs_naive):
    Compares GraphRAG hybrid retrieval vs dense-only (naive) retrieval.
    Proves the "60% QA correctness improvement" resume claim.

  Experiment B (model_comparison):
    Compares gpt-4o vs gpt-4.1-nano on the same GraphRAG-retrieved context.
    Proves "quality maintained after model swap" claim.

Usage:
  python run_eval.py --experiment graphrag_vs_naive
  python run_eval.py --experiment model_comparison
  python run_eval.py --experiment all
  python run_eval.py --experiment graphrag_vs_naive --dataset qa_dataset.json --output results/

Requires .env with:
  OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
  SEARCH_SERVICE_URL, SEARCH_SERVICE_TOKEN (optional if not authenticated)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tabulate import tabulate

import graphrag_runner
import naive_runner
import judge
from config import GRAPHRAG_MODEL, COMPARE_MODEL


def load_dataset(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data


def run_graphrag_vs_naive(dataset: list, output_dir: Path) -> pd.DataFrame:
    """
    For each question:
      - Get GraphRAG answer + contexts
      - Get naive answer + contexts
      - Score both with answer_relevancy and faithfulness
      - Score answer_correctness if ground_truth is available
    """
    rows = []
    total = len(dataset)

    for i, item in enumerate(dataset, 1):
        qid = item["id"]
        question = item["question"]
        ground_truth = item.get("ground_truth")

        print(f"[{i}/{total}] {qid}: {question[:60]}...")

        # GraphRAG
        try:
            gr_result = graphrag_runner.query(question, session_id=f"eval_{qid}")
            gr_answer = gr_result["answer"]
            gr_contexts = gr_result["contexts"]
            gr_internal_faithfulness = gr_result.get("quality_score")
        except Exception as e:
            print(f"  GraphRAG ERROR: {e}")
            gr_answer, gr_contexts, gr_internal_faithfulness = f"[ERROR: {e}]", [], None

        # Naive
        try:
            nv_result = naive_runner.query(question)
            nv_answer = nv_result["answer"]
            nv_contexts = nv_result["contexts"]
        except Exception as e:
            print(f"  Naive ERROR: {e}")
            nv_answer, nv_contexts = f"[ERROR: {e}]", []

        # Score GraphRAG
        gr_relevancy = judge.answer_relevancy(question, gr_answer)
        gr_faithfulness = judge.faithfulness(gr_answer, gr_contexts)
        gr_correctness = judge.answer_correctness(question, gr_answer, ground_truth)

        # Score Naive
        nv_relevancy = judge.answer_relevancy(question, nv_answer)
        nv_faithfulness = judge.faithfulness(nv_answer, nv_contexts)
        nv_correctness = judge.answer_correctness(question, nv_answer, ground_truth)

        row = {
            "id": qid,
            "category": item.get("category", ""),
            "question": question[:80],
            "gr_relevancy": gr_relevancy.get("score"),
            "gr_faithfulness": gr_faithfulness.get("score") or gr_internal_faithfulness,
            "gr_correctness": gr_correctness.get("score"),
            "nv_relevancy": nv_relevancy.get("score"),
            "nv_faithfulness": nv_faithfulness.get("score"),
            "nv_correctness": nv_correctness.get("score"),
        }
        rows.append(row)

        # Save incremental progress
        _save_result(output_dir / f"graphrag_vs_naive_{qid}.json", {
            "question": question,
            "ground_truth": ground_truth,
            "graphrag": {
                "answer": gr_answer,
                "contexts": gr_contexts,
                "scores": {
                    "relevancy": gr_relevancy,
                    "faithfulness": gr_faithfulness,
                    "correctness": gr_correctness,
                    "internal_faithfulness": gr_internal_faithfulness,
                },
            },
            "naive": {
                "answer": nv_answer,
                "contexts": nv_contexts,
                "scores": {
                    "relevancy": nv_relevancy,
                    "faithfulness": nv_faithfulness,
                    "correctness": nv_correctness,
                },
            },
        })

    naive_runner.close()
    return pd.DataFrame(rows)


def run_model_comparison(dataset: list, output_dir: Path) -> pd.DataFrame:
    """
    For each question:
      - Get GraphRAG contexts (retrieval step only, model-agnostic)
      - Generate answers with GRAPHRAG_MODEL (gpt-4.1-nano) and COMPARE_MODEL (gpt-4o)
      - Score both answers with answer_relevancy and answer_correctness
    """
    rows = []
    total = len(dataset)

    for i, item in enumerate(dataset, 1):
        qid = item["id"]
        question = item["question"]
        ground_truth = item.get("ground_truth")

        print(f"[{i}/{total}] {qid}: {question[:60]}...")

        # Get GraphRAG contexts (retrieved by the deployed service, which uses gpt-4.1-nano)
        try:
            gr_result = graphrag_runner.query(question, session_id=f"eval_mc_{qid}")
            contexts = gr_result["contexts"]
        except Exception as e:
            print(f"  GraphRAG context fetch ERROR: {e}")
            contexts = []

        # Generate with production model (gpt-4.1-nano via deployed service)
        prod_answer = gr_result.get("answer", "") if "gr_result" in dir() else ""

        # Generate with comparison model (gpt-4o) on the same contexts
        try:
            compare_answer = judge.model_comparison_answer(question, contexts, COMPARE_MODEL)
        except Exception as e:
            print(f"  Compare model ERROR: {e}")
            compare_answer = f"[ERROR: {e}]"

        # Score both
        prod_relevancy = judge.answer_relevancy(question, prod_answer)
        prod_correctness = judge.answer_correctness(question, prod_answer, ground_truth)
        compare_relevancy = judge.answer_relevancy(question, compare_answer)
        compare_correctness = judge.answer_correctness(question, compare_answer, ground_truth)

        row = {
            "id": qid,
            "category": item.get("category", ""),
            "question": question[:80],
            f"{GRAPHRAG_MODEL}_relevancy": prod_relevancy.get("score"),
            f"{GRAPHRAG_MODEL}_correctness": prod_correctness.get("score"),
            f"{COMPARE_MODEL}_relevancy": compare_relevancy.get("score"),
            f"{COMPARE_MODEL}_correctness": compare_correctness.get("score"),
        }
        rows.append(row)

        _save_result(output_dir / f"model_comparison_{qid}.json", {
            "question": question,
            "ground_truth": ground_truth,
            "contexts_from_graphrag": contexts,
            GRAPHRAG_MODEL: {
                "answer": prod_answer,
                "scores": {"relevancy": prod_relevancy, "correctness": prod_correctness},
            },
            COMPARE_MODEL: {
                "answer": compare_answer,
                "scores": {"relevancy": compare_relevancy, "correctness": compare_correctness},
            },
        })

    return pd.DataFrame(rows)


def _save_result(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def print_summary(df: pd.DataFrame, experiment: str):
    print(f"\n{'='*70}")
    print(f"RESULTS: {experiment}")
    print(f"{'='*70}")

    numeric_cols = df.select_dtypes(include="float").columns
    means = df[numeric_cols].mean().round(3)
    print("\nMean scores across all questions:")
    print(tabulate(means.reset_index(), headers=["metric", "mean"], tablefmt="simple"))

    if experiment == "graphrag_vs_naive":
        gr_cols = [c for c in numeric_cols if c.startswith("gr_")]
        nv_cols = [c for c in numeric_cols if c.startswith("nv_")]
        metric_names = [c.replace("gr_", "") for c in gr_cols]

        comparison = []
        for gr_col, nv_col, metric in zip(gr_cols, nv_cols, metric_names):
            gr_mean = df[gr_col].dropna().mean()
            nv_mean = df[nv_col].dropna().mean()
            if nv_mean and nv_mean > 0:
                delta_pct = ((gr_mean - nv_mean) / nv_mean) * 100
            else:
                delta_pct = float("nan")
            comparison.append({
                "metric": metric,
                "graphrag": round(gr_mean, 3),
                "naive": round(nv_mean, 3),
                "delta_%": round(delta_pct, 1),
            })

        print("\nGraphRAG vs Naive comparison:")
        print(tabulate(comparison, headers="keys", tablefmt="simple"))

    print(f"\nFull results saved to results/ directory.")


def save_summary(df: pd.DataFrame, experiment: str, output_dir: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{experiment}_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run GraphRAG evaluation experiments")
    parser.add_argument(
        "--experiment",
        choices=["graphrag_vs_naive", "model_comparison", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--dataset",
        default="qa_dataset.json",
        help="Path to Q&A dataset JSON (default: qa_dataset.json)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results (default: results/)",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    ground_truth_count = sum(1 for q in dataset if q.get("ground_truth"))
    if ground_truth_count == 0:
        print(
            "\nWARNING: No ground_truth values found in dataset.\n"
            "answer_correctness scores will be None.\n"
            "To add ground truth: edit qa_dataset.json and fill in 'ground_truth' fields,\n"
            "or run: python generate_ground_truth.py\n"
        )

    if args.experiment in ("graphrag_vs_naive", "all"):
        print("\n--- Experiment A: GraphRAG vs Naive ---")
        df_a = run_graphrag_vs_naive(dataset, output_dir)
        print_summary(df_a, "graphrag_vs_naive")
        save_summary(df_a, "graphrag_vs_naive", output_dir)

    if args.experiment in ("model_comparison", "all"):
        print(f"\n--- Experiment B: {GRAPHRAG_MODEL} vs {COMPARE_MODEL} ---")
        df_b = run_model_comparison(dataset, output_dir)
        print_summary(df_b, "model_comparison")
        save_summary(df_b, "model_comparison", output_dir)


if __name__ == "__main__":
    main()
