"""
Eval runner — two experiments:

  Experiment A (graphrag_vs_naive):
    Compares GraphRAG hybrid retrieval vs dense-only (naive) retrieval.
    Proves the "60% QA correctness improvement" resume claim.
    Metrics: Faithfulness, ResponseRelevancy, FactualCorrectness*, LLMContextRecall*
    (* only when ground_truth is available in the dataset)

  Experiment B (model_comparison):
    Compares gpt-4o vs gpt-4.1-nano on identical GraphRAG-retrieved context.
    Proves "quality maintained after model swap" claim.
    Metrics: Faithfulness, ResponseRelevancy, FactualCorrectness*

Usage:
  python run_eval.py --experiment graphrag_vs_naive
  python run_eval.py --experiment model_comparison
  python run_eval.py --experiment all
  python run_eval.py --experiment all --dataset qa_dataset.json --output results/

Requires .env with:
  OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
  SEARCH_SERVICE_URL, SEARCH_SERVICE_TOKEN (optional for unauthenticated services)
"""

import argparse
import json
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
        return json.load(f)


# ── Experiment A ─────────────────────────────────────────────────────────────

def collect_graphrag_vs_naive(dataset: list) -> tuple[list, list]:
    """
    Run all queries through GraphRAG and naive runners.
    Returns (graphrag_samples, naive_samples) — lists ready for RAGAS scoring.
    """
    graphrag_samples, naive_samples = [], []
    total = len(dataset)

    for i, item in enumerate(dataset, 1):
        qid = item["id"]
        question = item["question"]
        ground_truth = item.get("ground_truth")

        print(f"  [{i}/{total}] {qid}: {question[:65]}...")

        # GraphRAG
        try:
            gr = graphrag_runner.query(question, session_id=f"eval_{qid}")
            graphrag_samples.append({
                "id": qid,
                "category": item.get("category", ""),
                "question": question,
                "answer": gr["answer"],
                "contexts": gr["contexts"],
                "ground_truth": ground_truth,
                "internal_faithfulness": gr.get("quality_score"),
            })
        except Exception as e:
            print(f"    GraphRAG ERROR: {e}")

        # Naive
        try:
            nv = naive_runner.query(question)
            naive_samples.append({
                "id": qid,
                "category": item.get("category", ""),
                "question": question,
                "answer": nv["answer"],
                "contexts": nv["contexts"],
                "ground_truth": ground_truth,
            })
        except Exception as e:
            print(f"    Naive ERROR: {e}")

    naive_runner.close()
    return graphrag_samples, naive_samples


def run_graphrag_vs_naive(dataset: list, output_dir: Path) -> pd.DataFrame:
    print("Collecting answers from GraphRAG and naive runners...")
    graphrag_samples, naive_samples = collect_graphrag_vs_naive(dataset)

    print(f"\nScoring {len(graphrag_samples)} GraphRAG samples with RAGAS...")
    gr_scores = judge.score_batch(graphrag_samples)
    gr_scores.insert(0, "system", "graphrag")

    print(f"Scoring {len(naive_samples)} naive samples with RAGAS...")
    nv_scores = judge.score_batch(naive_samples)
    nv_scores.insert(0, "system", "naive")

    # Attach internal faithfulness (from HallucinationValidator) to GraphRAG rows
    internal_scores = {s["id"]: s.get("internal_faithfulness") for s in graphrag_samples}
    if "user_input" in gr_scores.columns:
        # RAGAS attaches user_input; map back via position since order is preserved
        gr_scores["internal_faithfulness"] = [
            graphrag_samples[i].get("internal_faithfulness")
            for i in range(len(gr_scores))
        ]

    combined = pd.concat([gr_scores, nv_scores], ignore_index=True)

    _save_result(output_dir / "graphrag_vs_naive_full.json", {
        "graphrag_samples": graphrag_samples,
        "naive_samples": naive_samples,
        "graphrag_scores": gr_scores.to_dict(orient="records"),
        "naive_scores": nv_scores.to_dict(orient="records"),
    })

    return combined


# ── Experiment B ─────────────────────────────────────────────────────────────

def collect_model_comparison(dataset: list) -> tuple[list, list]:
    """
    For each question, retrieve context via GraphRAG then generate answers
    with both GRAPHRAG_MODEL and COMPARE_MODEL using identical context.
    Returns (prod_samples, compare_samples).
    """
    prod_samples, compare_samples = [], []
    total = len(dataset)

    for i, item in enumerate(dataset, 1):
        qid = item["id"]
        question = item["question"]
        ground_truth = item.get("ground_truth")

        print(f"  [{i}/{total}] {qid}: {question[:65]}...")

        # Retrieve context via GraphRAG (model-agnostic retrieval step)
        try:
            gr = graphrag_runner.query(question, session_id=f"mc_{qid}")
            contexts = gr["contexts"]
            prod_answer = gr["answer"]  # already generated by GRAPHRAG_MODEL
        except Exception as e:
            print(f"    GraphRAG ERROR: {e}")
            contexts, prod_answer = [], f"[ERROR: {e}]"

        # Generate with comparison model on the same contexts
        try:
            compare_answer = judge.model_comparison_answer(question, contexts, COMPARE_MODEL)
        except Exception as e:
            print(f"    {COMPARE_MODEL} ERROR: {e}")
            compare_answer = f"[ERROR: {e}]"

        base = {"id": qid, "category": item.get("category", ""), "question": question,
                "contexts": contexts, "ground_truth": ground_truth}

        prod_samples.append({**base, "answer": prod_answer})
        compare_samples.append({**base, "answer": compare_answer})

    return prod_samples, compare_samples


def run_model_comparison(dataset: list, output_dir: Path) -> pd.DataFrame:
    print(f"Collecting answers: {GRAPHRAG_MODEL} vs {COMPARE_MODEL}...")
    prod_samples, compare_samples = collect_model_comparison(dataset)

    print(f"\nScoring {GRAPHRAG_MODEL} samples with RAGAS...")
    prod_scores = judge.score_batch(prod_samples)
    prod_scores.insert(0, "model", GRAPHRAG_MODEL)

    print(f"Scoring {COMPARE_MODEL} samples with RAGAS...")
    compare_scores = judge.score_batch(compare_samples)
    compare_scores.insert(0, "model", COMPARE_MODEL)

    combined = pd.concat([prod_scores, compare_scores], ignore_index=True)

    _save_result(output_dir / "model_comparison_full.json", {
        "prod_samples": prod_samples,
        "compare_samples": compare_samples,
        "prod_scores": prod_scores.to_dict(orient="records"),
        "compare_scores": compare_scores.to_dict(orient="records"),
    })

    return combined


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_result(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def print_summary(df: pd.DataFrame, experiment: str):
    print(f"\n{'='*70}")
    print(f"RESULTS: {experiment}")
    print(f"{'='*70}")

    metric_cols = [c for c in df.columns if c not in
                   ("system", "model", "user_input", "response", "retrieved_contexts",
                    "reference", "id", "category", "internal_faithfulness")]

    group_col = "system" if experiment == "graphrag_vs_naive" else "model"
    if group_col not in df.columns:
        return

    summary = df.groupby(group_col)[metric_cols].mean().round(3)
    print(tabulate(summary, headers="keys", tablefmt="simple"))

    if experiment == "graphrag_vs_naive" and set(["graphrag", "naive"]).issubset(df[group_col].values):
        print("\nDelta (GraphRAG - Naive) / Naive  [improvement %]:")
        gr_row = summary.loc["graphrag"]
        nv_row = summary.loc["naive"]
        deltas = []
        for col in metric_cols:
            gr_val, nv_val = gr_row.get(col), nv_row.get(col)
            if gr_val is not None and nv_val and nv_val > 0:
                pct = round(((gr_val - nv_val) / nv_val) * 100, 1)
            else:
                pct = float("nan")
            deltas.append({"metric": col, "graphrag": gr_val, "naive": nv_val, "delta_%": pct})
        print(tabulate(deltas, headers="keys", tablefmt="simple"))

    print(f"\nFull results saved to results/ directory.")


def save_summary(df: pd.DataFrame, experiment: str, output_dir: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{experiment}_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV: {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run GraphRAG RAGAS evaluation experiments")
    parser.add_argument(
        "--experiment",
        choices=["graphrag_vs_naive", "model_comparison", "all"],
        default="all",
    )
    parser.add_argument("--dataset", default="datasets/qa_dataset.json")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    has_ground_truth = sum(1 for q in dataset if q.get("ground_truth"))
    if has_ground_truth == 0:
        print(
            "\nWARNING: No ground_truth in dataset — FactualCorrectness and LLMContextRecall "
            "will be skipped.\nRun: python generate_ground_truth.py  to populate ground truth first.\n"
        )
    else:
        print(f"\nGround truth available for {has_ground_truth}/{len(dataset)} questions.\n")

    if args.experiment in ("graphrag_vs_naive", "all"):
        print("--- Experiment A: GraphRAG vs Naive ---")
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
