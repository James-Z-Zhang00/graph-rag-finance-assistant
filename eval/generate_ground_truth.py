"""
Ground truth generator — enriches qa_dataset.json with reference answers.

Strategy: calls the GraphRAG system with COMPARE_MODEL (gpt-4o) as the judge model,
treating gpt-4o answers as the gold standard. This is the standard approach when
manual ground truth is unavailable.

Usage:
  python generate_ground_truth.py
  python generate_ground_truth.py --dataset qa_dataset.json --output qa_dataset.json
"""

import argparse
import json
from config import COMPARE_MODEL
import graphrag_runner
import judge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/qa_dataset.json")
    parser.add_argument("--output", default="datasets/qa_dataset.json")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    total = len(dataset)
    for i, item in enumerate(dataset, 1):
        if item.get("ground_truth"):
            print(f"[{i}/{total}] {item['id']}: already has ground_truth, skipping")
            continue

        print(f"[{i}/{total}] Generating ground truth for {item['id']}...")
        try:
            # Get GraphRAG contexts
            gr_result = graphrag_runner.query(item["question"], session_id=f"gt_{item['id']}")
            contexts = gr_result["contexts"]

            # Generate reference answer with the stronger model
            reference = judge.model_comparison_answer(item["question"], contexts, COMPARE_MODEL)
            item["ground_truth"] = reference
            print(f"  Done: {reference[:80]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            item["ground_truth"] = None

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    populated = sum(1 for q in dataset if q.get("ground_truth"))
    print(f"\nGround truth generated for {populated}/{total} questions → {args.output}")


if __name__ == "__main__":
    main()
