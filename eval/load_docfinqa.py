"""
DocFinQA dataset loader.

Downloads kensho/DocFinQA from HuggingFace and samples 30–50 QA pairs,
optionally filtering to only questions whose source filing matches what
you have ingested into the Neo4j graph.

DocFinQA fields:
  id, question, answer, doc_name, page_num, evidence (list of text passages)

The `evidence` field maps directly to our ground_truth. The `doc_name` tells
you which 10-K filing the question came from — use this to verify that the
filing was ingested before running eval.

Usage:
  # Sample 40 questions from any filing
  python load_docfinqa.py --n 40 --output datasets/docfinqa_sample.json

  # Filter to specific companies (must match ingested filings)
  python load_docfinqa.py --n 40 --companies AAPL MSFT TSLA --output datasets/docfinqa_sample.json

  # List available company tickers in the dataset
  python load_docfinqa.py --list-companies
"""

import argparse
import json
import random
from collections import Counter
from datasets import load_dataset


def load_docfinqa():
    print("Downloading kensho/DocFinQA from HuggingFace...")
    ds = load_dataset("kensho/DocFinQA", split="train")
    print(f"Loaded {len(ds)} QA pairs.")
    return ds


def extract_ticker(doc_name: str) -> str:
    """Best-effort extraction of company ticker from doc_name field."""
    # DocFinQA doc_names typically look like: 'AAPL_10K_2022.pdf' or 'apple_10k_2022'
    return doc_name.split("_")[0].upper()


def list_companies(ds):
    tickers = [extract_ticker(row["doc_name"]) for row in ds]
    counts = Counter(tickers).most_common(30)
    print("\nTop companies in DocFinQA:")
    for ticker, count in counts:
        print(f"  {ticker}: {count} questions")


def sample_dataset(ds, n: int, companies: list[str] | None) -> list[dict]:
    rows = list(ds)

    if companies:
        companies_upper = [c.upper() for c in companies]
        rows = [r for r in rows if extract_ticker(r["doc_name"]) in companies_upper]
        print(f"Filtered to {len(rows)} questions matching companies: {companies_upper}")

    if len(rows) < n:
        print(f"WARNING: Only {len(rows)} questions available after filtering. Using all.")
        n = len(rows)

    sampled = random.sample(rows, n)

    output = []
    for i, row in enumerate(sampled, 1):
        # Combine evidence passages as ground_truth — same text the answer was derived from
        evidence = row.get("evidence") or []
        ground_truth = row.get("answer", "")

        output.append({
            "id": f"dfq{i:03d}",
            "question": row["question"],
            "ground_truth": ground_truth,
            "evidence": evidence,            # source passages from the actual filing
            "doc_name": row.get("doc_name", ""),
            "company": extract_ticker(row.get("doc_name", "")),
            "category": "docfinqa",
        })

    return output


def main():
    parser = argparse.ArgumentParser(description="Sample DocFinQA for eval")
    parser.add_argument("--n", type=int, default=40, help="Number of QA pairs to sample")
    parser.add_argument(
        "--companies",
        nargs="+",
        default=None,
        help="Filter to specific company tickers (e.g. AAPL MSFT TSLA)",
    )
    parser.add_argument(
        "--output",
        default="datasets/docfinqa_sample.json",
        help="Output path for sampled dataset",
    )
    parser.add_argument(
        "--list-companies",
        action="store_true",
        help="Print available companies and exit",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    ds = load_docfinqa()

    if args.list_companies:
        list_companies(ds)
        return

    sampled = sample_dataset(ds, args.n, args.companies)

    with open(args.output, "w") as f:
        json.dump(sampled, f, indent=2)

    companies_in_sample = list({r["company"] for r in sampled})
    print(f"\nSampled {len(sampled)} questions from {len(companies_in_sample)} companies.")
    print(f"Companies: {sorted(companies_in_sample)}")
    print(
        f"\nIMPORTANT: These filings must be ingested into Neo4j before running eval.\n"
        f"Download them from SEC EDGAR (search by company + '10-K') and run the build pipeline.\n"
        f"Saved to: {args.output}"
    )


if __name__ == "__main__":
    main()
