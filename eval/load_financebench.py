"""
FinanceBench dataset loader.

Downloads PatronusAI/financebench from HuggingFace (150 rows, train split),
converts to our qa_dataset.json format, and lists the exact PDF filenames
you need to download from GitHub and upload to GCS.

PDF source (FinanceBench GitHub):
  https://github.com/patronus-ai/financebench/tree/main/pdfs
  Each PDF is named exactly: {doc_name}.pdf

Usage:
  # Load all 150 questions
  python load_financebench.py --output datasets/financebench_sample.json

  # Filter to specific companies
  python load_financebench.py --companies "3M" "Microsoft" "Netflix" --output datasets/financebench_sample.json

  # Filter to specific question types
  python load_financebench.py --question-type domain-relevant --output datasets/financebench_sample.json

  # List all companies available in the dataset
  python load_financebench.py --list-companies

  # List all PDFs you need to download (without saving)
  python load_financebench.py --list-pdfs
"""

import argparse
import json
from collections import Counter
from datasets import load_dataset

GITHUB_PDF_BASE = "https://github.com/patronus-ai/financebench/blob/main/pdfs"


def load_financebench():
    print("Downloading PatronusAI/financebench from HuggingFace...")
    ds = load_dataset("PatronusAI/financebench", split="train")
    print(f"Loaded {len(ds)} QA pairs.")
    return ds


def list_companies(ds):
    counts = Counter(row["company"] for row in ds).most_common()
    print(f"\n{'Company':<35} {'Questions':>9}")
    print("-" * 46)
    for company, count in counts:
        print(f"  {company:<33} {count:>9}")
    print(f"\n{len(counts)} companies total, {len(ds)} questions total.")


def list_pdfs(ds):
    doc_names = sorted({row["doc_name"] for row in ds})
    print(f"\n{len(doc_names)} unique PDFs needed:\n")
    for doc in doc_names:
        print(f"  {doc}.pdf")
        print(f"    → {GITHUB_PDF_BASE}/{doc}.pdf")
    print(
        f"\nDownload all PDFs from:\n"
        f"  https://github.com/patronus-ai/financebench/tree/main/pdfs\n"
        f"Then upload to GCS: gs://graph-rag-finance-assistant/small/"
    )


def convert(ds, companies: list[str] | None, question_type: str | None) -> list[dict]:
    rows = list(ds)

    if companies:
        companies_lower = [c.lower() for c in companies]
        rows = [r for r in rows if r["company"].lower() in companies_lower]
        print(f"Filtered to {len(rows)} questions for companies: {companies}")

    if question_type:
        rows = [r for r in rows if r.get("question_type", "") == question_type]
        print(f"Filtered to {len(rows)} questions of type: {question_type}")

    output = []
    for row in rows:
        # Flatten evidence into a list of text passages
        evidence_texts = []
        if row.get("evidence"):
            for e in row["evidence"]:
                if isinstance(e, dict) and e.get("evidence_text"):
                    evidence_texts.append(e["evidence_text"])

        output.append({
            "id": row["financebench_id"],
            "question": row["question"],
            "ground_truth": row["answer"],
            "doc_name": row["doc_name"],
            "company": row["company"],
            "doc_type": row.get("doc_type", ""),
            "doc_period": row.get("doc_period", ""),
            "doc_link": row.get("doc_link", ""),
            "category": row.get("question_type", ""),
            "question_reasoning": row.get("question_reasoning", ""),
            "evidence": evidence_texts,
            "pdf_filename": f"{row['doc_name']}.pdf",
            "pdf_url": f"{GITHUB_PDF_BASE}/{row['doc_name']}.pdf",
        })

    return output


def print_pdf_list(output: list[dict]):
    doc_names = sorted({r["doc_name"] for r in output})
    print(f"\n{len(doc_names)} PDF(s) needed for this sample:")
    for doc in doc_names:
        print(f"  {doc}.pdf")
        print(f"    → {GITHUB_PDF_BASE}/{doc}.pdf")
    print(
        f"\nAfter downloading, upload to GCS:\n"
        f"  gsutil cp *.pdf gs://graph-rag-finance-assistant/small/\n"
        f"Then trigger the build pipeline to populate Neo4j."
    )


def main():
    parser = argparse.ArgumentParser(description="Load FinanceBench dataset for eval")
    parser.add_argument("--output", default="datasets/financebench_sample.json")
    parser.add_argument(
        "--companies",
        nargs="+",
        default=None,
        help='Filter by company name, e.g. --companies "3M" "Microsoft"',
    )
    parser.add_argument(
        "--question-type",
        choices=["metrics-generated", "domain-relevant", "novel-generated"],
        default=None,
        help="Filter by question type",
    )
    parser.add_argument("--list-companies", action="store_true", help="List companies and exit")
    parser.add_argument("--list-pdfs", action="store_true", help="List required PDFs and exit")
    args = parser.parse_args()

    ds = load_financebench()

    if args.list_companies:
        list_companies(ds)
        return

    if args.list_pdfs:
        list_pdfs(ds)
        return

    output = convert(ds, args.companies, args.question_type)

    if not output:
        print("No questions matched the filters. Check --companies / --question-type values.")
        return

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print_pdf_list(output)
    print(f"\nSaved {len(output)} questions → {args.output}")


if __name__ == "__main__":
    main()
