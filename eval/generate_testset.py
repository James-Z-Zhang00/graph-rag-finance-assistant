"""
RAGAS TestsetGenerator — generates QA pairs from PDF documents.

Loads PDFs from a local directory, generates synthetic QA pairs using RAGAS,
and saves them to our qa_dataset.json format with ground truth included.

Usage:
  # Generate 30 questions from all PDFs in a folder
  python generate_testset.py --pdfs ../TEST-FILE/pdf/ --n 30 --output datasets/generated_testset.json

  # Generate from specific PDF files
  python generate_testset.py --pdfs ../TEST-FILE/pdf/MICROSOFT_2016_10K.pdf --n 20 --output datasets/generated_testset.json

  # Diagnose PDFs without generating (check page count and extracted text)
  python generate_testset.py --pdfs ../TEST-FILE/pdf/ --diagnose
"""

import argparse
import json
import sys
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from openai import OpenAI
from ragas.testset import TestsetGenerator
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.testset.transforms.default import default_transforms

from config import OPENAI_API_KEY, JUDGE_MODEL

MIN_CHARS_PER_DOC = 500   # warn if a PDF yields less than this


def load_pdfs(paths: list[Path]) -> list:
    """Load PDF files using PyMuPDF and return LangChain Document objects."""
    all_docs = []
    for path in paths:
        print(f"  Loading {path.name}...")
        loader = PyMuPDFLoader(str(path))
        docs = loader.load()
        total_chars = sum(len(d.page_content) for d in docs)

        for doc in docs:
            doc.metadata["source_pdf"] = path.name

        all_docs.extend(docs)
        print(f"    → {len(docs)} pages, {total_chars:,} characters extracted")

        if total_chars < MIN_CHARS_PER_DOC:
            print(
                f"    ⚠ WARNING: very little text extracted from {path.name}.\n"
                f"      This PDF is likely image-based (scanned). PyMuPDF cannot extract\n"
                f"      text from scanned pages. Download the HTML version from SEC EDGAR instead:\n"
                f"      https://efts.sec.gov/LATEST/search-index?q=\"{path.stem.split('-')[0]}\"&forms=10-K,10-Q"
            )

    return all_docs


def collect_pdf_paths(input_paths: list[str]) -> list[Path]:
    """Accept either a directory or individual file paths."""
    pdf_paths = []
    for p in input_paths:
        path = Path(p)
        if path.is_dir():
            found = sorted(path.glob("*.pdf"))
            if not found:
                print(f"WARNING: No PDFs found in {path}")
            pdf_paths.extend(found)
        elif path.suffix.lower() == ".pdf" and path.exists():
            pdf_paths.append(path)
        else:
            print(f"WARNING: Skipping {p} — not a PDF or does not exist")
    return pdf_paths


def get_transforms(docs, llm, embeddings):
    """Return the full default RAGAS transforms pipeline."""
    return default_transforms(documents=docs, llm=llm, embedding_model=embeddings)


def generate(docs: list, n: int) -> list[dict]:
    """Run RAGAS TestsetGenerator and return samples as dicts."""
    total_chars = sum(len(d.page_content) for d in docs)
    if total_chars < 1000:
        print(
            f"\nERROR: Only {total_chars} characters of text across all documents.\n"
            "Not enough content to generate meaningful QA pairs.\n"
            "Fix: download the HTML version of these filings from SEC EDGAR."
        )
        sys.exit(1)

    print(f"\nGenerating {n} QA pairs from {total_chars:,} characters of content...")
    print("(This calls gpt-4o — may take a few minutes)\n")

    client = OpenAI(api_key=OPENAI_API_KEY)
    llm = llm_factory(JUDGE_MODEL, client=client)
    embeddings = RagasOpenAIEmbeddings(model="text-embedding-3-small", client=client)

    transforms = get_transforms(docs, llm, embeddings)
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    testset = generator.generate_with_langchain_docs(
        docs, testset_size=n, transforms=transforms
    )

    rows = testset.to_list()
    print(f"Generated {len(rows)} QA pairs.")

    output = []
    for i, row in enumerate(rows):
        output.append({
            "id": f"gen{i+1:03d}",
            "question": row.get("user_input", ""),
            "ground_truth": row.get("reference", ""),
            "contexts": row.get("reference_contexts", []),
            "category": row.get("synthesizer_name", "ragas-generated"),
            "doc_name": "",
        })

    return output


def diagnose(paths: list[Path]):
    """Print page count and character count per PDF without generating anything."""
    print(f"\n{'File':<40} {'Pages':>6} {'Characters':>12} {'Status'}")
    print("-" * 72)
    for path in paths:
        loader = PyMuPDFLoader(str(path))
        docs = loader.load()
        chars = sum(len(d.page_content) for d in docs)
        status = "OK" if chars >= MIN_CHARS_PER_DOC else "⚠ IMAGE-BASED / TOO SPARSE"
        print(f"  {path.name:<38} {len(docs):>6} {chars:>12,}  {status}")
    print(
        "\nIf a file shows IMAGE-BASED, download the HTML version from:\n"
        "  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate QA testset from PDFs using RAGAS")
    parser.add_argument(
        "--pdfs",
        nargs="+",
        required=True,
        help="PDF file paths or a directory. e.g. --pdfs ../TEST-FILE/pdf/",
    )
    parser.add_argument("--n", type=int, default=30, help="Number of QA pairs to generate")
    parser.add_argument("--output", default="datasets/generated_testset.json")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Check page count and character extraction without generating QA",
    )
    args = parser.parse_args()

    pdf_paths = collect_pdf_paths(args.pdfs)
    if not pdf_paths:
        print("No PDF files found. Exiting.")
        return

    print(f"\nPDFs to process ({len(pdf_paths)}):")
    for p in pdf_paths:
        print(f"  {p.name}")

    if args.diagnose:
        diagnose(pdf_paths)
        return

    docs = load_pdfs(pdf_paths)
    print(f"\nTotal: {len(docs)} pages loaded")

    samples = generate(docs, args.n)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2)

    has_ground_truth = sum(1 for s in samples if s.get("ground_truth"))
    print(f"\nSaved {len(samples)} QA pairs ({has_ground_truth} with ground truth) → {args.output}")
    print(f"\nNext steps:")
    print(f"  python verify_ingestion.py --dataset {args.output}")
    print(f"  python run_eval.py --dataset {args.output}")


if __name__ == "__main__":
    main()
