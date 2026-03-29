"""
Pre-eval verification — checks that the documents referenced in the QA dataset
are actually present in the Neo4j graph before running eval.

A question scored against the wrong document (or a missing document) will produce
low RAGAS scores for the wrong reason — this script catches that before it happens.

Usage:
  python verify_ingestion.py --dataset datasets/financebench_sample.json
  python verify_ingestion.py --dataset datasets/docfinqa_sample.json

Exit codes:
  0 — all documents verified, safe to run eval
  1 — one or more documents missing from graph, do not run eval
"""

import argparse
import json
import sys
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


def load_dataset(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def extract_doc_identifiers(dataset: list) -> list[dict]:
    """
    Extract document identifiers from any supported dataset format.

    FinanceBench: uses doc_name + company + doc_period
    SECQUE:       uses accession_number
    DocFinQA:     no external document needed (context embedded in record)
    Generic:      uses doc_name if present
    """
    identifiers = []
    for item in dataset:
        category = item.get("category", "")

        if item.get("accession_number"):
            identifiers.append({
                "id": item["id"],
                "type": "accession_number",
                "value": item["accession_number"],
            })
        elif item.get("doc_name"):
            identifiers.append({
                "id": item["id"],
                "type": "doc_name",
                "value": item["doc_name"],
                "company": item.get("company", ""),
            })
        elif category == "docfinqa":
            # DocFinQA embeds context in the record — no external ingestion needed
            identifiers.append({
                "id": item["id"],
                "type": "embedded",
                "value": item.get("doc_name", "embedded"),
            })
        else:
            identifiers.append({
                "id": item["id"],
                "type": "unknown",
                "value": None,
            })

    return identifiers


def check_company_in_graph(driver, company: str) -> bool:
    """Check if a Company node exists in the graph matching the given name."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e)
            WHERE (e:__Entity__ OR e:Entity)
              AND toLower(e.name) CONTAINS toLower($company)
            RETURN count(e) AS cnt
            """,
            company=company,
        )
        return result.single()["cnt"] > 0


def check_filing_in_graph(driver, doc_name: str) -> bool:
    """Check if a filing/chunk referencing this document name exists in the graph."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:__Chunk__)
            WHERE toLower(c.fileName) CONTAINS toLower($doc_name)
               OR toLower(c.id) CONTAINS toLower($doc_name)
            RETURN count(c) AS cnt
            """,
            doc_name=doc_name,
        )
        return result.single()["cnt"] > 0


def check_graph_populated(driver) -> int:
    """Return total chunk count — a populated graph has > 0 chunks."""
    with driver.session() as session:
        result = session.run("MATCH (c:__Chunk__) RETURN count(c) AS cnt")
        return result.single()["cnt"]


def main():
    parser = argparse.ArgumentParser(description="Verify QA dataset documents are in Neo4j graph")
    parser.add_argument("--dataset", required=True, help="Path to QA dataset JSON")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    identifiers = extract_doc_identifiers(dataset)

    print(f"Connecting to Neo4j: {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Basic sanity check
    total_chunks = check_graph_populated(driver)
    if total_chunks == 0:
        print("\n[FAIL] Graph is empty — run the build pipeline first.")
        driver.close()
        sys.exit(1)

    print(f"[OK] Graph has {total_chunks} chunks.\n")

    # Check each document
    results = []
    unique_docs = {i["value"]: i for i in identifiers if i["value"]}.values()

    for doc in unique_docs:
        if doc["type"] == "embedded":
            status = "SKIP"
            detail = "context embedded in dataset record — no ingestion needed"
        elif doc["type"] == "unknown":
            status = "SKIP"
            detail = "no document identifier in this record"
        elif doc["type"] == "doc_name":
            # Try matching by filename, fall back to company name
            found = check_filing_in_graph(driver, doc["value"])
            if not found and doc.get("company"):
                found = check_company_in_graph(driver, doc["company"])
            status = "OK  " if found else "MISSING"
            detail = doc["value"]
        elif doc["type"] == "accession_number":
            found = check_filing_in_graph(driver, doc["value"])
            status = "OK  " if found else "MISSING"
            detail = f"accession: {doc['value']}"
        else:
            status = "SKIP"
            detail = ""

        results.append({"status": status, "detail": detail})
        print(f"  [{status}] {detail}")

    driver.close()

    missing = [r for r in results if r["status"] == "MISSING"]
    ok = [r for r in results if r["status"] == "OK  "]
    skipped = [r for r in results if r["status"] == "SKIP"]

    print(f"\nSummary: {len(ok)} verified, {len(missing)} missing, {len(skipped)} skipped")

    if missing:
        print(
            f"\n[FAIL] {len(missing)} document(s) not found in graph.\n"
            "Upload the missing filings to GCS and re-run the build pipeline before running eval.\n"
            "Filing URLs are in the doc_link field (FinanceBench) or constructable from accession_number (SECQUE)."
        )
        sys.exit(1)
    else:
        print("\n[PASS] All documents verified. Safe to run eval.")
        sys.exit(0)


if __name__ == "__main__":
    main()
