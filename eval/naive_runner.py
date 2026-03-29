"""
Naive (dense-only) runner — bypasses all graph traversal and community context.

Pipeline:
  1. Embed the question with OpenAI
  2. Query Neo4j '__Chunk__' vector index directly (no entity/relationship/community lookup)
  3. Feed top-k chunk texts to the same LLM for generation

This is the baseline that GraphRAG is compared against.
"""

from typing import List
from openai import OpenAI
from neo4j import GraphDatabase
from config import (
    OPENAI_API_KEY,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    GRAPHRAG_MODEL,
    EVAL_TOP_K,
)

_openai = OpenAI(api_key=OPENAI_API_KEY)
_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    return _driver


def _embed(text: str) -> List[float]:
    resp = _openai.embeddings.create(input=text, model="text-embedding-3-small")
    return resp.data[0].embedding


def _vector_search(embedding: List[float], top_k: int) -> List[str]:
    """Query the '__Chunk__' vector index and return chunk texts."""
    cypher = """
    CALL db.index.vector.queryNodes('vector', $top_k, $embedding)
    YIELD node, score
    RETURN node.text AS text, score
    ORDER BY score DESC
    """
    with _get_driver().session() as session:
        results = session.run(cypher, embedding=embedding, top_k=top_k)
        return [r["text"] for r in results if r["text"]]


def _generate(question: str, contexts: List[str]) -> str:
    context_str = "\n\n---\n\n".join(contexts) if contexts else "No relevant information found."
    response = _openai.chat.completions.create(
        model=GRAPHRAG_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analyst assistant. Answer the question based solely "
                    "on the provided SEC filing excerpts. Be specific and cite figures where available."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {question}",
            },
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def query(question: str) -> dict:
    """
    Run dense-only retrieval (no graph context) and generate an answer.

    Returns a dict with:
      answer    — generated answer string
      contexts  — list of retrieved chunk texts (vector search only)
    """
    embedding = _embed(question)
    contexts = _vector_search(embedding, top_k=EVAL_TOP_K)
    answer = _generate(question, contexts)
    return {"answer": answer, "contexts": contexts}


def close():
    global _driver
    if _driver:
        _driver.close()
        _driver = None
