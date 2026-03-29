"""
RAGAS-based scoring module.

Metrics used:
  Faithfulness        — are all answer claims grounded in retrieved contexts?
  ResponseRelevancy   — does the answer address the question?
  FactualCorrectness  — does the answer match the reference answer? (requires ground_truth)
  LLMContextRecall    — does retrieved context cover what the reference answer contains? (requires ground_truth)

Usage pattern (batch — preferred):
  samples = [{"question": ..., "answer": ..., "contexts": [...], "ground_truth": ...}, ...]
  result_df = score_batch(samples)

Also exposes model_comparison_answer() for Experiment B (model swap eval).
"""

from typing import List, Optional
from openai import OpenAI
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy, FactualCorrectness, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import OPENAI_API_KEY, JUDGE_MODEL

_openai = OpenAI(api_key=OPENAI_API_KEY)


def _ragas_llm():
    return LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, api_key=OPENAI_API_KEY, temperature=0))


def _ragas_embeddings():
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=OPENAI_API_KEY))


def score_batch(samples: List[dict]) -> "pandas.DataFrame":
    """
    Score a list of samples using RAGAS evaluate().

    Each sample dict:
      question     str          — the input question
      answer       str          — model's generated answer
      contexts     List[str]    — retrieved context strings
      ground_truth str | None   — reference answer (enables FactualCorrectness + LLMContextRecall)

    Returns a pandas DataFrame with per-sample scores and metric means.
    """
    import pandas as pd

    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s.get("contexts") or [],
            reference=s.get("ground_truth") or None,
        )
        for s in samples
    ]

    llm = _ragas_llm()
    embeddings = _ragas_embeddings()

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
    ]

    has_ground_truth = any(s.get("ground_truth") for s in samples)
    if has_ground_truth:
        metrics += [
            FactualCorrectness(llm=llm),
            LLMContextRecall(llm=llm),
        ]

    dataset = EvaluationDataset(samples=ragas_samples)
    result = evaluate(dataset=dataset, metrics=metrics)
    return result.to_pandas()


def model_comparison_answer(question: str, contexts: List[str], model: str) -> str:
    """
    Generate an answer using a specific model on the provided contexts.
    Used in Experiment B to compare gpt-4o vs gpt-4.1-nano on identical retrieved context.
    """
    context_str = "\n\n---\n\n".join(contexts[:3]) if contexts else "No context available."
    response = _openai.chat.completions.create(
        model=model,
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
