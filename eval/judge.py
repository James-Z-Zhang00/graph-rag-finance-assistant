"""
LLM-as-judge scoring module.

Three metrics:
  answer_correctness  — semantic agreement between answer and ground_truth (0–1)
  answer_relevancy    — does the answer address the question? (0–1)
  faithfulness        — is the answer grounded in the provided contexts? (0–1)

Each scorer calls the JUDGE_MODEL (default: gpt-4o) and extracts a 0–1 score
from a structured JSON response.
"""

import json
from openai import OpenAI
from config import OPENAI_API_KEY, JUDGE_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)

_CORRECTNESS_PROMPT = """\
You are an evaluation judge for RAG systems. Score how well the ANSWER matches the REFERENCE ANSWER.

Question: {question}
Reference Answer: {ground_truth}
Answer to Evaluate: {answer}

Score from 0.0 to 1.0:
  1.0 = answer covers all key facts in the reference with no contradictions
  0.7 = answer covers most key facts, minor omissions
  0.5 = answer partially correct, missing important details or has minor errors
  0.3 = answer has significant omissions or factual errors
  0.0 = answer is completely wrong or irrelevant

Respond ONLY with valid JSON: {{"score": <float>, "reason": "<one sentence>"}}"""

_RELEVANCY_PROMPT = """\
You are an evaluation judge for RAG systems. Score whether the ANSWER actually addresses the QUESTION.

Question: {question}
Answer: {answer}

Score from 0.0 to 1.0:
  1.0 = answer directly and completely addresses the question
  0.7 = answer mostly addresses the question with minor tangents
  0.5 = answer partially addresses the question
  0.3 = answer is loosely related but does not address the core question
  0.0 = answer is completely off-topic

Respond ONLY with valid JSON: {{"score": <float>, "reason": "<one sentence>"}}"""

_FAITHFULNESS_PROMPT = """\
You are an evaluation judge for RAG systems. Score whether the ANSWER is grounded in the CONTEXT.

Context: {context}
Answer: {answer}

Score from 0.0 to 1.0:
  1.0 = every claim in the answer is directly supported by the context
  0.7 = most claims are supported; minor extrapolations
  0.5 = about half the claims are supported; notable unsupported claims
  0.3 = few claims are grounded; significant hallucination
  0.0 = answer is entirely unsupported by the context

Respond ONLY with valid JSON: {{"score": <float>, "reason": "<one sentence>"}}"""


def _call_judge(prompt: str) -> dict:
    response = _client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def answer_correctness(question: str, answer: str, ground_truth: str) -> dict:
    """Score answer against ground truth. Requires ground_truth to be non-null."""
    if not ground_truth:
        return {"score": None, "reason": "No ground_truth provided"}
    result = _call_judge(
        _CORRECTNESS_PROMPT.format(
            question=question, ground_truth=ground_truth, answer=answer
        )
    )
    return result


def answer_relevancy(question: str, answer: str) -> dict:
    """Score whether the answer addresses the question."""
    return _call_judge(_RELEVANCY_PROMPT.format(question=question, answer=answer))


def faithfulness(answer: str, contexts: list) -> dict:
    """Score whether the answer is grounded in retrieved contexts."""
    if not contexts:
        return {"score": None, "reason": "No contexts provided"}
    context_str = "\n\n---\n\n".join(contexts[:3])  # cap at 3 to stay in token budget
    return _call_judge(_FAITHFULNESS_PROMPT.format(context=context_str, answer=answer))


def model_comparison_answer(question: str, contexts: list, model: str) -> str:
    """
    Generate an answer using a specific model on the provided contexts.
    Used to compare gpt-4o vs gpt-4.1-nano on the same retrieved context.
    """
    context_str = "\n\n---\n\n".join(contexts[:3]) if contexts else "No context available."
    response = _client.chat.completions.create(
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
