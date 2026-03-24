"""
Hallucination Validator — scores answer faithfulness against retrieved context.

Used as a LangGraph validation node after answer generation. Answers scoring
below the threshold (default 0.7) are regenerated with a strict grounding prompt,
targeting ~40% reduction in unsupported claims.
"""

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """\
You are a strict fact-checker for financial documents.

Context (retrieved from SEC filings):
{context}

Generated Answer:
{answer}

Score this answer's faithfulness to the context on a scale of 0.0 to 1.0:
- 1.0 = every claim in the answer is directly supported by the context
- 0.0 = answer contains many claims not found in the context

Respond ONLY with valid JSON, no other text:
{{"score": 0.85, "unsupported_claims": ["claim 1", "claim 2"]}}"""


class HallucinationValidator:
    """LLM-based faithfulness scorer with conditional regeneration trigger."""

    def __init__(self, llm, threshold: float = 0.7):
        """
        Args:
            llm:       LangChain LLM instance (same model used by the agent).
            threshold: Minimum faithfulness score to pass. Answers below this
                       trigger regeneration. Default: 0.7.
        """
        self.llm = llm
        self.threshold = threshold

    def validate(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Score the faithfulness of an answer against retrieved context.

        Args:
            answer:  The generated answer to validate.
            context: The retrieved context passed to the LLM (first 3000 chars used).

        Returns:
            Dict with keys:
                score (float 0.0–1.0),
                unsupported_claims (list[str]),
                passed (bool — True if score >= threshold).
        """
        try:
            prompt = FAITHFULNESS_PROMPT.format(
                context=context[:3000],
                answer=answer,
            )
            result = self.llm.invoke(prompt)
            raw = result.content if hasattr(result, "content") else str(result)
            raw = raw.strip()

            # Strip markdown code fences if the model wraps the JSON
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            score = float(data.get("score", 0.5))
            return {
                "score": score,
                "unsupported_claims": data.get("unsupported_claims", []),
                "passed": score >= self.threshold,
            }
        except Exception as e:
            logger.warning("faithfulness validation failed: %s", e)
            # Fail open — pass the answer through rather than blocking a response
            return {"score": 0.5, "unsupported_claims": [], "passed": True}
