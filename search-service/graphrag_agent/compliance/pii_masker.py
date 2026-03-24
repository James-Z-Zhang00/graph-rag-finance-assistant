"""
PII Masker — detects and masks personal identifiable information in query text.

Applied to incoming query text before caching, audit writes, and LLM calls.
NOT applied to retrieved SEC financial context (entity names are intentional content).
"""

import re
from typing import Tuple, List

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}


class PIIMasker:
    """Regex-based PII detector and masker."""

    def mask(self, text: str) -> Tuple[str, List[str]]:
        """
        Scan text for PII patterns and replace each match with a placeholder.

        Args:
            text: Input text (e.g. a user query).

        Returns:
            Tuple of (masked_text, list_of_detected_pii_types).
            Example: ("What is revenue? My SSN is [PII:SSN]", ["ssn"])
        """
        found_types: List[str] = []
        for pii_type, pattern in PII_PATTERNS.items():
            new_text, count = re.subn(
                pattern,
                f"[PII:{pii_type.upper()}]",
                text,
                flags=re.IGNORECASE,
            )
            if count > 0:
                found_types.append(pii_type)
                text = new_text
        return text, found_types
