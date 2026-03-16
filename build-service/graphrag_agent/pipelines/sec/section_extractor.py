from __future__ import annotations

import re
from typing import List, Optional

from graphrag_agent.pipelines.sec.models import FilingSection, FormType, SECDocument


class SectionExtractor:
    """
    SEC filing section extractor for 10-K/10-Q/8-K item sections.
    Works on normalized text content (tables already separated).
    """

    FORM_ITEM_MAP = {
        "10-K": [
            "1",
            "1A",
            "1B",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "7A",
            "8",
            "9",
            "9A",
            "9B",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
        ],
        "10-Q": [
            "1",
            "1A",
            "1B",
            "2",
            "3",
            "4",
        ],
        "8-K": [
            "1.01",
            "1.02",
            "1.03",
            "1.04",
            "1.05",
            "2.01",
            "2.02",
            "2.03",
            "2.04",
            "2.05",
            "2.06",
            "3.01",
            "3.02",
            "3.03",
            "4.01",
            "4.02",
            "5.01",
            "5.02",
            "5.03",
            "5.04",
            "5.05",
            "5.06",
            "5.07",
            "5.08",
            "6.01",
            "6.02",
            "6.03",
            "6.04",
            "6.05",
            "7.01",
            "8.01",
            "9.01",
        ],
    }

    # Map string form types to FormType enum
    FORM_TYPE_MAP = {
        "10-K": FormType.FORM_10K,
        "10-Q": FormType.FORM_10Q,
        "8-K": FormType.FORM_8K,
    }

    ITEM_HEADING_REGEX = re.compile(
        r"(?im)^(item\s+([0-9]{1,2}[a-z]?)(?:\.([0-9]{2}))?)"
        r"[\s\.\-–—:]+(.*)$"
    )

    def __init__(self, min_section_length: int = 200) -> None:
        self.min_section_length = min_section_length

    def extract_sections_from_document(self, doc: SECDocument) -> None:
        """
        Extract sections from an SECDocument in place.
        Reads from doc.normalized_content for sections, detects form type
        from doc.text_content, and populates doc.form_type and doc.sections.

        Args:
            doc: SECDocument to process
        """
        content = doc.normalized_content
        if not content:
            return

        # Detect and set form type from text_content
        form_type_str = self.detect_form_type(doc.text_content or content)
        doc.form_type = self.FORM_TYPE_MAP.get(form_type_str, FormType.UNKNOWN)

        # Extract sections
        sections = self._extract_sections_internal(content, form_type_str)
        doc.sections.extend(sections)

    def _extract_sections_internal(
        self,
        content: str,
        form_type: str,
    ) -> List[FilingSection]:
        """
        Internal method to extract sections from content.

        Args:
            content: Normalized text content
            form_type: Form type string (e.g., "10-K")

        Returns:
            List of FilingSection objects
        """
        if not content:
            return []

        normalized_form = form_type.upper()
        allowed_items = self.FORM_ITEM_MAP.get(normalized_form)

        matches = list(self.ITEM_HEADING_REGEX.finditer(content))
        if not matches:
            return []

        sections: List[FilingSection] = []
        for idx, match in enumerate(matches):
            raw_item = match.group(2) or ""
            decimal_tail = match.group(3)
            item = raw_item.upper()
            if decimal_tail:
                item = f"{raw_item}.{decimal_tail}"
            title = (match.group(4) or "").strip()

            if allowed_items and item not in allowed_items:
                continue

            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            section_text = content[start:end].strip()

            if len(section_text) < self.min_section_length:
                continue

            sections.append(
                FilingSection(
                    item=item,
                    title=title,
                    content=section_text,
                    start=start,
                    end=end,
                )
            )

        return sections

    def detect_form_type(self, content: str) -> str:
        """
        Detect form type from filing content.

        Args:
            content: Filing content text

        Returns:
            Normalized form type string (e.g., "10-K", "10-Q", "8-K", "UNKNOWN")
        """
        if not content:
            return "UNKNOWN"

        # Check SEC header first
        header_match = re.search(
            r"(?im)conformed\s+submission\s+type:\s*([a-z0-9\-/]+)",
            content,
        )
        if header_match:
            form_str = header_match.group(1).upper()
            # Normalize variations (e.g., "10-K/A" -> "10-K")
            if "10-K" in form_str:
                return "10-K"
            if "10-Q" in form_str:
                return "10-Q"
            if "8-K" in form_str:
                return "8-K"
            return form_str

        # Check for "Form 10-K" pattern
        form_match = re.search(r"(?im)\bform\s+(10\-?k|10\-?q|8\-?k)\b", content)
        if form_match:
            form_str = form_match.group(1).upper().replace(" ", "")
            if "K" in form_str and "10" in form_str:
                return "10-K"
            if "Q" in form_str and "10" in form_str:
                return "10-Q"
            if "8" in form_str and "K" in form_str:
                return "8-K"

        # Check for annual/quarterly report mentions
        if re.search(r"(?i)\bannual\s+report\b", content):
            return "10-K"
        if re.search(r"(?i)\bquarterly\s+report\b", content):
            return "10-Q"
        if re.search(r"(?i)\bcurrent\s+report\b", content):
            return "8-K"

        return "UNKNOWN"

__all__ = ["SectionExtractor"]
