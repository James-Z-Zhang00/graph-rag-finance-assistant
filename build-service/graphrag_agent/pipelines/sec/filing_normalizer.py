from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import List

from graphrag_agent.pipelines.sec.models import FileType, SECDocument


class _HTMLStripper(HTMLParser):
    """
    HTML tag stripper that excludes XBRL tags and table elements.
    XBRL tags and tables are already handled by HTMLFilingParser.

    Inserts newlines at block-level element boundaries so that
    headings and paragraphs remain on separate lines (required
    by the section extractor's start-of-line regex).
    """

    # Tags to skip entirely (including their content)
    _SKIP_TAGS = {"table", "script", "style"}
    # XBRL tag prefixes to skip (content preserved, tag stripped)
    _XBRL_PREFIXES = ("ix:", "xbrli:", "xbrldi:")
    # Block-level tags that should produce a line break
    _BLOCK_TAGS = {
        "div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "ul", "ol", "blockquote", "pre", "section",
        "article", "header", "footer", "nav", "aside",
        "hr", "br", "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def _emit_newline(self) -> None:
        """Append a newline only if the last chunk doesn't already end with one."""
        if self._skip_depth == 0:
            if not self._chunks or not self._chunks[-1].endswith("\n"):
                self._chunks.append("\n")

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS:
            self._skip_depth += 1
        # XBRL context/unit definition tags — skip their content
        if tag_lower in ("context", "xbrli:context", "unit", "xbrli:unit"):
            self._skip_depth += 1
        if tag_lower in self._BLOCK_TAGS:
            self._emit_newline()

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag_lower in ("context", "xbrli:context", "unit", "xbrli:unit"):
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag_lower in self._BLOCK_TAGS:
            self._emit_newline()

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            # Insert a space between adjacent inline text chunks when
            # one side ends/starts at a word boundary (digit, punctuation,
            # or closing bracket). This handles cases like:
            #   <span>Item 8.01</span><span>Other Events.</span>
            #     → "Item 8.01 Other Events."  (space after digit)
            # but avoids splitting words broken across spans:
            #   <span>FINANC</span><span>IAL</span>
            #     → "FINANCIAL"  (both sides are mid-word letters)
            if (
                self._chunks
                and self._chunks[-1]
                and not self._chunks[-1][-1].isspace()
                and not data[0].isspace()
            ):
                prev_char = self._chunks[-1][-1]
                next_char = data[0]
                # Insert space only when at least one side is at a word
                # boundary (non-letter), so mid-word letter joins stay glued
                if not prev_char.isalpha() or not next_char.isalpha():
                    self._chunks.append(" ")
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


class FilingNormalizer:
    """
    Normalize SEC filing content into clean text for parsing/chunking.
    Provides differentiated logic for PDF, HTML/HTM, XBRL, and text files.
    """

    def __init__(
        self,
        collapse_whitespace: bool = True,
        strip_html: bool = True,
    ) -> None:
        self.collapse_whitespace = collapse_whitespace
        self.strip_html = strip_html

    def normalize_document(self, doc: SECDocument) -> None:
        """
        Normalize an SECDocument in place.
        Updates doc.normalized_content based on file type.

        Args:
            doc: SECDocument to normalize
        """
        if doc.file_type == FileType.PDF:
            doc.normalized_content = self._normalize_pdf(doc)
        elif doc.file_type in {FileType.HTML, FileType.HTM, FileType.XBRL}:
            doc.normalized_content = self._normalize_html(doc)
        else:
            doc.normalized_content = self._normalize_text(doc)

    def _normalize_pdf(self, doc: SECDocument) -> str:
        """
        Normalize PDF content.
        PDF text has already been extracted separately from tables by the reader.

        Args:
            doc: SECDocument with PDF content

        Returns:
            Normalized text content
        """
        content = doc.text_content
        if not content:
            return ""

        # Basic cleaning
        text = self._normalize_line_endings(content)
        text = self._remove_null_chars(text)

        # PDF-specific cleaning
        text = self._remove_page_markers(text)
        text = self._clean_pdf_artifacts(text)
        text = self._remove_pdf_print_headers(text)

        # Remove URLs (common in PDFs printed from web)
        text = self._remove_urls(text)

        # Common cleaning
        if self.collapse_whitespace:
            text = self._collapse_whitespace(text)

        return text.strip()

    def _normalize_html(self, doc: SECDocument) -> str:
        """
        Normalize HTML/HTM content to clean text.

        XBRL tags and tables are already extracted by HTMLFilingParser in stage 1.
        This method only strips remaining HTML tags and cleans up the text.

        Args:
            doc: SECDocument with HTML content

        Returns:
            Normalized text content
        """
        content = doc.raw_content
        if not content:
            return ""

        # Basic cleaning
        text = self._normalize_line_endings(content)
        text = self._remove_null_chars(text)

        # Remove SEC header blocks
        text = self._remove_sec_header(text)

        # Strip HTML tags (skips tables and XBRL context/unit definitions)
        if self.strip_html:
            text = self._strip_html_tags(text)
            text = html.unescape(text)

        # Remove XBRL taxonomy artifacts from text
        text = self._remove_xbrl_artifacts(text)

        # Common cleaning
        if self.collapse_whitespace:
            text = self._collapse_whitespace(text)

        return text.strip()

    def _normalize_text(self, doc: SECDocument) -> str:
        """
        Normalize plain text content (TXT, MD, etc.).

        Args:
            doc: SECDocument with text content

        Returns:
            Normalized text content
        """
        content = doc.text_content or doc.raw_content
        if not content:
            return ""

        # Basic cleaning
        text = self._normalize_line_endings(content)
        text = self._remove_null_chars(text)

        # Text-specific cleaning
        text = self._remove_sec_header(text)
        text = self._remove_separator_lines(text)

        # Common cleaning
        if self.collapse_whitespace:
            text = self._collapse_whitespace(text)

        return text.strip()

    # -------------------------------------------------------------------------
    # Common cleaning utilities
    # -------------------------------------------------------------------------

    def _remove_xbrl_artifacts(self, content: str) -> str:
        """
        Remove XBRL/iXBRL metadata artifacts that have no semantic value.

        Inline XBRL (iXBRL) embeds machine-readable data directly in HTML files.
        This method removes those artifacts while preserving actual document content.
        """
        # --- URLs ---
        # Standard XBRL taxonomy URLs
        content = re.sub(
            r'https?://(?:www\.)?(?:fasb\.org|xbrl\.org|xbrl\.sec\.gov|sec\.gov)/[^\s]*',
            '', content
        )
        # Company extension URLs (domain/YYYYMMDD#ConceptName)
        content = re.sub(r'https?://[^\s]+/\d{8}#[A-Za-z][A-Za-z0-9]*', '', content)
        # Any URL with XBRL-style fragment (#CamelCaseConcept)
        content = re.sub(r'https?://[^\s]+#[A-Z][a-zA-Z0-9]+', '', content)

        # --- XBRL namespace prefixes ---
        # iso4217:USD, dei:DocumentType, us-gaap:Revenue, CELU:ClassName...
        content = re.sub(r'\b[A-Za-z][A-Za-z0-9-]*:[A-Za-z][A-Za-z0-9]*\b', '', content)

        # --- Dates and durations ---
        # ISO 8601 durations: P3Y, P5M, P1D, P1Y6M, etc.
        content = re.sub(r'\bP\d+[YMWD](?:\d+[YMWD])*\b', '', content)
        # Standalone ISO dates (YYYY-MM-DD on their own line)
        content = re.sub(r'^\s*\d{4}-\d{2}-\d{2}\s*$', '', content, flags=re.MULTILINE)
        # Fiscal year end pattern (--MM-DD)
        content = re.sub(r'^\s*--\d{2}-\d{2}\s*$', '', content, flags=re.MULTILINE)
        # Standalone date fragments (MM/DD format on their own line)
        content = re.sub(r'^\s*\d{2}/\d{2}\s*$', '', content, flags=re.MULTILINE)

        # --- XBRL identifiers ---
        # Document identifiers: tsla-20251231, aapl-20251228, etc.
        content = re.sub(r'\b[a-z]{2,5}-\d{8}\b', '', content)
        # XBRL period modifiers: -overlappingPeriod, -adjustedBalance, etc.
        content = re.sub(r'^\s*-[a-zA-Z]+\s*$', '', content, flags=re.MULTILINE)

        # --- Standalone values (on their own line) ---
        # Booleans
        content = re.sub(r'^\s*(true|false)\s*$', '', content, flags=re.MULTILINE)
        # CIK numbers (10-digit)
        content = re.sub(r'^\s*\d{10}\s*$', '', content, flags=re.MULTILINE)
        # Fiscal period indicators (Q1-Q4, FY)
        content = re.sub(r'^\s*(Q[1-4]|FY)\s*$', '', content, flags=re.MULTILINE)
        # Year numbers (4-digit)
        content = re.sub(r'^\s*(19|20)\d{2}\s*$', '', content, flags=re.MULTILINE)
        # Small standalone numbers (1-4 digits, likely XBRL context counts)
        content = re.sub(r'^\s*\d{1,4}\s*$', '', content, flags=re.MULTILINE)
        # Standalone word "One", "Two", etc. (XBRL text values)
        content = re.sub(r'^\s*(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\s*$',
                        '', content, flags=re.MULTILINE)

        return content

    def _normalize_line_endings(self, content: str) -> str:
        """Normalize line endings to Unix style."""
        return content.replace("\r\n", "\n").replace("\r", "\n")

    def _remove_null_chars(self, content: str) -> str:
        """Remove null characters."""
        return content.replace("\x00", "")

    def _collapse_whitespace(self, content: str) -> str:
        """Collapse multiple spaces and newlines."""
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content

    # -------------------------------------------------------------------------
    # HTML-specific utilities
    # -------------------------------------------------------------------------

    def _strip_html_tags(self, content: str) -> str:
        """
        Strip HTML tags, extracting text content.
        Skips table elements and XBRL context/unit definitions
        (already handled by HTMLFilingParser).
        """
        stripper = _HTMLStripper()
        stripper.feed(content)
        return stripper.get_text()

    def _remove_sec_header(self, content: str) -> str:
        """Remove SEC-HEADER blocks."""
        return re.sub(
            r"<SEC-HEADER>.*?</SEC-HEADER>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )

    # -------------------------------------------------------------------------
    # PDF-specific utilities
    # -------------------------------------------------------------------------

    def _remove_page_markers(self, content: str) -> str:
        """Remove or normalize page markers from PDF extraction."""
        # Keep page markers but normalize format
        content = re.sub(r"-{3,}\s*Page\s+(\d+)\s*-{3,}", r"\n[Page \1]\n", content)
        return content

    def _clean_pdf_artifacts(self, content: str) -> str:
        """Clean common PDF extraction artifacts."""
        # Remove excessive dots (often from table of contents)
        content = re.sub(r"\.{4,}", " ", content)
        # Remove form feed characters
        content = content.replace("\f", "\n")
        return content

    def _remove_pdf_print_headers(self, content: str) -> str:
        """
        Remove browser print headers/footers from PDFs.

        When PDFs are created via "Print to PDF" from a browser, they often
        contain headers like "1/29/26, 11:23 AM sec.gov/Archives/edgar/..."
        """
        # Pattern: MM/DD/YY, HH:MM AM/PM followed by URL
        content = re.sub(
            r'\d{1,2}/\d{1,2}/\d{2},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\s*\S*sec\.gov\S*',
            '', content, flags=re.IGNORECASE
        )
        # Pattern: just date/time headers at start of lines
        content = re.sub(
            r'^\s*\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\s*$',
            '', content, flags=re.MULTILINE
        )
        return content

    def _remove_urls(self, content: str) -> str:
        """
        Remove URLs from content.

        Removes:
        - Full URLs (http://, https://)
        - SEC EDGAR URLs without protocol (sec.gov/Archives/...)
        - Common file paths that look like URLs
        """
        # Full URLs with protocol
        content = re.sub(r'https?://[^\s]+', '', content)
        # SEC EDGAR URLs without protocol
        content = re.sub(r'sec\.gov/[^\s]+', '', content, flags=re.IGNORECASE)
        # Common SEC file patterns (efXXXXXXXX_8k.htm, etc.)
        content = re.sub(r'\bef\d+_\d*[a-z\-]+\.htm\b', '', content, flags=re.IGNORECASE)
        return content

    # -------------------------------------------------------------------------
    # Text-specific utilities
    # -------------------------------------------------------------------------

    def _remove_separator_lines(self, content: str) -> str:
        """Remove decorative separator lines."""
        cleaned_lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue
            # Skip lines that are only separators
            if re.fullmatch(r"[-_—–=]{3,}", stripped):
                continue
            # Clean separators within lines
            line = re.sub(r"[_]{3,}", " ", line)
            line = re.sub(r"[-=]{3,}", " ", line)
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


__all__ = ["FilingNormalizer"]
