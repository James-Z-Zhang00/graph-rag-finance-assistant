from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import List

from sec.models import FileType, SECDocument


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
            if (
                self._chunks
                and self._chunks[-1]
                and not self._chunks[-1][-1].isspace()
                and not data[0].isspace()
            ):
                prev_char = self._chunks[-1][-1]
                next_char = data[0]
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
        content = doc.text_content
        if not content:
            return ""

        text = self._normalize_line_endings(content)
        text = self._remove_null_chars(text)
        text = self._remove_page_markers(text)
        text = self._clean_pdf_artifacts(text)
        text = self._remove_pdf_print_headers(text)
        text = self._remove_urls(text)

        if self.collapse_whitespace:
            text = self._collapse_whitespace(text)

        return text.strip()

    def _normalize_html(self, doc: SECDocument) -> str:
        content = doc.raw_content
        if not content:
            return ""

        text = self._normalize_line_endings(content)
        text = self._remove_null_chars(text)
        text = self._remove_sec_header(text)

        if self.strip_html:
            text = self._strip_html_tags(text)
            text = html.unescape(text)

        text = self._remove_xbrl_artifacts(text)

        if self.collapse_whitespace:
            text = self._collapse_whitespace(text)

        return text.strip()

    def _normalize_text(self, doc: SECDocument) -> str:
        content = doc.text_content or doc.raw_content
        if not content:
            return ""

        text = self._normalize_line_endings(content)
        text = self._remove_null_chars(text)
        text = self._remove_sec_header(text)
        text = self._remove_separator_lines(text)

        if self.collapse_whitespace:
            text = self._collapse_whitespace(text)

        return text.strip()

    # -------------------------------------------------------------------------
    # Common cleaning utilities
    # -------------------------------------------------------------------------

    def _remove_xbrl_artifacts(self, content: str) -> str:
        content = re.sub(
            r'https?://(?:www\.)?(?:fasb\.org|xbrl\.org|xbrl\.sec\.gov|sec\.gov)/[^\s]*',
            '', content
        )
        content = re.sub(r'https?://[^\s]+/\d{8}#[A-Za-z][A-Za-z0-9]*', '', content)
        content = re.sub(r'https?://[^\s]+#[A-Z][a-zA-Z0-9]+', '', content)
        content = re.sub(r'\b[A-Za-z][A-Za-z0-9-]*:[A-Za-z][A-Za-z0-9]*\b', '', content)
        content = re.sub(r'\bP\d+[YMWD](?:\d+[YMWD])*\b', '', content)
        content = re.sub(r'^\s*\d{4}-\d{2}-\d{2}\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*--\d{2}-\d{2}\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d{2}/\d{2}\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\b[a-z]{2,5}-\d{8}\b', '', content)
        content = re.sub(r'^\s*-[a-zA-Z]+\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(true|false)\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d{10}\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(Q[1-4]|FY)\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(19|20)\d{2}\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d{1,4}\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\s*$',
                        '', content, flags=re.MULTILINE)
        return content

    def _normalize_line_endings(self, content: str) -> str:
        return content.replace("\r\n", "\n").replace("\r", "\n")

    def _remove_null_chars(self, content: str) -> str:
        return content.replace("\x00", "")

    def _collapse_whitespace(self, content: str) -> str:
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content

    # -------------------------------------------------------------------------
    # HTML-specific utilities
    # -------------------------------------------------------------------------

    def _strip_html_tags(self, content: str) -> str:
        stripper = _HTMLStripper()
        stripper.feed(content)
        return stripper.get_text()

    def _remove_sec_header(self, content: str) -> str:
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
        content = re.sub(r"-{3,}\s*Page\s+(\d+)\s*-{3,}", r"\n[Page \1]\n", content)
        return content

    def _clean_pdf_artifacts(self, content: str) -> str:
        content = re.sub(r"\.{4,}", " ", content)
        content = content.replace("\f", "\n")
        return content

    def _remove_pdf_print_headers(self, content: str) -> str:
        content = re.sub(
            r'\d{1,2}/\d{1,2}/\d{2},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\s*\S*sec\.gov\S*',
            '', content, flags=re.IGNORECASE
        )
        content = re.sub(
            r'^\s*\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\s*$',
            '', content, flags=re.MULTILINE
        )
        return content

    def _remove_urls(self, content: str) -> str:
        content = re.sub(r'https?://[^\s]+', '', content)
        content = re.sub(r'sec\.gov/[^\s]+', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\bef\d+_\d*[a-z\-]+\.htm\b', '', content, flags=re.IGNORECASE)
        return content

    # -------------------------------------------------------------------------
    # Text-specific utilities
    # -------------------------------------------------------------------------

    def _remove_separator_lines(self, content: str) -> str:
        cleaned_lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue
            if re.fullmatch(r"[-_—–=]{3,}", stripped):
                continue
            line = re.sub(r"[_]{3,}", " ", line)
            line = re.sub(r"[-=]{3,}", " ", line)
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


__all__ = ["FilingNormalizer"]
