"""
XBRL-aware HTML parser for SEC filings using BeautifulSoup.

Multi-pass DOM parser that processes HTML/HTM documents to extract:
- XBRL context and unit definitions
- ExtractedTable with XBRL-enriched TableCell objects (colspan/rowspan aware)
- XBRLNumeric facts from narrative text (ix:nonFraction tags outside tables)

For non-HTML files (PDF, TXT), this parser is not used.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from graphrag_agent.pipelines.sec.models import (
    ExtractedTable,
    SECDocument,
    TableCell,
    XBRLNumeric,
)

logger = logging.getLogger(__name__)


class _TableVerdict(Enum):
    """Classification result for a table element."""

    DATA = "data"
    LAYOUT = "layout"
    UNCERTAIN = "uncertain"


class HTMLFilingParser:
    """
    BeautifulSoup-based XBRL-aware parser for HTML/HTM SEC filings.

    Processes an HTML document in multiple passes over the DOM to produce:
    - XBRLNumeric facts from narrative text (outside tables)
    - ExtractedTable objects with XBRL-enriched TableCell objects

    Supports nested tables, colspan/rowspan, and proper XBRL context resolution.

    Text normalization (HTML stripping, cleanup) is handled separately
    by FilingNormalizer.

    Not used for non-HTML files (PDF, TXT). Those are handled
    by their respective readers.
    """

    def parse_document(self, doc: SECDocument) -> None:
        """
        Parse an HTML/HTM SEC document using BeautifulSoup.

        Populates doc.numeric_facts and doc.tables.
        Text normalization is handled separately by FilingNormalizer.

        Args:
            doc: SECDocument with raw_content populated (HTML/HTM only)
        """
        if not doc.raw_content:
            return

        soup = BeautifulSoup(doc.raw_content, "html.parser")

        # Pass 1: Build context and unit lookup dicts
        contexts = self._parse_contexts(soup)
        units = self._parse_units(soup)

        # Pass 2: Extract tables with XBRL-enriched cells
        data_tables, layout_texts, layout_xbrl = self._parse_tables(
            soup, contexts, units
        )
        doc.tables = data_tables

        # Append text extracted from layout tables
        if layout_texts:
            layout_text = "\n".join(layout_texts)
            if doc.text_content:
                doc.text_content += "\n" + layout_text
            else:
                doc.text_content = layout_text

        # Pass 3: Collect XBRL facts from narrative (outside tables)
        doc.numeric_facts = self._parse_narrative_xbrl(soup, contexts, units)

        # Merge in XBRL facts found inside layout tables
        doc.numeric_facts.extend(layout_xbrl)

        # Pass 4: Harvest XBRL facts from data table cells (income statement,
        # balance sheet, etc.). These are stored in cell.xbrl but were never
        # promoted to doc.numeric_facts — causing revenue/segment facts to be missing.
        for table in data_tables:
            for cell in table.cells:
                if cell.xbrl is not None:
                    doc.numeric_facts.append(cell.xbrl)

    # ------------------------------------------------------------------
    # Pass 1: Context definitions
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_contexts(soup: BeautifulSoup) -> Dict[str, Dict[str, Any]]:
        """Parse <context> / <xbrli:context> elements into a lookup dict."""
        contexts: Dict[str, Dict[str, Any]] = {}

        for ctx_tag in soup.find_all(["context", "xbrli:context"]):
            ctx_id = ctx_tag.get("id")
            if not ctx_id:
                continue

            data: Dict[str, Any] = {}

            # Entity
            entity_tag = ctx_tag.find(["entity", "xbrli:entity"])
            if entity_tag:
                ident = entity_tag.find(["identifier", "xbrli:identifier"])
                if ident:
                    data["entity"] = ident.get_text(strip=True)

                # Segment dimensions
                segment_tag = entity_tag.find(["segment", "xbrli:segment"])
                if segment_tag:
                    segment: Dict[str, str] = {}
                    for member in segment_tag.find_all(True):
                        dimension = member.get("dimension")
                        if dimension:
                            segment[dimension] = member.get_text(strip=True)
                    if segment:
                        data["segment"] = segment

            # Period
            period_tag = ctx_tag.find(["period", "xbrli:period"])
            if period_tag:
                for tag_name, key in [
                    ("startdate", "period_start"),
                    ("xbrli:startdate", "period_start"),
                    ("enddate", "period_end"),
                    ("xbrli:enddate", "period_end"),
                    ("instant", "period_end"),
                    ("xbrli:instant", "period_end"),
                ]:
                    el = period_tag.find(tag_name)
                    if el:
                        data[key] = el.get_text(strip=True)

            contexts[ctx_id] = data

        return contexts

    # ------------------------------------------------------------------
    # Pass 1: Unit definitions
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_units(soup: BeautifulSoup) -> Dict[str, str]:
        """Parse <unit> / <xbrli:unit> elements into a lookup dict."""
        units: Dict[str, str] = {}

        for unit_tag in soup.find_all(["unit", "xbrli:unit"]):
            unit_id = unit_tag.get("id")
            if not unit_id:
                continue
            measure = unit_tag.find(["measure", "xbrli:measure"])
            if measure:
                units[unit_id] = measure.get_text(strip=True)

        return units

    # ------------------------------------------------------------------
    # Pass 2: Table extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_table_dom(table_tag: Tag) -> _TableVerdict:
        """
        Phase 1: DOM-level classification of a <table> tag.

        Checks structural signals before any row collection to quickly
        identify obvious layout or data tables.
        """
        # 1. role="presentation" or role="none" → LAYOUT
        role = (table_tag.get("role") or "").lower()
        if role in ("presentation", "none"):
            return _TableVerdict.LAYOUT

        # 2. Contains ix:nonfraction descendant → DATA
        if table_tag.find("ix:nonfraction"):
            return _TableVerdict.DATA

        # 3. Style-hidden table → LAYOUT
        style = (table_tag.get("style") or "").lower().replace(" ", "")
        if ("display:none" in style
                or "visibility:hidden" in style
                or "visibility:collapse" in style):
            return _TableVerdict.LAYOUT

        # 4. Nested inside another <table> → LAYOUT
        if table_tag.find_parent("table"):
            return _TableVerdict.LAYOUT

        return _TableVerdict.UNCERTAIN

    @staticmethod
    def _classify_table_content(table_tag: Tag, rows: List[Tag]) -> _TableVerdict:
        """
        Phase 2: Content-density classification on collected rows.

        Only called when Phase 1 returns UNCERTAIN.
        """
        # 1. Contains any ix:nonfraction or ix:nonnumeric → DATA
        if table_tag.find(["ix:nonfraction", "ix:nonnumeric"]):
            return _TableVerdict.DATA

        # Collect visible rows (filter out hidden rows)
        visible_rows: List[Tag] = []
        for tr in rows:
            style = (tr.get("style") or "").lower().replace(" ", "")
            if "visibility:collapse" in style or "display:none" in style:
                continue
            visible_rows.append(tr)

        # 2. 0 visible rows → LAYOUT
        if not visible_rows:
            return _TableVerdict.LAYOUT

        # 3. Single visible row → LAYOUT
        if len(visible_rows) == 1:
            return _TableVerdict.LAYOUT

        # Gather cell stats from visible rows
        total_cells = 0
        non_empty_cells = 0
        empty_rows = 0
        # Track per-row non-empty column counts for single-column check
        rows_with_single_nonempty = 0

        for tr in visible_rows:
            cell_tags = [
                c for c in tr.children
                if isinstance(c, Tag) and c.name in ("td", "th")
            ]
            row_non_empty = 0
            for cell in cell_tags:
                total_cells += 1
                text = cell.get_text(strip=True).replace("\xa0", "")
                if text:
                    non_empty_cells += 1
                    row_non_empty += 1
            if row_non_empty == 0:
                empty_rows += 1
            if row_non_empty <= 1:
                rows_with_single_nonempty += 1

        # 4. All cells empty → LAYOUT
        if non_empty_cells == 0:
            return _TableVerdict.LAYOUT

        # 5. Spacer-row dominated: >40% of visible rows entirely empty → LAYOUT
        num_visible = len(visible_rows)
        if num_visible >= 3 and empty_rows / num_visible > 0.40:
            return _TableVerdict.LAYOUT

        # 6. Content density < 0.40 AND total_cells >= 6 → LAYOUT
        if total_cells >= 6:
            density = non_empty_cells / total_cells
            if density < 0.40:
                return _TableVerdict.LAYOUT

        # 7. Single effective column (>80% rows have ≤1 non-empty cell) AND ≥3 rows
        if num_visible >= 3:
            if rows_with_single_nonempty / num_visible > 0.80:
                return _TableVerdict.LAYOUT

        return _TableVerdict.DATA

    def _extract_layout_table(
        self,
        table_tag: Tag,
        contexts: Dict[str, Dict[str, Any]],
        units: Dict[str, str],
        layout_texts: List[str],
        layout_xbrl: List[XBRLNumeric],
    ) -> None:
        """Extract text and XBRL facts from a layout table."""
        # Extract plain text, collapsing whitespace
        text = table_tag.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            layout_texts.append(text)

        # Extract XBRL facts
        for xbrl_tag in table_tag.find_all(["ix:nonfraction", "ix:nonnumeric"]):
            raw_text = xbrl_tag.get_text(strip=True)
            numeric = self._build_xbrl_numeric(
                dict(xbrl_tag.attrs), raw_text, contexts, units
            )
            if numeric:
                layout_xbrl.append(numeric)

    def _parse_tables(
        self,
        soup: BeautifulSoup,
        contexts: Dict[str, Dict[str, Any]],
        units: Dict[str, str],
    ) -> Tuple[List[ExtractedTable], List[str], List[XBRLNumeric]]:
        """Extract data tables; collect text and XBRL from layout tables."""
        tables: List[ExtractedTable] = []
        layout_texts: List[str] = []
        layout_xbrl: List[XBRLNumeric] = []
        table_counter = 0

        for table_tag in soup.find_all("table"):
            table_counter += 1

            # Phase 1: DOM-level classification
            verdict = self._classify_table_dom(table_tag)
            if verdict == _TableVerdict.LAYOUT:
                logger.debug("Table %d layout (DOM)", table_counter)
                self._extract_layout_table(
                    table_tag, contexts, units, layout_texts, layout_xbrl
                )
                continue

            # Collect rows that belong directly to this table (not nested)
            rows = self._collect_table_rows(table_tag)
            if not rows:
                continue

            # Phase 2: Content classification (only if UNCERTAIN)
            if verdict == _TableVerdict.UNCERTAIN:
                verdict = self._classify_table_content(table_tag, rows)
                if verdict == _TableVerdict.LAYOUT:
                    logger.debug(
                        "Table %d layout (content)", table_counter
                    )
                    self._extract_layout_table(
                        table_tag, contexts, units, layout_texts, layout_xbrl
                    )
                    continue

            # Build grid with colspan/rowspan expansion
            grid = self._build_grid(rows, contexts, units)
            if not grid:
                continue

            # Extract caption
            caption_tag = table_tag.find("caption", recursive=False)
            # Also check inside thead
            if not caption_tag:
                thead = table_tag.find("thead", recursive=False)
                if thead:
                    caption_tag = thead.find("caption", recursive=False)
            caption = caption_tag.get_text(strip=True) if caption_tag else None

            # Build column/row headers and TableCell objects
            column_headers = [cell[0] for cell in grid[0]] if grid else []
            row_headers = [
                row[0][0] for row in grid[1:] if row
            ]

            cells: List[TableCell] = []
            for row_idx, row in enumerate(grid):
                for col_idx, (cell_value, cell_xbrl) in enumerate(row):
                    col_header = (
                        column_headers[col_idx]
                        if row_idx > 0 and col_idx < len(column_headers)
                        else None
                    )
                    row_header = (
                        row[0][0] if row_idx > 0 and col_idx > 0 and row
                        else None
                    )
                    cells.append(
                        TableCell(
                            value=cell_value,
                            row=row_idx,
                            col=col_idx,
                            column_header=col_header,
                            row_header=row_header,
                            xbrl=cell_xbrl,
                        )
                    )

            tables.append(
                ExtractedTable(
                    table_id=f"table_{table_counter}",
                    cells=cells,
                    caption=caption,
                    source="html",
                    column_headers=column_headers,
                    row_headers=row_headers,
                )
            )

        return tables, layout_texts, layout_xbrl

    @staticmethod
    def _collect_table_rows(table_tag: Tag) -> List[Tag]:
        """
        Collect <tr> elements that belong directly to this table,
        excluding rows inside nested sub-tables.
        """
        rows: List[Tag] = []
        # Look inside thead/tbody/tfoot wrappers and direct children
        for child in table_tag.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "tr":
                rows.append(child)
            elif child.name in ("thead", "tbody", "tfoot"):
                for sub in child.children:
                    if isinstance(sub, Tag) and sub.name == "tr":
                        rows.append(sub)
        return rows

    def _build_grid(
        self,
        rows: List[Tag],
        contexts: Dict[str, Dict[str, Any]],
        units: Dict[str, str],
    ) -> List[List[Tuple[str, Optional[XBRLNumeric]]]]:
        """
        Build a 2D grid from <tr> rows, expanding colspan/rowspan.

        Each grid cell is (text, xbrl_or_none).
        """
        # First pass: determine grid dimensions
        num_rows = len(rows)
        if num_rows == 0:
            return []

        # We'll use a dynamic grid with slot tracking for rowspan
        # occupied[row][col] = True if already filled by a rowspan from above
        occupied: Dict[int, Dict[int, bool]] = {}
        grid: List[List[Tuple[str, Optional[XBRLNumeric]]]] = []

        for row_idx, tr in enumerate(rows):
            grid_row: List[Tuple[str, Optional[XBRLNumeric]]] = []
            col_idx = 0

            # Get cells (td/th) that are direct children of this tr
            cell_tags = [
                c for c in tr.children
                if isinstance(c, Tag) and c.name in ("td", "th")
            ]

            for cell_tag in cell_tags:
                # Skip columns occupied by rowspan from above
                while occupied.get(row_idx, {}).get(col_idx, False):
                    grid_row.append(("", None))
                    col_idx += 1

                # Extract cell content
                cell_text, cell_xbrl = self._extract_cell_content(
                    cell_tag, contexts, units
                )

                colspan = int(cell_tag.get("colspan", 1))
                rowspan = int(cell_tag.get("rowspan", 1))

                # Place the cell value in the first slot
                grid_row.append((cell_text, cell_xbrl))

                # Mark additional colspan slots as empty
                for c in range(1, colspan):
                    grid_row.append(("", None))

                # Mark rowspan slots as occupied for future rows
                if rowspan > 1:
                    for r in range(1, rowspan):
                        future_row = row_idx + r
                        if future_row not in occupied:
                            occupied[future_row] = {}
                        for c in range(colspan):
                            occupied[future_row][col_idx + c] = True

                col_idx += colspan

            # Fill any remaining occupied columns at the end
            while occupied.get(row_idx, {}).get(col_idx, False):
                grid_row.append(("", None))
                col_idx += 1

            grid.append(grid_row)

        return grid

    def _extract_cell_content(
        self,
        cell_tag: Tag,
        contexts: Dict[str, Dict[str, Any]],
        units: Dict[str, str],
    ) -> Tuple[str, Optional[XBRLNumeric]]:
        """
        Extract text and XBRL data from a <td>/<th> cell.

        Returns (display_text, xbrl_numeric_or_none).
        """
        # Check for XBRL tags inside the cell
        xbrl_tag = cell_tag.find(["ix:nonfraction", "ix:nonnumeric"])
        cell_xbrl: Optional[XBRLNumeric] = None

        if xbrl_tag:
            raw_text = xbrl_tag.get_text(strip=True)
            cell_xbrl = self._build_xbrl_numeric(
                dict(xbrl_tag.attrs), raw_text, contexts, units
            )

        # Full cell text (includes XBRL display text naturally)
        cell_text = cell_tag.get_text(strip=True)
        return cell_text, cell_xbrl

    # ------------------------------------------------------------------
    # Pass 3: Narrative XBRL facts (outside tables)
    # ------------------------------------------------------------------

    def _parse_narrative_xbrl(
        self,
        soup: BeautifulSoup,
        contexts: Dict[str, Dict[str, Any]],
        units: Dict[str, str],
    ) -> List[XBRLNumeric]:
        """Collect XBRL facts from ix:nonFraction / ix:nonNumeric tags not inside tables."""
        facts: List[XBRLNumeric] = []

        for xbrl_tag in soup.find_all(["ix:nonfraction", "ix:nonnumeric"]):
            # Skip if inside a table
            if xbrl_tag.find_parent("table"):
                continue

            raw_text = xbrl_tag.get_text(strip=True)
            numeric = self._build_xbrl_numeric(
                dict(xbrl_tag.attrs), raw_text, contexts, units
            )
            if numeric:
                facts.append(numeric)

        return facts

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_xbrl_numeric(
        attrs: Dict[str, str],
        raw_text: str,
        contexts: Dict[str, Dict[str, Any]],
        units: Dict[str, str],
    ) -> Optional[XBRLNumeric]:
        """Build an XBRLNumeric from tag attributes and display text."""
        name = attrs.get("name")
        context_ref = attrs.get("contextref", "")

        if not name:
            return None

        # Parse the raw display number
        raw_number = HTMLFilingParser._parse_display_number(raw_text)
        if raw_number is None:
            return None

        # Apply scale
        scale = int(attrs.get("scale", "0"))
        value = raw_number * (10 ** scale)

        # Apply sign
        sign = attrs.get("sign")
        if sign == "-":
            value = -value

        # Parse decimals
        decimals_str = attrs.get("decimals")
        decimals = None
        if decimals_str and decimals_str.lower() != "inf":
            try:
                decimals = int(decimals_str)
            except ValueError:
                pass

        # Resolve context
        context_data = contexts.get(context_ref, {})
        entity = context_data.get("entity")
        period_start = context_data.get("period_start")
        period_end = context_data.get("period_end")
        segment = context_data.get("segment")

        # Resolve unit
        unit_ref = attrs.get("unitref", "")
        unit = HTMLFilingParser._resolve_unit(unit_ref, units)

        return XBRLNumeric(
            value=value,
            raw=raw_text,
            name=name,
            context_ref=context_ref,
            entity=entity,
            period_start=period_start,
            period_end=period_end,
            segment=segment,
            unit_ref=unit_ref if unit_ref else None,
            unit=unit,
            decimals=decimals,
            scale=scale,
            sign=sign,
            format=attrs.get("format"),
            fact_id=attrs.get("id"),
        )

    @staticmethod
    def _resolve_unit(unit_ref: str, units: Dict[str, str]) -> Optional[str]:
        """Resolve a unitRef to a human-readable unit string."""
        if not unit_ref:
            return None
        raw_unit = units.get(unit_ref)
        if not raw_unit:
            return None
        # Strip namespace prefix (e.g., "iso4217:USD" → "USD")
        if ":" in raw_unit:
            return raw_unit.split(":")[-1]
        return raw_unit

    @staticmethod
    def _parse_display_number(text: str) -> Optional[float]:
        """Parse a display number string like '1,500' or '(3.42)' to float."""
        if not text:
            return None
        # Remove currency symbols, whitespace
        cleaned = re.sub(r'[$€£\s]', '', text)
        # Handle parentheses as negative (accounting convention)
        negative = False
        if cleaned.startswith('(') and cleaned.endswith(')'):
            negative = True
            cleaned = cleaned[1:-1]
        # Strip trailing footnote markers e.g. (1), (a), *, †
        cleaned = re.sub(r'\([0-9a-zA-Z]+\)$', '', cleaned)
        cleaned = re.sub(r'[*†‡§#]+$', '', cleaned)
        # Remove commas
        cleaned = cleaned.replace(',', '')
        # Handle en-dash/em-dash as zero (common in SEC filings for nil values)
        if cleaned in ('—', '–', '-', ''):
            return 0.0
        try:
            value = float(cleaned)
            return -value if negative else value
        except ValueError:
            return None


__all__ = ["HTMLFilingParser"]
