import codecs
import os
from typing import Any, List, Optional, Tuple

from graphrag_agent.pipelines.sec.models import (
    ExtractedTable,
    FileType,
    SECDocument,
    TableCell,
)


import pdfplumber
from img2table.document import PDF as Img2TablePDF
import camelot

from graphrag_agent.pipelines.sec.html_filing_parser import HTMLFilingParser



# Map file extensions to FileType enum
EXTENSION_TO_FILETYPE = {
    ".pdf": FileType.PDF,
    ".html": FileType.HTML,
    ".htm": FileType.HTM,
    ".xbrl": FileType.XBRL,
    ".txt": FileType.TXT,
    ".md": FileType.MD,
}


class SecFileReader:
    """
    SEC-specific file reader that reads files into SECDocument objects.
    Supports PDF (with table extraction), HTML, HTM, XBRL, and text files.
    """

    def __init__(
        self,
        directory_path: str,
        html_filing_parser: Optional[HTMLFilingParser] = None,
        pdf_table_method: str = "img2table",
    ) -> None:
        """
        Args:
            directory_path: Path to the directory containing SEC filings
            html_filing_parser: Parser for HTML/HTM files
            pdf_table_method: PDF table extraction method - "img2table" or "camelot"
        """
        self.directory_path = directory_path
        self.html_filing_parser = html_filing_parser or HTMLFilingParser()
        self.pdf_table_method = pdf_table_method

    def read_files(
        self,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[SECDocument]:
        """
        Read files with specified extensions into SECDocument objects.

        Args:
            file_extensions: List of file extensions; if None, use SEC defaults
            recursive: Whether to read subdirectories recursively

        Returns:
            List of SECDocument objects
        """
        if file_extensions is None:
            file_extensions = [".txt", ".pdf", ".md", ".html", ".htm", ".xbrl"]

        results: List[SECDocument] = []

        # Collect all file paths first
        file_paths = self._collect_file_paths(file_extensions, recursive)

        # Process each file
        for rel_path, abs_path, extension in file_paths:
            doc = self._read_single_file(rel_path, abs_path, extension)
            results.append(doc)

        return results

    def _collect_file_paths(
        self,
        file_extensions: List[str],
        recursive: bool,
    ) -> List[Tuple[str, str, str]]:
        """
        Collect all file paths matching the extensions.

        Returns:
            List of (relative_path, absolute_path, extension) tuples
        """
        file_paths: List[Tuple[str, str, str]] = []
        extensions_set = set(file_extensions)

        if recursive:
            for root, _, filenames in os.walk(self.directory_path):
                for filename in filenames:
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in extensions_set:
                        abs_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(abs_path, self.directory_path)
                        file_paths.append((rel_path, abs_path, file_ext))
        else:
            for filename in os.listdir(self.directory_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in extensions_set:
                    abs_path = os.path.join(self.directory_path, filename)
                    file_paths.append((filename, abs_path, file_ext))

        return file_paths

    def _read_single_file(
        self,
        rel_path: str,
        abs_path: str,
        extension: str,
    ) -> SECDocument:
        """
        Read a single file into an SECDocument object.

        Args:
            rel_path: Relative path to the file
            abs_path: Absolute path to the file
            extension: File extension

        Returns:
            SECDocument object with raw content populated
        """
        filename = os.path.basename(rel_path)
        file_type = EXTENSION_TO_FILETYPE.get(extension, FileType.TXT)

        # Create the document
        doc = SECDocument(
            file_path=rel_path,
            filename=filename,
            extension=extension,
            file_type=file_type,
        )

        # Read based on file type
        if extension == ".pdf":
            self._read_pdf_into_document(abs_path, doc)
        elif extension in {".html", ".htm"}:
            self._read_html_into_document(abs_path, doc)
        elif extension == ".xbrl":
            self._read_xbrl_into_document(abs_path, doc)
        else:
            self._read_text_into_document(abs_path, doc)

        return doc

    def _read_pdf_into_document(self, file_path: str, doc: SECDocument) -> None:
        """
        Read PDF file and populate SECDocument with text and tables separately.

        Table extraction uses either img2table or camelot based on pdf_table_method.
        Text extraction always uses pdfplumber.

        Args:
            file_path: Path to the PDF file
            doc: SECDocument to populate
        """
        # --- Table extraction ---
        table_bboxes_by_page: dict = {}

        if self.pdf_table_method == "camelot":
            table_bboxes_by_page = self._extract_tables_camelot(file_path, doc)
        else:
            table_bboxes_by_page = self._extract_tables_img2table(file_path, doc)

        # --- Text extraction via pdfplumber ---
        try:
            text_parts: List[str] = []
            failed_pages: List[int] = []

            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        page_idx = page_num - 1
                        table_bboxes = table_bboxes_by_page.get(page_idx, [])

                        if table_bboxes:
                            try:
                                text = self._extract_text_outside_tables(page, table_bboxes)
                            except Exception:
                                # Fallback: extract full page text if char-level
                                # filtering fails (e.g. pdfplumber compat issues)
                                text = page.extract_text() or ""
                        else:
                            text = page.extract_text() or ""

                        if text and text.strip():
                            text_parts.append(f"--- Page {page_num} ---\n{text}")
                    except Exception:
                        failed_pages.append(page_num)

            doc.text_content = "\n\n".join(text_parts)
            doc.raw_content = doc.text_content

            if failed_pages:
                doc.processing_errors.append(
                    f"PDF text extraction failed on page(s): {failed_pages}"
                )

        except Exception as e:
            doc.processing_errors.append(f"PDF text extraction failed: {e}")

    def _extract_tables_img2table(
        self, file_path: str, doc: SECDocument
    ) -> dict:
        """
        Extract tables from PDF using img2table.

        Returns:
            Dict mapping page_index to list of bounding boxes for text exclusion.
        """
        table_bboxes_by_page: dict = {}
        try:
            img2table_pdf = Img2TablePDF(
                src=file_path,
                detect_rotation=False,
                pdf_text_extraction=True,
            )
            all_tables = img2table_pdf.extract_tables(
                borderless_tables=True,
                implicit_rows=True,
            )

            table_index = 0
            for page_idx, page_tables in all_tables.items():
                page_num = page_idx + 1
                bboxes = []
                for table_obj in page_tables:
                    df = table_obj.df
                    if df.empty:
                        continue
                    rows = [list(df.columns)] + df.values.tolist()
                    cleaned_rows = self._clean_table_rows(rows)
                    extracted_table = self._build_extracted_table(
                        cleaned_rows=cleaned_rows,
                        table_index=table_index,
                        page_num=page_num,
                        caption=None,
                    )
                    doc.tables.append(extracted_table)
                    table_index += 1
                    if table_obj.bbox:
                        bboxes.append(table_obj.bbox)
                if bboxes:
                    table_bboxes_by_page[page_idx] = bboxes

            # Convert img2table pixel bboxes to PDF-point bboxes so that
            # _extract_text_outside_tables (which uses pdfplumber coords)
            # can correctly exclude table regions.
            if table_bboxes_by_page:
                try:
                    images = img2table_pdf.images
                    with pdfplumber.open(file_path) as pdf:
                        for page_idx, page_bboxes in table_bboxes_by_page.items():
                            if page_idx >= len(pdf.pages) or page_idx >= len(images):
                                continue
                            img = images[page_idx]
                            img_h, img_w = img.shape[:2]
                            page = pdf.pages[page_idx]
                            sx = float(page.width) / img_w
                            sy = float(page.height) / img_h
                            table_bboxes_by_page[page_idx] = [
                                (b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy)
                                for b in page_bboxes
                            ]
                except Exception:
                    pass  # If conversion fails, keep original bboxes
        except Exception as e:
            doc.processing_errors.append(f"img2table extraction failed: {e}")

        return table_bboxes_by_page

    def _extract_tables_camelot(
        self, file_path: str, doc: SECDocument
    ) -> dict:
        """
        Extract tables from PDF using Camelot.

        Tries 'lattice' flavor first (bordered tables), then falls back
        to 'stream' flavor (borderless tables).

        Returns:
            Dict mapping page_index to list of bounding boxes for text exclusion.
        """
        table_bboxes_by_page: dict = {}
        try:
            # Get total page count via pdfplumber
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

            table_index = 0
            for page_num in range(1, total_pages + 1):
                page_str = str(page_num)
                page_idx = page_num - 1

                # Try lattice first (bordered tables)
                tables = camelot.read_pdf(
                    file_path, pages=page_str, flavor="lattice"
                )

                # Fall back to stream (borderless tables) if lattice finds nothing
                if len(tables) == 0:
                    tables = camelot.read_pdf(
                        file_path, pages=page_str, flavor="stream"
                    )

                bboxes = []
                for table_obj in tables:
                    df = table_obj.df
                    if df.empty:
                        continue
                    rows = df.values.tolist()
                    cleaned_rows = self._clean_table_rows(rows)
                    extracted_table = self._build_extracted_table(
                        cleaned_rows=cleaned_rows,
                        table_index=table_index,
                        page_num=page_num,
                        caption=None,
                    )
                    doc.tables.append(extracted_table)
                    table_index += 1

                    # Camelot bbox: (x1, y1, x2, y2) from table_obj._bbox
                    if hasattr(table_obj, '_bbox') and table_obj._bbox:
                        bboxes.append(table_obj._bbox)
                if bboxes:
                    table_bboxes_by_page[page_idx] = bboxes

        except Exception as e:
            doc.processing_errors.append(f"Camelot extraction failed: {e}")

        return table_bboxes_by_page

    def _read_html_into_document(self, file_path: str, doc: SECDocument) -> None:
        """
        Read HTML/HTM file and process in a single pass via HTMLFilingParser.

        Populates doc.raw_content, doc.normalized_content, doc.tables,
        and doc.numeric_facts all at once.

        Args:
            file_path: Path to the HTML file
            doc: SECDocument to populate
        """
        content = self._read_text_file(file_path)
        if content.startswith("[Unable to read"):
            doc.processing_errors.append(content)
            return

        doc.raw_content = content
        self.html_filing_parser.parse_document(doc)

    def _read_xbrl_into_document(self, file_path: str, doc: SECDocument) -> None:
        """
        Read XBRL file and process in a single pass via HTMLFilingParser.

        Populates doc.raw_content, doc.normalized_content, doc.tables,
        and doc.numeric_facts all at once.

        Args:
            file_path: Path to the XBRL file
            doc: SECDocument to populate
        """
        content = self._read_text_file(file_path)
        if content.startswith("[Unable to read"):
            doc.processing_errors.append(content)
            return

        doc.raw_content = content
        self.html_filing_parser.parse_document(doc)

    def _read_text_into_document(self, file_path: str, doc: SECDocument) -> None:
        """
        Read text file and populate SECDocument.

        Args:
            file_path: Path to the text file
            doc: SECDocument to populate
        """
        content = self._read_text_file(file_path)
        if content.startswith("[Unable to read"):
            doc.processing_errors.append(content)
        else:
            doc.raw_content = content
            doc.text_content = content

    def _clean_table_rows(
        self,
        table: List[List[Optional[str]]],
    ) -> List[List[str]]:
        """
        Clean and normalize table rows.

        Args:
            table: Raw table data from pdfplumber

        Returns:
            Cleaned table rows with normalized cell values
        """
        cleaned_rows: List[List[str]] = []
        for row in table:
            cleaned_row: List[str] = []
            for cell in row:
                cell_text = str(cell) if cell is not None else ""
                # Normalize whitespace
                cell_text = " ".join(cell_text.split())
                cleaned_row.append(cell_text)
            cleaned_rows.append(cleaned_row)
        return cleaned_rows

    def _build_extracted_table(
        self,
        cleaned_rows: List[List[str]],
        table_index: int,
        page_num: int,
        caption: Optional[str],
    ) -> ExtractedTable:
        """
        Build an ExtractedTable with TableCell objects from cleaned row data.

        Args:
            cleaned_rows: 2D list of cleaned cell strings
            table_index: Table index for ID generation
            page_num: PDF page number
            caption: Detected table caption/title

        Returns:
            ExtractedTable with cells, headers populated
        """
        # Detect headers (first row as column headers)
        column_headers = cleaned_rows[0] if cleaned_rows else []
        # Detect row labels (first column, skipping header row)
        row_headers = [row[0] for row in cleaned_rows[1:] if row]

        # Build TableCell objects
        cells: List[TableCell] = []
        for row_idx, row in enumerate(cleaned_rows):
            for col_idx, cell_value in enumerate(row):
                col_header = (
                    column_headers[col_idx]
                    if row_idx > 0 and col_idx < len(column_headers)
                    else None
                )
                row_header = (
                    row[0] if row_idx > 0 and col_idx > 0 and row
                    else None
                )
                cells.append(
                    TableCell(
                        value=cell_value,
                        row=row_idx,
                        col=col_idx,
                        column_header=col_header,
                        row_header=row_header,
                    )
                )

        return ExtractedTable(
            table_id=f"table_{table_index}",
            cells=cells,
            caption=caption,
            source="pdf",
            page=page_num,
            column_headers=column_headers,
            row_headers=row_headers,
        )

    def _detect_table_caption(
        self,
        page: Any,
        table_bboxes: List[Tuple[float, float, float, float]],
        table_index: int,
    ) -> Optional[str]:
        """
        Detect caption/title text immediately above a table.

        Looks for text in the region just above the table's bounding box.

        Args:
            page: pdfplumber page object
            table_bboxes: List of table bounding boxes
            table_index: Index of the current table

        Returns:
            Caption text if found, None otherwise
        """
        if table_index >= len(table_bboxes):
            return None

        bbox = table_bboxes[table_index]
        table_top = bbox[1]  # y0 of the table

        # Look for text in a small region above the table (30 points)
        caption_region_top = max(0, table_top - 30)
        caption_bbox = (bbox[0], caption_region_top, bbox[2], table_top)

        try:
            cropped = page.within_bbox(caption_bbox)
            caption_text = cropped.extract_text()
            if caption_text and caption_text.strip():
                return " ".join(caption_text.strip().split())
        except Exception:
            pass

        return None

    def _extract_text_outside_tables(
        self,
        page: Any,
        table_bboxes: List[Tuple[float, float, float, float]],
    ) -> str:
        """
        Extract text from page excluding table regions.

        Args:
            page: pdfplumber page object
            table_bboxes: List of table bounding boxes (x0, y0, x1, y1)

        Returns:
            Text content outside of table regions
        """
        chars = page.chars

        if not chars:
            return ""

        # Filter characters not within any table bbox
        filtered_chars = []
        for char in chars:
            char_x = (char["x0"] + char["x1"]) / 2
            char_y = (char["top"] + char["bottom"]) / 2
            in_table = False

            for bbox in table_bboxes:
                x0, y0, x1, y1 = bbox
                if x0 <= char_x <= x1 and y0 <= char_y <= y1:
                    in_table = True
                    break

            if not in_table:
                filtered_chars.append(char)

        if not filtered_chars:
            return ""

        # Reconstruct text from filtered characters
        filtered_chars.sort(key=lambda c: (c["top"], c["x0"]))

        lines: List[str] = []
        current_line: List[str] = []
        current_top = None
        line_threshold = 3

        for char in filtered_chars:
            if current_top is None:
                current_top = char["top"]

            if abs(char["top"] - current_top) > line_threshold:
                if current_line:
                    lines.append("".join(current_line))
                current_line = [char["text"]]
                current_top = char["top"]
            else:
                current_line.append(char["text"])

        if current_line:
            lines.append("".join(current_line))

        return "\n".join(lines)

    def _read_text_file(self, file_path: str) -> str:
        """
        Read a text file with encoding detection.

        Args:
            file_path: Path to the file

        Returns:
            File content as string
        """
        try:
            with codecs.open(file_path, "r", encoding="utf-8", errors="replace") as file:
                return file.read()
        except Exception as exc:
            return f"[Unable to read file content: {str(exc)}]"


__all__ = ["SecFileReader"]
