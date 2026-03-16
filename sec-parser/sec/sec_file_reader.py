import codecs
import os
from typing import Any, List, Optional, Tuple

from sec.models import (
    ExtractedTable,
    FileType,
    SECDocument,
    TableCell,
)

import pdfplumber
from img2table.document import PDF as Img2TablePDF
import camelot

from sec.html_filing_parser import HTMLFilingParser


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
        self.directory_path = directory_path
        self.html_filing_parser = html_filing_parser or HTMLFilingParser()
        self.pdf_table_method = pdf_table_method

    def read_files(
        self,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[SECDocument]:
        if file_extensions is None:
            file_extensions = [".txt", ".pdf", ".md", ".html", ".htm", ".xbrl"]

        results: List[SECDocument] = []
        file_paths = self._collect_file_paths(file_extensions, recursive)

        for rel_path, abs_path, extension in file_paths:
            doc = self._read_single_file(rel_path, abs_path, extension)
            results.append(doc)

        return results

    def _collect_file_paths(
        self,
        file_extensions: List[str],
        recursive: bool,
    ) -> List[Tuple[str, str, str]]:
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
        filename = os.path.basename(rel_path)
        file_type = EXTENSION_TO_FILETYPE.get(extension, FileType.TXT)

        doc = SECDocument(
            file_path=rel_path,
            filename=filename,
            extension=extension,
            file_type=file_type,
        )

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
        table_bboxes_by_page: dict = {}

        if self.pdf_table_method == "camelot":
            table_bboxes_by_page = self._extract_tables_camelot(file_path, doc)
        else:
            table_bboxes_by_page = self._extract_tables_img2table(file_path, doc)

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

    def _extract_tables_img2table(self, file_path: str, doc: SECDocument) -> dict:
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
                    pass
        except Exception as e:
            doc.processing_errors.append(f"img2table extraction failed: {e}")

        return table_bboxes_by_page

    def _extract_tables_camelot(self, file_path: str, doc: SECDocument) -> dict:
        table_bboxes_by_page: dict = {}
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

            table_index = 0
            for page_num in range(1, total_pages + 1):
                page_str = str(page_num)
                page_idx = page_num - 1

                tables = camelot.read_pdf(file_path, pages=page_str, flavor="lattice")

                if len(tables) == 0:
                    tables = camelot.read_pdf(file_path, pages=page_str, flavor="stream")

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

                    if hasattr(table_obj, '_bbox') and table_obj._bbox:
                        bboxes.append(table_obj._bbox)
                if bboxes:
                    table_bboxes_by_page[page_idx] = bboxes

        except Exception as e:
            doc.processing_errors.append(f"Camelot extraction failed: {e}")

        return table_bboxes_by_page

    def _read_html_into_document(self, file_path: str, doc: SECDocument) -> None:
        content = self._read_text_file(file_path)
        if content.startswith("[Unable to read"):
            doc.processing_errors.append(content)
            return

        doc.raw_content = content
        self.html_filing_parser.parse_document(doc)

    def _read_xbrl_into_document(self, file_path: str, doc: SECDocument) -> None:
        content = self._read_text_file(file_path)
        if content.startswith("[Unable to read"):
            doc.processing_errors.append(content)
            return

        doc.raw_content = content
        self.html_filing_parser.parse_document(doc)

    def _read_text_into_document(self, file_path: str, doc: SECDocument) -> None:
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
        cleaned_rows: List[List[str]] = []
        for row in table:
            cleaned_row: List[str] = []
            for cell in row:
                cell_text = str(cell) if cell is not None else ""
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
        column_headers = cleaned_rows[0] if cleaned_rows else []
        row_headers = [row[0] for row in cleaned_rows[1:] if row]

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

    def _extract_text_outside_tables(
        self,
        page: Any,
        table_bboxes: List[Tuple[float, float, float, float]],
    ) -> str:
        chars = page.chars

        if not chars:
            return ""

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
        try:
            with codecs.open(file_path, "r", encoding="utf-8", errors="replace") as file:
                return file.read()
        except Exception as exc:
            return f"[Unable to read file content: {str(exc)}]"


__all__ = ["SecFileReader"]
