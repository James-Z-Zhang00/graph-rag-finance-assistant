from __future__ import annotations

from typing import List, Optional

from config.settings import CHUNK_SIZE, OVERLAP
from ingestion.text_chunker import TextChunker
from sec.filing_normalizer import FilingNormalizer
from dataclasses import asdict

from sec.models import (
    FINANCIAL_FACT_WRITE_FIELDS,
    SECDocument,
    TextChunk,
)
from sec.sec_file_reader import SecFileReader
from sec.section_extractor import SectionExtractor


class SecFilingProcessor:
    """
    SEC filing processor for 10-K/10-Q/8-K style documents.

    Pipeline stages:
        1. Read files
           - PDF: text + table extraction (tables as TableCell objects)
           - HTML/HTM/XBRL: single-pass HTMLFilingParser (tables, XBRL numerics, normalized text)
           - TXT/MD: raw text read
        2. Normalize content (PDF/TXT only; HTML/XBRL already normalized in stage 1)
        3. Extract sections
        4. Chunk text

    Uses SECDocument as the structured data container throughout the pipeline.
    """

    def __init__(
        self,
        directory_path: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = OVERLAP,
        normalizer: Optional[FilingNormalizer] = None,
        section_extractor: Optional[SectionExtractor] = None,
        pdf_table_method: str = "img2table",
    ):
        self.directory_path = directory_path
        self.file_reader = SecFileReader(directory_path, pdf_table_method=pdf_table_method)
        self.chunker = TextChunker(chunk_size, overlap)
        self.normalizer = normalizer or FilingNormalizer()
        self.section_extractor = section_extractor or SectionExtractor()

    def process(
        self,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[SECDocument]:
        """
        Main entry point: process SEC filings through the complete pipeline.

        Returns:
            List of SECDocument objects with all extracted data
        """
        documents = self._stage1_read_files(file_extensions, recursive)
        print(f"SecFilingProcessor found {len(documents)} files")

        for doc in documents:
            self._stage2_normalize(doc)
            self._stage3_extract_sections(doc)
            self._stage4_chunk(doc)

        return documents

    def process_directory(
        self,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[dict]:
        """
        Process SEC filings and return results as JSON-serializable dicts.

        Compatible with DocumentProcessor.process_directory() format for the
        build pipeline (build_graph.py).

        Returns:
            List of dicts with keys: filepath, filename, extension,
            content, content_length, chunks, chunk_count, chunk_lengths,
            average_chunk_length, numeric_facts, tables, sections, form_type,
            cik, company_name, filing_date
        """
        documents = self.process(file_extensions, recursive)
        results = []
        for doc in documents:
            chunks_as_token_lists = [tc.tokens for tc in doc.chunks]
            chunk_lengths = [len(tc.content) for tc in doc.chunks]

            numeric_facts = []
            for f in doc.numeric_facts:
                d = {k: v for k, v in asdict(f).items() if k in FINANCIAL_FACT_WRITE_FIELDS}
                if d.get("segment") is not None:
                    d["segment"] = str(d["segment"])
                if d.get("unit") is None:
                    d["unit"] = ""
                numeric_facts.append(d)

            tables = [
                {
                    "table_id": t.table_id,
                    "caption": t.caption or "",
                    "section": t.section or "",
                    "source": t.source,
                    "content": self._table_to_text(t),
                }
                for t in doc.tables
            ]

            file_result = {
                "filepath": doc.file_path,
                "filename": doc.filename,
                "extension": doc.extension,
                "content": doc.normalized_content or doc.text_content or doc.raw_content,
                "content_length": len(doc.normalized_content or doc.text_content or doc.raw_content),
                "chunks": chunks_as_token_lists,
                "chunk_count": doc.chunk_count,
                "chunk_lengths": chunk_lengths,
                "average_chunk_length": (
                    sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
                ),
                "numeric_facts": numeric_facts,
                "tables": tables,
                "form_type":    doc.form_type.value,
                "file_path":    doc.file_path,
                "cik":          doc.cik or "",
                "company_name": doc.company_name or "",
                "filing_date":  doc.filing_date or "",
                "sections": [
                    {
                        "item":    s.item,
                        "title":   s.title,
                        "content": s.content,
                        "start":   s.start,
                        "end":     s.end,
                    }
                    for s in doc.sections
                ],
            }
            results.append(file_result)
        return results

    @staticmethod
    def _table_to_text(table) -> str:
        """Convert an ExtractedTable to a readable text string for storage."""
        lines = []
        if table.caption:
            lines.append(f"Table: {table.caption}")
        if table.column_headers:
            lines.append(" | ".join(str(h) for h in table.column_headers))
            lines.append("-" * 40)
        rows: dict = {}
        for cell in table.cells:
            rows.setdefault(cell.row, {})[cell.col] = cell.value
        for row_idx in sorted(rows.keys()):
            row = rows[row_idx]
            lines.append(" | ".join(str(row.get(col, "")) for col in sorted(row.keys())))
        return "\n".join(lines)

    def _stage1_read_files(
        self,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[SECDocument]:
        return self.file_reader.read_files(
            file_extensions=file_extensions,
            recursive=recursive,
        )

    def _stage2_normalize(self, doc: SECDocument) -> None:
        self.normalizer.normalize_document(doc)

    def _stage3_extract_sections(self, doc: SECDocument) -> None:
        self.section_extractor.extract_sections_from_document(doc)

    def _stage4_chunk(self, doc: SECDocument) -> None:
        if not doc.normalized_content:
            return

        try:
            raw_chunks = self.chunker.chunk_text(doc.normalized_content)
            for idx, chunk_tokens in enumerate(raw_chunks):
                chunk_content = "".join(chunk_tokens)
                text_chunk = TextChunk(
                    content=chunk_content,
                    chunk_index=idx,
                    tokens=chunk_tokens,
                )
                doc.chunks.append(text_chunk)
        except Exception as exc:
            error_msg = f"Chunking error ({doc.file_path}): {str(exc)}"
            doc.processing_errors.append(error_msg)
            print(error_msg)


__all__ = ["SecFilingProcessor"]
