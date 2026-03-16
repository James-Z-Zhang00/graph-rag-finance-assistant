from __future__ import annotations

from typing import List, Optional

from graphrag_agent.config.settings import CHUNK_SIZE, OVERLAP
from graphrag_agent.pipelines.ingestion.text_chunker import TextChunker
from graphrag_agent.pipelines.sec.filing_normalizer import FilingNormalizer
from dataclasses import asdict

from graphrag_agent.pipelines.sec.models import (
    FINANCIAL_FACT_WRITE_FIELDS,
    SECDocument,
    TextChunk,
)
from graphrag_agent.pipelines.sec.sec_file_reader import SecFileReader
from graphrag_agent.pipelines.sec.section_extractor import SectionExtractor


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
        """
        Initialize the SEC filing processor.

        Args:
            directory_path: File directory path
            chunk_size: Chunk size (tokens)
            overlap: Chunk overlap size (tokens)
            normalizer: Optional normalizer instance
            section_extractor: Optional section extractor instance
            pdf_table_method: PDF table extraction method - "img2table" or "camelot"
        """
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

        Args:
            file_extensions: File extensions to process; if None, use SEC defaults
            recursive: Whether to process subdirectories recursively

        Returns:
            List of SECDocument objects with all extracted data
        """
        # Stage 1: Read files
        documents = self._stage1_read_files(file_extensions, recursive)
        print(f"SecFilingProcessor found {len(documents)} files")

        # Stage 2-4: Process each document
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
        Process SEC filings and return results in the same dict format
        as DocumentProcessor.process_directory() for compatibility with
        the build pipeline (build_graph.py).

        Returns:
            List of dicts with keys: filepath, filename, extension,
            content, content_length, chunks, chunk_count, chunk_lengths,
            average_chunk_length
        """
        documents = self.process(file_extensions, recursive)
        results = []
        for doc in documents:
            # Convert TextChunk objects to token lists (same format as TextChunker output)
            chunks_as_token_lists = [tc.tokens for tc in doc.chunks]
            chunk_lengths = [len(tc.content) for tc in doc.chunks]

            # Serialize numeric facts (XBRL) — use asdict() so new XBRLNumeric
            # fields flow through automatically; filter to only Neo4j write fields.
            numeric_facts = []
            for f in doc.numeric_facts:
                d = {k: v for k, v in asdict(f).items() if k in FINANCIAL_FACT_WRITE_FIELDS}
                # segment is Optional[Dict] — stringify for Neo4j string property
                if d.get("segment") is not None:
                    d["segment"] = str(d["segment"])
                if d.get("unit") is None:
                    d["unit"] = ""
                numeric_facts.append(d)

            # Serialize tables — convert cells to a readable text block
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
        # Group cells by row index
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
        """
        Stage 1: Read files into SECDocument objects.

        - PDF: text + table extraction (tables as TableCell objects)
        - HTML/HTM/XBRL: single-pass HTMLFilingParser (tables, XBRL numerics, normalized text)
        - TXT/MD: raw text read
        """
        return self.file_reader.read_files(
            file_extensions=file_extensions,
            recursive=recursive,
        )

    def _stage2_normalize(self, doc: SECDocument) -> None:
        """
        Stage 2: Normalize content.

        - PDF: cleans PDF artifacts, page markers, URLs
        - HTML/HTM/XBRL: strips HTML tags (skipping tables and XBRL tags
          already extracted by HTMLFilingParser in stage 1), removes XBRL artifacts
        - TXT/MD: removes SEC headers, separator lines
        """
        self.normalizer.normalize_document(doc)

    def _stage3_extract_sections(self, doc: SECDocument) -> None:
        """
        Stage 3: Extract sections and detect form type.
        Works on normalized_content (populated by stage 1 or stage 2).
        """
        self.section_extractor.extract_sections_from_document(doc)

    def _stage4_chunk(self, doc: SECDocument) -> None:
        """
        Stage 4: Chunk text for LLM processing.
        """
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
