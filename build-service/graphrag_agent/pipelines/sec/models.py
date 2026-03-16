"""
Data models for SEC filing processing pipeline.

These dataclasses provide structured containers for passing data
through the entire pipeline: file reading -> normalization ->
table extraction -> numeric parsing -> section extraction ->
chunking -> LLM.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class FileType(Enum):
    """Supported SEC file types."""
    PDF = "pdf"
    HTML = "html"
    HTM = "htm"
    XBRL = "xbrl"
    TXT = "txt"
    MD = "md"


class FormType(Enum):
    """SEC form types."""
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    UNKNOWN = "UNKNOWN"


@dataclass
class TableCell:
    """
    Represents a single cell in an extracted table.

    For HTML tables with XBRL tags, the xbrl field carries full semantic meaning.
    For PDF tables, xbrl is None and meaning comes from table-level context
    (caption, column_header, row_header).

    Attributes:
        value: Display text content of the cell
        row: Row index in the original table
        col: Column index in the original table
        column_header: Text of the column header for this cell (from first row)
        row_header: Text of the row label for this cell (from first column)
        xbrl: XBRL semantic data if available (HTML/XBRL sources only)
    """
    value: str
    row: int
    col: int
    column_header: Optional[str] = None
    row_header: Optional[str] = None
    xbrl: Optional["XBRLNumeric"] = None


@dataclass
class ExtractedTable:
    """
    Represents a table extracted from an SEC filing.
    Unified model for both HTML and PDF sources.

    For HTML tables: cells carry XBRL semantic data via TableCell.xbrl.
    For PDF tables: cells rely on table-level context (caption, headers)
    for meaning. An agent can enrich PDF cells later.

    Attributes:
        table_id: Unique identifier for the table within the document
        cells: List of annotated cell objects
        caption: Table caption or title text
        section: Section reference (e.g., "Item 7", "Item 8")
        source: Source format ("pdf", "html")
        page: Page number (for PDFs)
        column_headers: Detected column header texts
        row_headers: Detected row label texts
    """
    table_id: str
    cells: List[TableCell] = field(default_factory=list)
    caption: Optional[str] = None
    section: Optional[str] = None
    source: str = "unknown"
    page: Optional[int] = None
    column_headers: List[str] = field(default_factory=list)
    row_headers: List[str] = field(default_factory=list)


@dataclass
class XBRLNumeric:
    """
    Represents a numeric fact parsed from an Inline XBRL tag in SEC filings.
    This data structure is for XBRL tag numeric information only (ix:nonFraction).
    PDF and other non-XBRL sources are handled separately.

    Attributes:
        value: Computed numeric value (raw number * 10^scale, with sign applied)
        raw: Original displayed text inside the XBRL tag
        name: XBRL taxonomy concept (e.g., "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")
        context_ref: Reference to the <context> element defining entity, period, and dimensions
        entity: Company identifier (CIK number) resolved from contextRef
        period_start: Period start date resolved from contextRef (None for instant facts)
        period_end: Period end date or instant date resolved from contextRef
        segment: Business segment or dimension info resolved from contextRef (e.g., geographic, product line)
        unit_ref: Reference to the <unit> element (e.g., "usd", "shares", "usdPerShare")
        unit: Resolved unit of measurement from unitRef (e.g., "USD", "shares", "USD/share")
        decimals: Precision indicator (e.g., -6 means rounded to millions, 2 means two decimal places)
        scale: Power of 10 applied to the displayed number (e.g., 6 means multiply by 10^6)
        sign: Negation flag ("-" if the value should be negated)
        format: Display format hint (e.g., "ixt:num-dot-decimal")
        fact_id: Unique identifier for this fact within the filing (ix:id attribute)
    """
    value: float
    raw: str
    name: str
    context_ref: str
    entity: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    segment: Optional[Dict[str, str]] = None
    unit_ref: Optional[str] = None
    unit: Optional[str] = None
    decimals: Optional[int] = None
    scale: int = 0
    sign: Optional[str] = None
    format: Optional[str] = None
    fact_id: Optional[str] = None


@dataclass
class FilingSection:
    """
    Represents a section (Item) from an SEC filing.

    Attributes:
        item: Item number (e.g., "1", "1A", "7", "2.01")
        title: Section title
        content: Section text content
        start: Start position in document
        end: End position in document
    """
    item: str
    title: str
    content: str
    start: int
    end: int


@dataclass
class TextChunk:
    """
    Represents a chunk of text for LLM processing.

    Attributes:
        content: The chunk text
        chunk_index: Index of this chunk within the document
        tokens: List of tokens if tokenized
        source_section: Which section this chunk came from
        source_page: Which page this chunk came from
        has_table: Whether this chunk contains table data
        metadata: Additional metadata
    """
    content: str
    chunk_index: int
    tokens: List[str] = field(default_factory=list)
    source_section: Optional[str] = None
    source_page: Optional[int] = None
    has_table: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SECDocument:
    """
    Main container for SEC filing data throughout the processing pipeline.

    This object is created by the file reader and enriched as it passes
    through each stage of the pipeline.

    Attributes:
        file_path: Relative path to the file
        filename: Base filename
        extension: File extension (e.g., ".pdf", ".html")
        file_type: Enum indicating file type

        raw_content: Original file content as read
        normalized_content: Content after normalization
        text_content: Text content only (tables extracted separately)

        tables: List of extracted tables
        numeric_facts: List of extracted numeric facts
        sections: List of extracted sections

        form_type: Type of SEC form (10-K, 10-Q, 8-K)
        cik: SEC Central Index Key
        company_name: Company name from filing
        filing_date: Date of filing

        chunks: Text chunks ready for LLM

        processing_errors: Errors encountered during processing
        metadata: Additional metadata
    """
    # File identification
    file_path: str
    filename: str
    extension: str
    file_type: FileType

    # Content stages
    raw_content: str = ""
    normalized_content: str = ""
    text_content: str = ""

    # Structured extractions
    tables: List[ExtractedTable] = field(default_factory=list)
    numeric_facts: List[XBRLNumeric] = field(default_factory=list)
    sections: List[FilingSection] = field(default_factory=list)

    # Form metadata
    form_type: FormType = FormType.UNKNOWN
    cik: Optional[str] = None
    company_name: Optional[str] = None
    filing_date: Optional[str] = None

    # Chunked for LLM
    chunks: List[TextChunk] = field(default_factory=list)

    # Pipeline tracking
    processing_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_length(self) -> int:
        """Length of normalized content."""
        return len(self.normalized_content)

    @property
    def table_count(self) -> int:
        """Number of extracted tables."""
        return len(self.tables)

    @property
    def section_count(self) -> int:
        """Number of extracted sections."""
        return len(self.sections)

    @property
    def chunk_count(self) -> int:
        """Number of text chunks."""
        return len(self.chunks)

    @property
    def has_errors(self) -> bool:
        """Whether any processing errors occurred."""
        return len(self.processing_errors) > 0


# Fields from XBRLNumeric that are written to Neo4j FinancialFact nodes.
# Import this in graph_writer and hybrid_tool to avoid hardcoding property names.
FINANCIAL_FACT_WRITE_FIELDS: tuple = (
    "name",
    "value",
    "raw",
    "unit",
    "period_start",
    "period_end",
    "segment",
    "context_ref",
    "scale",
    "decimals",
    # XBRL extras — previously missing
    "entity",
    "unit_ref",
    "sign",
    "format",
    "fact_id",
)

__all__ = [
    "FileType",
    "FormType",
    "ExtractedTable",
    "XBRLNumeric",
    "FilingSection",
    "TextChunk",
    "SECDocument",
    "FINANCIAL_FACT_WRITE_FIELDS",
]
