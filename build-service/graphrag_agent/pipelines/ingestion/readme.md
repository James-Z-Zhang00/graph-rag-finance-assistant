# Document Processor Module

## Project Structure
```
graphrag_agent/
├── config/                     # Configuration directory
│   └── settings.py             # Global configuration parameters
└── pipelines/
    └── ingestion/              # Document ingestion module
        ├── __init__.py         # Package init
        ├── document_processor.py  # Core document processor
        ├── file_reader.py      # Multi-format file reader
        └── text_chunker.py     # Text chunker
```

## Module Overview

Document Processor is a module for reading, processing, and chunking documents in multiple formats. It supports a wide range of file types and performs semantic-aware chunking to prepare content for downstream vectorization and retrieval.

## Core Functionality and Design

### 1. File Reading (FileReader)

`FileReader` provides robust file-reading capabilities for common document types:

- Text files (TXT)
- PDF documents (PDF)
- Markdown documents (MD)
- Word documents (DOCX, DOC)
- Data files (CSV, JSON, YAML/YML)

The reader uses multiple strategies to ensure content is correctly loaded:
- Supports multiple encodings with auto-detection
- Uses fallback methods for legacy Word documents (.doc)
- Provides helpers for structured outputs (e.g., CSV to dict rows)

```python
# Usage example
reader = FileReader(directory_path)
file_contents = reader.read_files(['.txt', '.pdf'])  # Specify file types
```

### 2. Text chunking (TextChunker)

`TextChunker` handles text chunking and can:

- Perform tokenization
- Split text at sentence boundaries
- Support chunk overlap for context continuity
- Handle exceptions and very long text

```python
# Usage example
chunker = TextChunker(chunk_size=500, overlap=100)
chunks = chunker.chunk_text(text_content)
```

### 3. Document Processor (DocumentProcessor)

`DocumentProcessor` integrates file reading and text chunking to provide a complete processing workflow:

- Batch process files in a target directory
- Generate file statistics (type distribution, content length, etc.)
- Chunk each file's content
- Collect processing results and errors

```python
# Usage example
processor = DocumentProcessor(directory_path)
stats = processor.get_file_stats()  # Get file statistics
results = processor.process_directory()  # Process all supported files
```

## Core Functions

1. **`FileReader.read_files()`**: Read file contents by extension and return (filename, content) tuples.

2. **`TextChunker.chunk_text()`**: Split a single text into chunks with sentence boundaries and overlap.

3. **`DocumentProcessor.process_directory()`**: Process all supported files and return detailed file results with content and chunks.

4. **`DocumentProcessor.get_file_stats()`**: Return directory file statistics, including type distribution and length metrics.

## Use Cases

This module is suitable for scenarios that require large-scale document processing and downstream semantic analysis, such as:

- Building document search systems
- Knowledge base Q&A applications
- Intelligent document processing and classification
- Corpus construction and analysis