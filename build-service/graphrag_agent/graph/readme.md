# Graph Construction Module

## Directory Structure

```
graphrag_agent/graph/
├── __init__.py                # Module entry; exports main classes and functions
├── core/                      # Core components
│   ├── __init__.py            # Exports core components
│   ├── base_indexer.py        # Base indexer class
│   ├── graph_connection.py    # Graph database connection management
│   └── utils.py               # Utility functions (timers, hash generation, etc.)
├── extraction/                # Entity-relationship extraction components
│   ├── __init__.py            # Exports extraction components
│   ├── entity_extractor.py    # Entity-relationship extractor
│   └── graph_writer.py        # Graph data writer
├── graph_consistency_validator.py  # Graph consistency validation tool
├── indexing/                  # Index management components
│   ├── __init__.py            # Exports indexing components
│   ├── chunk_indexer.py       # Text chunk index management
│   ├── embedding_manager.py   # Embedding vector management
│   └── entity_indexer.py      # Entity index management
├── processing/                # Entity processing components
│   ├── __init__.py            # Exports processing components
│   ├── entity_merger.py       # Entity merge manager
│   ├── similar_entity.py      # Similar entity detection
│   ├── entity_disambiguation.py # Entity disambiguator
│   ├── entity_alignment.py    # Entity aligner
│   └── entity_quality.py      # Entity quality processor
└── structure/                 # Graph structure construction components
    ├── __init__.py            # Exports structure components
    └── struct_builder.py      # Graph structure builder
```

## Module Overview

This module is a complete graph construction and query system built on Neo4j. Core features include document parsing, entity-relationship extraction, embedding vector index creation, similar entity detection and merging, and more. The module uses a highly modular design that supports large-scale data processing and optimized query performance.

## Core Design

### 1. Graph Data Structure

The system builds its graph around the following core node types:
- `__Document__`: Document nodes representing a complete document
- `__Chunk__`: Text chunk nodes representing a segment of a document
- `__Entity__`: Entity nodes representing concepts, objects, etc. extracted from text

Relationships between nodes include:
- `PART_OF`: Ownership relationship between Chunk and Document
- `NEXT_CHUNK`: Sequential relationship between text chunks
- `MENTIONS`: Mention relationship between a text chunk and an entity
- `SIMILAR`: Similarity relationship between entities

### 2. Graph Construction Pipeline

1. **Document structuring**: `GraphStructureBuilder` splits documents into chunks and establishes their structure
2. **Entity-relationship extraction**: `EntityRelationExtractor` uses an LLM to extract entities and relationships from text
3. **Graph writing**: `GraphWriter` writes the extracted entities and relationships to Neo4j
4. **Vector index creation**: `ChunkIndexManager` and `EntityIndexManager` create embedding vector indexes for nodes
5. **Similar entity detection**: `SimilarEntityDetector` detects duplicate entities using vector similarity and GDS algorithms
6. **Entity merging**: `EntityMerger` merges similar entities based on LLM decisions
7. **Entity quality improvement**: `EntityQualityProcessor` further improves entity quality through disambiguation and alignment

### 3. Performance Optimization Strategies

- **Batch processing**: All modules implement batch operations to reduce database round-trips
- **Parallel processing**: Uses thread pools to process data in parallel
- **Caching**: Uses caches during entity extraction to avoid redundant computation
- **Efficient indexing**: Appropriate indexing strategy improves query performance
- **Error recovery**: Implements retry mechanisms and error recovery

## Core Classes

### Graph Database Connection

`GraphConnectionManager` provides connection management for Neo4j. It implements the singleton pattern to ensure connection reuse and manages queries and index creation uniformly.

```python
# Example usage
graph = connection_manager.get_connection()
result = graph.query("MATCH (n) RETURN count(n) as count")
```

### Graph Structure Construction

`GraphStructureBuilder` creates document and text chunk nodes and establishes the structural relationships between them:

```python
builder = GraphStructureBuilder()
builder.create_document(type="text", uri="path/to/doc", file_name="example.txt", domain="test")
chunks_with_hash = builder.create_relation_between_chunks(file_name, chunks)
```

### Entity-Relationship Extraction

`EntityRelationExtractor` extracts entities and relationships from text chunks using an LLM:

```python
extractor = EntityRelationExtractor(llm, system_template, human_template, entity_types, relationship_types)
processed_chunks = extractor.process_chunks(file_contents)
```

### Vector Index Management

`ChunkIndexManager` and `EntityIndexManager` compute embedding vectors and create indexes:

```python
chunk_indexer = ChunkIndexManager()
vector_store = chunk_indexer.create_chunk_index()
```

### Similar Entity Detection and Merging

`SimilarEntityDetector` and `EntityMerger` work together to deduplicate entities:

```python
detector = SimilarEntityDetector()
duplicate_candidates = detector.process_entities()

merger = EntityMerger()
merged_count = merger.process_duplicates(duplicate_candidates)
```

### Graph Consistency Validation

`GraphConsistencyValidator` detects and repairs consistency issues in the graph:

```python
validator = GraphConsistencyValidator()
validation_result = validator.validate_graph()
repair_result = validator.repair_graph()
```

### Entity Quality Improvement

`EntityQualityProcessor` integrates entity disambiguation and alignment to further improve entity quality:

```python
quality_processor = EntityQualityProcessor()
result = quality_processor.process()
```

**Sub-modules included:**

#### Entity Disambiguation (EntityDisambiguator)

Maps mentions to canonical entities in the knowledge graph:

```python
disambiguator = EntityDisambiguator()

# Disambiguate a single mention
result = disambiguator.disambiguate("entity name")

# Batch disambiguation
results = disambiguator.batch_disambiguate(["entity1", "entity2"])

# Apply to the entire graph
updated_count = disambiguator.apply_to_graph()
```

**Core pipeline:**
1. **String recall**: Uses edit distance to quickly find similar entity candidates
2. **Vector re-ranking**: Reranks candidates using semantic similarity
3. **NIL detection**: Identifies out-of-vocabulary entities (new entities not present in the knowledge base)
4. **Apply to graph**: Sets `canonical_id` for entities in WCC groups

#### Entity Alignment (EntityAligner)

Aligns and merges entities sharing the same `canonical_id`:

```python
aligner = EntityAligner()

# Run the full alignment pipeline
result = aligner.align_all(batch_size=100)
```

**Core pipeline:**
1. **Group by canonical_id**: Finds all entities pointing to the same canonical entity
2. **Conflict detection**: Detects semantic conflicts through relationship type similarity
3. **Conflict resolution**: Uses LLM to intelligently decide which entity to keep
4. **Entity merging**: Merges entities in the same group, preserving all relationships and attributes

**Key characteristics:**
- Uses CALL subqueries to isolate edge processing, ensuring pipeline robustness
- Preserves original relationship types without losing semantic information
- Supports batch processing to avoid memory overflow
- Intelligent conflict resolution based on Jaccard similarity and LLM judgment
