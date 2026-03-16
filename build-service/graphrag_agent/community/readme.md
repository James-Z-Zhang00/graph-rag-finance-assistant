# Community Detection and Summarization Module

## File Structure

```
graphrag_agent/community/
├── __init__.py                    # Module entry point; exports factory classes
├── readme.md                      # Module documentation
├── detector/                      # Community detector directory
│   ├── __init__.py                # Detector factory class
│   ├── base.py                    # Abstract base class for detectors
│   ├── leiden.py                  # Leiden algorithm implementation
│   ├── projections.py             # Graph projection mixin class
│   └── sllpa.py                   # SLLPA algorithm implementation
└── summary/                       # Community summarizer directory
    ├── __init__.py                # Summarizer factory class
    ├── base.py                    # Abstract base class for summarizers
    ├── leiden.py                  # Leiden community summarizer implementation
    └── sllpa.py                   # SLLPA community summarizer implementation
```

## Module Overview

This module provides community detection and summarization functionality for a Neo4j graph database. It is one of the core components of the GraphRAG knowledge graph project. Primary features include:

1. **Community detection**: Uses graph community discovery algorithms to identify clustered structures in the knowledge graph.
2. **Community summarization**: Uses an LLM to generate semantic summary descriptions for each community, used in global search scenarios.

### Use Cases

- **Global search**: Community summaries provide high-level semantic retrieval entry points.
- **Knowledge graph analysis**: Reveals implicit clustering relationships between entities.
- **Topic discovery**: Automatically identifies topic domains within a document collection.
- **Recommendation systems**: Entity recommendations based on community structure.

## Design and Implementation

### Design Patterns

This module uses multiple design patterns to ensure code maintainability and extensibility:

1. **Factory pattern**: `CommunityDetectorFactory` and `CommunitySummarizerFactory` create different types of detectors and summarizers, hiding implementation details and making algorithm switching easy.
2. **Mixin class**: `GraphProjectionMixin` provides shared graph projection functionality to avoid code duplication.
3. **Context manager**: `BaseCommunityDetector` uses `_graph_projection_context` to manage GDS projection resource lifecycle, ensuring resources are properly released.
4. **Template method pattern**: Base classes define the algorithm skeleton; subclasses implement specific steps to ensure pipeline consistency.
5. **Strategy pattern**: Different community detection strategies are implemented through configurable algorithm types (`leiden`/`sllpa`).

### Core Components and Pipeline

#### 1. Community Detection

**Core class**: `BaseCommunityDetector`
**Implemented algorithms**:
- **Leiden algorithm** (`LeidenDetector`): Hierarchical clustering based on modularity optimization; suitable for large-scale graphs.
- **SLLPA algorithm** (`SLLPADetector`): Speaker-Listener Label Propagation Algorithm; label propagation-based; well-suited for detecting overlapping communities.

**Key pipeline**:
1. **Graph projection**: `create_projection()` projects the native Neo4j graph into an in-memory graph structure in the GDS (Graph Data Science) library.
   - Supports node label filtering (processes only entity nodes).
   - Supports relationship type filtering.
   - Includes a three-tier fallback strategy: standard mode → filtered mode → conservative mode.
2. **Community detection**: `detect_communities()` runs the specific algorithm to identify community structure.
   - Leiden: Optimizes modularity, resolution, random seed, and other parameters.
   - SLLPA: Automatically falls back to Leiden if no communities are detected.
3. **Result persistence**: `save_communities()` persists community IDs to graph database node properties.
   - Property names: `leidenCommunity` or `sllpaCommunity`.
   - Batch write optimization.
4. **Resource cleanup**: `cleanup()` releases memory occupied by GDS projections.

**Adaptive optimization**:
- **Resource awareness**: Automatically adjusts algorithm parameters based on available system memory (configured via `GDS_MEMORY_LIMIT`).
- **Concurrency control**: Controls GDS operation parallelism via the `GDS_CONCURRENCY` environment variable.
- **Error recovery**: Multi-layer error handling and fallback mechanisms (e.g., automatic SLLPA → Leiden fallback).
- **Performance monitoring**: Records projection time, detection time, write time, and other statistics.

#### 2. Community Summarization

**Core class**: `BaseSummarizer`
**Supporting components**:
- `BaseCommunityDescriber`: Generates natural language descriptions for communities.
- `BaseCommunityRanker`: Computes community importance rankings.
- `BaseCommunityStorer`: Persists summary results to the graph database.

**Key pipeline**:
1. **Community ranking**: `calculate_ranks()` computes community importance.
   - Based on community size (node count).
   - Based on relationship density within the community.
   - Supports custom ranking strategies.
2. **Information collection**: `collect_community_info()` batch-fetches community content.
   - Collects all entity nodes and their properties within the community.
   - Collects all relationships and their properties within the community.
   - Processes large communities in batches to avoid memory overflow.
3. **Summary generation**: Uses an LLM to generate semantic summaries for each community.
   - Structured prompt design (entity list + relationship list → summary).
   - Supports custom summary length and style.
   - Processes multiple communities in parallel for improved generation efficiency.
4. **Result storage**: Saves summary information back to the graph database.
   - Creates or updates community summary nodes (with `CommunitySummary` label).
   - Establishes associations between communities and summaries.
   - Records summary generation timestamps and metadata.

**Performance optimization**:
- **Parallel processing**: Uses `ThreadPoolExecutor` for multi-threaded summary generation; concurrency is configurable via `MAX_WORKERS`.
- **Batch processing**: Fetches large-scale community data in batches (batch size: `BATCH_SIZE`) to reduce single-query load.
- **Caching**: Avoids regenerating existing community summaries (optional).
- **Performance statistics**: Records elapsed time for ranking, information collection, summary generation, and storage.

## Algorithm Selection Guide

### Leiden vs SLLPA

| Feature | Leiden | SLLPA |
|---------|--------|-------|
| **Algorithm type** | Hierarchical clustering based on modularity optimization | Community detection based on label propagation |
| **Best for** | Large-scale graphs, clear community boundaries | Overlapping communities, dynamic graphs |
| **Time complexity** | O(n log n) | O(m + n) |
| **Community type** | Non-overlapping communities | Can detect overlapping communities |
| **Parameter sensitivity** | Moderate (resolution parameter affects community granularity) | High (iteration count, thresholds) |
| **Stability** | High (deterministic results) | Moderate (may require multiple runs) |
| **Recommendation** | Default choice; suitable for most scenarios | Use when overlapping community detection is needed |

**Configuration**:
```bash
# Set in the .env file
GRAPH_COMMUNITY_ALGORITHM=leiden  # or sllpa
```

Or in `graphrag_agent/config/settings.py`:
```python
community_algorithm = "leiden"  # or "sllpa"
```

## Configuration Parameters

### Community Detection

```bash
# GDS memory limit (GB)
GDS_MEMORY_LIMIT=6

# GDS concurrency
GDS_CONCURRENCY=4

# Community detection algorithm
GRAPH_COMMUNITY_ALGORITHM=leiden

# Leiden-specific parameters (optional; defaults are fine for most cases)
# LEIDEN_MAX_LEVELS=10
# LEIDEN_GAMMA=1.0
# LEIDEN_THETA=0.01
```

### Community Summarization

```bash
# Summary generation concurrency
MAX_WORKERS=4

# Community info batch size
BATCH_SIZE=100

# LLM configuration (used for summary generation)
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=http://localhost:13000/v1
OPENAI_LLM_MODEL=gpt-4o
```

## Core Functions

### Community Detection Module

**`BaseCommunityDetector.process()`**

Runs the complete community detection pipeline including projection, detection, and saving.

```python
def process(self) -> Dict[str, Any]:
    """
    Run the complete community detection pipeline.

    Returns:
        Dict: A dictionary containing statistics:
            - community_count: Number of communities detected
            - node_count: Number of nodes processed
            - projection_time: Time spent on graph projection (seconds)
            - detection_time: Time spent on community detection (seconds)
            - save_time: Time spent saving results (seconds)
            - total_time: Total elapsed time (seconds)
    """
```

**`GraphProjectionMixin.create_projection()`**

Creates a graph projection with support for multiple fallback strategies to handle memory or data issues.

```python
def create_projection(self) -> Tuple[Any, Dict]:
    """
    Create a graph projection with standard, filtered, and conservative modes.

    Fallback strategy:
    1. Standard mode: Project all nodes and relationships.
    2. Filtered mode: Project only entity nodes (Entity label).
    3. Conservative mode: Cypher projection to minimize memory usage.

    Returns:
        Tuple[Any, Dict]: (GDS graph object, projection statistics)
    """
```

**`LeidenDetector.detect_communities()`**

Runs the Leiden community detection algorithm with parameter optimization and failure fallback.

```python
def detect_communities(self) -> Dict[str, Any]:
    """
    Run the Leiden community detection algorithm.

    Key parameters:
    - includeIntermediateCommunities: Whether to include intermediate-level communities
    - randomSeed: Random seed for reproducibility
    - maxLevels: Maximum number of levels
    - gamma: Resolution parameter controlling community granularity

    Returns:
        Dict: Detection result statistics
    """
```

**`SLLPADetector.detect_communities()`**

Runs the SLLPA algorithm and automatically falls back to Leiden if no communities are detected.

```python
def detect_communities(self) -> Dict[str, Any]:
    """
    Run the SLLPA community detection algorithm.

    Automatic fallback:
    - If SLLPA detects no communities, automatically switches to Leiden.
    - Guarantees usable community partition output in all cases.

    Returns:
        Dict: Detection result statistics
    """
```

### Community Summarization Module

**`BaseSummarizer.process_communities()`**

Runs the full community summarization pipeline including ranking, information collection, summary generation, and storage.

```python
def process_communities(self) -> List[Dict]:
    """
    Run the full community summarization pipeline.

    Pipeline steps:
    1. Calculate community importance rankings.
    2. Batch-collect entity and relationship information for each community.
    3. Call the LLM in parallel to generate summaries.
    4. Persist summaries to Neo4j.

    Returns:
        List[Dict]: List of all community summaries:
            - community_id: Community ID
            - summary: Summary text
            - node_count: Number of nodes in the community
            - relationship_count: Number of relationships in the community
            - rank: Community importance ranking
    """
```

**`BaseSummarizer._process_communities_parallel()`**

Generates community summaries in parallel using multithreading.

```python
def _process_communities_parallel(self, community_info: List[Dict], workers: int) -> List[Dict]:
    """
    Generate community summaries in parallel using a thread pool.

    Args:
        community_info: List of community information.
        workers: Thread pool size (defaults to MAX_WORKERS).

    Returns:
        List[Dict]: List of generated summaries.

    Notes:
    - Each thread independently calls the LLM API.
    - Automatically handles API rate limiting and retries.
    - Supports progress tracking.
    """
```

**`LeidenSummarizer.collect_community_info()`**

Collects detailed information for Leiden communities with support for large-scale batch processing.

```python
def collect_community_info(self) -> List[Dict]:
    """
    Collect community information with support for large-scale batch processing.

    Collected content:
    - Community ID and statistics (node count, relationship count)
    - All entity nodes in the community (including name, type, properties)
    - All relationships in the community (including type, properties, source/target entities)

    Optimization strategies:
    - Batch queries (BATCH_SIZE) to avoid overly large single queries
    - Uses UNWIND for batch processing
    - Filters out empty communities

    Returns:
        List[Dict]: List of community information
    """
```

**`BaseCommunityStorer.store_summary()`**

Persists a community summary to the Neo4j graph database.

```python
def store_summary(self, community_id: int, summary: str, metadata: Dict) -> bool:
    """
    Store a community summary in the graph database.

    Storage strategy:
    1. Create or update the CommunitySummary node.
    2. Set the summary text and metadata (node count, relationship count, generation time, etc.).
    3. Create a HAS_SUMMARY relationship linking the community node.

    Args:
        community_id: Community ID.
        summary: Summary text.
        metadata: Metadata dictionary.

    Returns:
        bool: Whether the storage succeeded.
    """
```

## Usage Examples

### 1. Basic Community Detection

```python
from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience
from graphrag_agent.community import CommunityDetectorFactory

# Initialize graph connections
graph = Neo4jGraph(
    url="neo4j://localhost:7687",
    username="neo4j",
    password="password"
)
gds = GraphDataScience(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

# Create a Leiden community detector
detector = CommunityDetectorFactory.create('leiden', gds, graph)

# Run community detection
results = detector.process()

print(f"Detected {results['community_count']} communities")
print(f"Processed {results['node_count']} nodes")
print(f"Total time: {results['total_time']:.2f}s")
```

### 2. Community Summary Generation

```python
from graphrag_agent.community import CommunitySummarizerFactory

# Create the corresponding Leiden summarizer
summarizer = CommunitySummarizerFactory.create_summarizer('leiden', graph)

# Generate community summaries
summaries = summarizer.process_communities()

print(f"Generated {len(summaries)} community summaries")

# View summary details
for summary in summaries[:3]:  # Show the first 3
    print(f"\nCommunity {summary['community_id']}:")
    print(f"  Nodes: {summary['node_count']}")
    print(f"  Relationships: {summary['relationship_count']}")
    print(f"  Summary: {summary['summary'][:100]}...")
```

### 3. Full Pipeline (Detection + Summarization)

```python
from graphrag_agent.community import CommunityDetectorFactory, CommunitySummarizerFactory

# Step 1: Community detection
algorithm = 'leiden'  # or 'sllpa'
detector = CommunityDetectorFactory.create(algorithm, gds, graph)
detection_results = detector.process()

print(f"Community detection complete: {detection_results['community_count']} communities")

# Step 2: Generate summaries
summarizer = CommunitySummarizerFactory.create_summarizer(algorithm, graph)
summaries = summarizer.process_communities()

print(f"Summary generation complete: {len(summaries)} summaries")
```

### 4. Using the SLLPA Algorithm

```python
# SLLPA is suited for detecting overlapping communities
detector = CommunityDetectorFactory.create('sllpa', gds, graph)
results = detector.process()

# If SLLPA detects no communities, it automatically falls back to Leiden
if results.get('fallback_to_leiden'):
    print("SLLPA detected no communities; automatically switched to Leiden")

# Generate SLLPA community summaries
summarizer = CommunitySummarizerFactory.create_summarizer('sllpa', graph)
summaries = summarizer.process_communities()
```

### 5. Use within the Full Build Pipeline

In `graphrag_agent/integrations/build/main.py`, community detection and summarization are part of the knowledge graph build:

```python
from graphrag_agent.integrations.build.builders import IndexCommunityBuilder

# The index-community builder internally calls community detection and summarization
builder = IndexCommunityBuilder()
builder.build()  # Automatically runs community detection + summary generation
```

## Performance Considerations

### Memory Management

- **GDS projection memory**: Graph projection loads Neo4j graph data into memory; memory usage scales with graph size.
  - Use `GDS_MEMORY_LIMIT` to cap maximum memory usage (in GB).
  - Provides a three-tier fallback: standard → filtered (entity nodes only) → conservative (Cypher projection).
  - Projections are automatically cleaned up after use to free memory.

### Parallel Processing

- **Community detection concurrency**: The GDS algorithm parallelizes internally; thread count is controlled by `GDS_CONCURRENCY`.
- **Summary generation concurrency**: Uses `ThreadPoolExecutor` to call the LLM in parallel; concurrency is configured via `MAX_WORKERS`.
  - Recommended value: 4–8 (depends on LLM API rate limiting policy).
  - Excessively high concurrency may trigger API rate limits.

### Batch Processing

- **Community information collection**: Queries in batches (`BATCH_SIZE`) to avoid overly large single queries.
  - Default batch size: 100.
  - Recommended range for large-scale graphs: 50–200.

### Performance Statistics

All operations record detailed performance metrics:

```python
results = detector.process()
# Example output:
# {
#     'community_count': 42,
#     'node_count': 1523,
#     'projection_time': 2.34,
#     'detection_time': 5.67,
#     'save_time': 1.23,
#     'total_time': 9.24
# }
```

### Recommendations for Very Large Graphs

For extremely large graphs (node count > 1 million):

1. **Reduce concurrency**: Avoid excessive memory pressure.
2. **Increase batch size**: Reduce the number of queries.
3. **Use SLLPA**: Better time complexity (O(m+n) vs O(n log n)).
4. **Process in stages**: Run detection first, then summarization to avoid resource contention.

## Extensibility

### Adding a New Community Detection Algorithm

Inherit from `BaseCommunityDetector` to add a custom algorithm:

```python
from graphrag_agent.community.detector.base import BaseCommunityDetector

class CustomDetector(BaseCommunityDetector):
    """Custom community detection algorithm"""

    def detect_communities(self) -> Dict[str, Any]:
        """Implement custom detection logic"""
        # Use self.gds to call GDS library functions
        # Or implement your own algorithm
        pass

    def save_communities(self, results: Dict) -> None:
        """Save results to the graph database"""
        # Implement saving logic
        pass

# Register with the factory
from graphrag_agent.community import CommunityDetectorFactory
CommunityDetectorFactory.register('custom', CustomDetector)

# Use it
detector = CommunityDetectorFactory.create('custom', gds, graph)
```

### Customizing Community Summary Generation

Inherit from `BaseSummarizer` to implement custom summarization logic:

```python
from graphrag_agent.community.summary.base import BaseSummarizer

class CustomSummarizer(BaseSummarizer):
    """Custom summarizer"""

    def collect_community_info(self) -> List[Dict]:
        """Custom information collection strategy"""
        pass

    def _generate_single_summary(self, community_info: Dict) -> str:
        """Custom summary generation logic"""
        # Can use a different LLM model
        # Or implement template-based summarization
        pass

# Register with the factory
from graphrag_agent.community import CommunitySummarizerFactory
CommunitySummarizerFactory.register('custom', CustomSummarizer)
```

### Extension Points

1. **Custom ranking strategy**: Inherit from `BaseCommunityRanker` to implement different community importance scoring methods.
2. **Custom description generation**: Inherit from `BaseCommunityDescriber` to use different prompts or models.
3. **Custom storage logic**: Inherit from `BaseCommunityStorer` to support other databases or storage backends.
4. **Graph projection strategy**: Extend `GraphProjectionMixin` to support additional filtering and optimization strategies.

## Data Structure Reference

### Community Property Storage

Community detection results are stored as node properties in Neo4j:

```cypher
// Leiden algorithm results
MATCH (e:Entity)
RETURN e.name, e.leidenCommunity

// SLLPA algorithm results
MATCH (e:Entity)
RETURN e.name, e.sllpaCommunity
```

### Community Summary Node Structure

Community summaries are stored as independent nodes:

```cypher
(:CommunitySummary {
    community_id: 42,              // Community ID
    summary: "This community is about...",  // Summary text
    node_count: 156,                // Number of nodes in the community
    relationship_count: 423,        // Number of relationships in the community
    rank: 0.85,                     // Importance ranking score
    created_at: "2025-10-25T10:30:00",  // Generation timestamp
    algorithm: "leiden"             // Algorithm used
})
```

### Community-Summary Relationship

```cypher
// Query a community and its summary
MATCH (e:Entity {leidenCommunity: 42})<-[:BELONGS_TO]-(s:CommunitySummary)
RETURN e, s
```
