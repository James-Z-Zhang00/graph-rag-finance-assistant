import re
import concurrent.futures
from typing import List, Set
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from graphrag_agent.graph.core import connection_manager
from graphrag_agent.config.settings import BATCH_SIZE as DEFAULT_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS
from graphrag_agent.pipelines.sec.models import FINANCIAL_FACT_WRITE_FIELDS

class GraphWriter:
    """
    Graph writer responsible for writing extracted entities and relationships to the Neo4j graph database.
    Handles parsing, conversion to GraphDocument, and batch writing to the graph database.
    """

    def __init__(self, graph: Neo4jGraph = None, batch_size: int = 50, max_workers: int = 4):
        """
        Initialize the graph writer.

        Args:
            graph: Neo4j graph object; if None, uses the connection manager
            batch_size: Batch processing size
            max_workers: Number of parallel worker threads
        """
        self.graph = graph or connection_manager.get_connection()
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS

        # Node cache to reduce duplicate node creation
        self.node_cache = {}

        # Track already-processed nodes to reduce duplicate operations
        self.processed_nodes: Set[str] = set()

    def convert_to_graph_document(self, chunk_id: str, input_text: str, result: str) -> GraphDocument:
        """
        Convert extracted entity-relationship text into a GraphDocument object.

        Args:
            chunk_id: Text chunk ID
            input_text: Input text
            result: Extraction result

        Returns:
            GraphDocument: Converted graph document object
        """
        node_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\s*\)')
        relationship_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')

        nodes = {}
        relationships = []

        # Efficient regex matching
        try:
            # Parse nodes — use cache for efficiency
            for match in node_pattern.findall(result):
                node_id, node_type, description = match
                if node_id in self.node_cache:
                    nodes[node_id] = self.node_cache[node_id]
                elif node_id not in nodes:
                    new_node = Node(
                        id=node_id,
                        type=node_type,
                        properties={'description': description}
                    )
                    nodes[node_id] = new_node
                    self.node_cache[node_id] = new_node

            # Parse relationships
            for match in relationship_pattern.findall(result):
                source_id, target_id, rel_type, description, weight = match
                # Ensure source node exists — check cache first
                if source_id not in nodes:
                    if source_id in self.node_cache:
                        nodes[source_id] = self.node_cache[source_id]
                    else:
                        new_node = Node(
                            id=source_id,
                            type="Unknown",
                            properties={'description': 'No additional data'}
                        )
                        nodes[source_id] = new_node
                        self.node_cache[source_id] = new_node

                # Ensure target node exists — check cache first
                if target_id not in nodes:
                    if target_id in self.node_cache:
                        nodes[target_id] = self.node_cache[target_id]
                    else:
                        new_node = Node(
                            id=target_id,
                            type="Unknown",
                            properties={'description': 'No additional data'}
                        )
                        nodes[target_id] = new_node
                        self.node_cache[target_id] = new_node

                relationships.append(
                    Relationship(
                        source=nodes[source_id],
                        target=nodes[target_id],
                        type=rel_type,
                        properties={
                            "description": description,
                            "weight": float(weight)
                        }
                    )
                )
        except Exception as e:
            print(f"Error parsing text: {e}")
            # Return empty GraphDocument rather than raising
            return GraphDocument(
                nodes=[],
                relationships=[],
                source=Document(
                    page_content=input_text,
                    metadata={"chunk_id": chunk_id, "error": str(e)}
                )
            )

        # Create and return GraphDocument
        return GraphDocument(
            nodes=list(nodes.values()),
            relationships=relationships,
            source=Document(
                page_content=input_text,
                metadata={"chunk_id": chunk_id}
            )
        )

    def process_and_write_graph_documents(self, file_contents: List) -> None:
        """
        Process and write GraphDocument objects for all files — uses parallel processing and batching.

        Args:
            file_contents: List of file contents
        """
        all_graph_documents = []
        all_chunk_ids = []

        # Pre-allocate list size
        total_chunks = sum(len(file_content[3]) for file_content in file_contents)
        all_graph_documents = [None] * total_chunks
        all_chunk_ids = [None] * total_chunks

        chunk_index = 0
        error_count = 0

        print(f"Starting GraphDocument processing for {total_chunks} chunks")

        # Use thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {}

            # Submit all tasks
            for file_content in file_contents:
                chunks = file_content[3]  # chunks_with_hash at index 3
                results = file_content[4]  # extraction results at index 4

                for i, (chunk, result) in enumerate(zip(chunks, results)):
                    future = executor.submit(
                        self.convert_to_graph_document,
                        chunk["chunk_id"],
                        chunk["chunk_doc"].page_content,
                        result
                    )
                    future_to_index[future] = chunk_index
                    chunk_index += 1

            # Collect results
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    graph_document = future.result()

                    # Keep only valid graph documents
                    if len(graph_document.nodes) > 0 or len(graph_document.relationships) > 0:
                        all_graph_documents[idx] = graph_document
                        all_chunk_ids[idx] = graph_document.source.metadata.get("chunk_id")
                    else:
                        all_graph_documents[idx] = None
                        all_chunk_ids[idx] = None

                except Exception as e:
                    error_count += 1
                    print(f"Error processing chunk ({error_count} errors so far): {e}")
                    all_graph_documents[idx] = None
                    all_chunk_ids[idx] = None

        # Filter out None values
        all_graph_documents = [doc for doc in all_graph_documents if doc is not None]
        all_chunk_ids = [id for id in all_chunk_ids if id is not None]

        print(f"Processed {total_chunks} chunks: {len(all_graph_documents)} valid documents, {error_count} errors")

        # Batch write graph documents
        self._batch_write_graph_documents(all_graph_documents)

        # Batch merge chunk relationships
        if all_chunk_ids:
            self.merge_chunk_relationships(all_chunk_ids)

    def _batch_write_graph_documents(self, documents: List[GraphDocument]) -> None:
        """
        Batch write graph documents.

        Args:
            documents: List of graph documents
        """
        if not documents:
            return

        # Dynamic batch size adjustment
        optimal_batch_size = min(self.batch_size, max(10, len(documents) // 10))
        total_batches = (len(documents) + optimal_batch_size - 1) // optimal_batch_size

        print(f"Batch writing {len(documents)} documents, batch size: {optimal_batch_size}, total batches: {total_batches}")

        for i in range(0, len(documents), optimal_batch_size):
            batch = documents[i:i+optimal_batch_size]
            if batch:
                try:
                    self.graph.add_graph_documents(
                        batch,
                        baseEntityLabel=True,
                        include_source=True
                    )
                    print(f"Wrote batch {i//optimal_batch_size + 1}/{total_batches}")
                except Exception as e:
                    print(f"Error writing graph document batch: {e}")
                    # If batch write fails, try writing one by one to avoid losing entire batch
                    for doc in batch:
                        try:
                            self.graph.add_graph_documents(
                                [doc],
                                baseEntityLabel=True,
                                include_source=True
                            )
                        except Exception as e2:
                            print(f"Single document write failed: {e2}")

    def write_numeric_facts(self, filename: str, numeric_facts: list) -> None:
        """
        Write XBRL numeric facts as FinancialFact nodes linked to their source Document.

        Args:
            filename: Source document filename (matches __Document__.fileName)
            numeric_facts: List of serialized fact dicts from SecFilingProcessor
        """
        if not numeric_facts:
            return

        # Build SET clause from FINANCIAL_FACT_WRITE_FIELDS, excluding MERGE keys
        _merge_keys = {"name", "context_ref"}
        _set_lines = [
            f"        f.{field} = fact.{field}"
            for field in FINANCIAL_FACT_WRITE_FIELDS
            if field not in _merge_keys
        ]
        _set_clause = ",\n".join(_set_lines)
        _cypher = f"""
                    UNWIND $batch AS fact
                    MATCH (d:`__Document__` {{fileName: $filename}})
                    MERGE (f:FinancialFact {{name: fact.name, context_ref: fact.context_ref}})
                    ON CREATE SET
{_set_clause}
                    ON MATCH SET
{_set_clause}
                    MERGE (d)-[:HAS_FACT]->(f)
                    """

        # Batch in chunks to avoid large Cypher payloads
        for i in range(0, len(numeric_facts), self.batch_size):
            batch = numeric_facts[i:i + self.batch_size]
            try:
                self.graph.query(
                    _cypher,
                    params={"batch": batch, "filename": filename},
                )
                print(f"Wrote {len(batch)} FinancialFact nodes for {filename}")
            except Exception as e:
                print(f"Error writing FinancialFact batch for {filename}: {e}")

    def write_tables(self, filename: str, tables: list) -> None:
        """
        Write extracted tables as Table nodes linked to their source Document.

        Args:
            filename: Source document filename (matches __Document__.fileName)
            tables: List of serialized table dicts from SecFilingProcessor
        """
        if not tables:
            return

        for i in range(0, len(tables), self.batch_size):
            batch = tables[i:i + self.batch_size]
            try:
                self.graph.query(
                    """
                    UNWIND $batch AS tbl
                    MATCH (d:`__Document__` {fileName: $filename})
                    MERGE (t:Table {table_id: tbl.table_id, fileName: $filename})
                    ON CREATE SET
                        t.caption = tbl.caption,
                        t.section = tbl.section,
                        t.source  = tbl.source,
                        t.content = tbl.content
                    ON MATCH SET
                        t.caption = tbl.caption,
                        t.section = tbl.section,
                        t.source  = tbl.source,
                        t.content = tbl.content
                    MERGE (d)-[:HAS_TABLE]->(t)
                    """,
                    params={"batch": batch, "filename": filename},
                )
                print(f"Wrote {len(batch)} Table nodes for {filename}")
            except Exception as e:
                print(f"Error writing Table batch for {filename}: {e}")

    def write_sections(self, filename: str, sections: list) -> None:
        """
        Write filing sections as FilingSection nodes linked to their source Document.

        Args:
            filename: Source document filename (matches __Document__.fileName)
            sections: List of section dicts with keys: item, title, content, start, end
        """
        if not sections:
            return

        for i in range(0, len(sections), self.batch_size):
            batch = sections[i:i + self.batch_size]
            try:
                self.graph.query(
                    """
                    UNWIND $batch AS sect
                    MATCH (d:`__Document__` {fileName: $filename})
                    MERGE (s:FilingSection {item: sect.item, fileName: $filename})
                    ON CREATE SET
                        s.title   = sect.title,
                        s.content = sect.content,
                        s.start   = sect.start,
                        s.end     = sect.end
                    ON MATCH SET
                        s.title   = sect.title,
                        s.content = sect.content,
                        s.start   = sect.start,
                        s.end     = sect.end
                    MERGE (d)-[:HAS_SECTION]->(s)
                    """,
                    params={"batch": batch, "filename": filename},
                )
                print(f"Wrote {len(batch)} FilingSection nodes for {filename}")
            except Exception as e:
                print(f"Error writing FilingSection batch for {filename}: {e}")

    def write_document_metadata(self, filename: str, metadata: dict) -> None:
        """
        Set SEC-specific metadata properties on an existing __Document__ node.

        Args:
            filename: Source document filename (matches __Document__.fileName)
            metadata: Dict with keys: form_type, file_path, extension, cik,
                      company_name, filing_date
        """
        try:
            self.graph.query(
                """
                MATCH (d:`__Document__` {fileName: $filename})
                SET d.form_type    = $form_type,
                    d.file_path    = $file_path,
                    d.extension    = $extension,
                    d.cik          = $cik,
                    d.company_name = $company_name,
                    d.filing_date  = $filing_date
                """,
                params={
                    "filename":     filename,
                    "form_type":    metadata.get("form_type", ""),
                    "file_path":    metadata.get("file_path", ""),
                    "extension":    metadata.get("extension", ""),
                    "cik":          metadata.get("cik", ""),
                    "company_name": metadata.get("company_name", ""),
                    "filing_date":  metadata.get("filing_date", ""),
                },
            )
            print(f"Wrote document metadata for {filename}")
        except Exception as e:
            print(f"Error writing document metadata for {filename}: {e}")

    def merge_chunk_relationships(self, chunk_ids: List[str]) -> None:
        """
        Merge relationships between Chunk nodes and Document nodes.

        Args:
            chunk_ids: List of chunk IDs
        """
        if not chunk_ids:
            return

        # Deduplicate chunk_ids to reduce operations
        unique_chunk_ids = list(set(chunk_ids))
        print(f"Merging relationships for {len(unique_chunk_ids)} unique chunks")

        # Dynamic batch size
        optimal_batch_size = min(self.batch_size, max(20, len(unique_chunk_ids) // 5))
        total_batches = (len(unique_chunk_ids) + optimal_batch_size - 1) // optimal_batch_size

        print(f"Merge batch size: {optimal_batch_size}, total batches: {total_batches}")

        # Process in batches to avoid overloading
        for i in range(0, len(unique_chunk_ids), optimal_batch_size):
            batch_chunk_ids = unique_chunk_ids[i:i+optimal_batch_size]
            batch_data = [{"chunk_id": chunk_id} for chunk_id in batch_chunk_ids]

            try:
                merge_query = """
                    UNWIND $batch_data AS data
                    MATCH (c:`__Chunk__` {id: data.chunk_id}), (d:Document{chunk_id:data.chunk_id})
                    WITH c, d
                    MATCH (d)-[r:MENTIONS]->(e)
                    MERGE (c)-[newR:MENTIONS]->(e)
                    ON CREATE SET newR += properties(r)
                    DETACH DELETE d
                """

                self.graph.query(merge_query, params={"batch_data": batch_data})
                print(f"Processed merge batch {i//optimal_batch_size + 1}/{total_batches}")
            except Exception as e:
                print(f"Error merging relationship batch: {e}")
                # If batch fails, try one by one
                for chunk_id in batch_chunk_ids:
                    try:
                        single_query = """
                            MATCH (c:`__Chunk__` {id: $chunk_id}), (d:Document{chunk_id:$chunk_id})
                            WITH c, d
                            MATCH (d)-[r:MENTIONS]->(e)
                            MERGE (c)-[newR:MENTIONS]->(e)
                            ON CREATE SET newR += properties(r)
                            DETACH DELETE d
                        """
                        self.graph.query(single_query, params={"chunk_id": chunk_id})
                    except Exception as e2:
                        print(f"Error processing single chunk relationship: {e2}")
