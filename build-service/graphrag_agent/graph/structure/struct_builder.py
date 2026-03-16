import time
import concurrent.futures
from typing import List, Dict
from langchain_core.documents import Document

from graphrag_agent.graph.core import connection_manager, generate_hash
from graphrag_agent.config.settings import BATCH_SIZE as DEFAULT_BATCH_SIZE
from graphrag_agent.config.settings import MAX_WORKERS as DEFAULT_MAX_WORKERS

class GraphStructureBuilder:
    """
    Graph structure builder responsible for creating and managing document and chunk
    node structures in Neo4j. Handles creation of Document nodes, Chunk nodes,
    and the relationships between them.
    """

    def __init__(self, batch_size=100):
        """
        Initialize the graph structure builder.

        Args:
            batch_size: Batch processing size
        """
        self.graph = connection_manager.get_connection()
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE

    def clear_database(self):
        """Clear the database."""
        clear_query = """
            MATCH (n)
            DETACH DELETE n
            """
        self.graph.query(clear_query)

    def create_document(self, type: str, uri: str, file_name: str, domain: str) -> Dict:
        """
        Create a Document node.

        Args:
            type: Document type
            uri: Document URI
            file_name: File name
            domain: Document domain

        Returns:
            Dict: Created document node info
        """
        query = """
        MERGE(d:`__Document__` {fileName: $file_name})
        SET d.type=$type, d.uri=$uri, d.domain=$domain
        RETURN d;
        """
        doc = self.graph.query(
            query,
            {"file_name": file_name, "type": type, "uri": uri, "domain": domain}
        )
        return doc

    def create_relation_between_chunks(self, file_name: str, chunks: List) -> List[Dict]:
        """
        Create Chunk nodes and establish relationships — batch-optimized version.

        Args:
            file_name: File name
            chunks: List of text chunks

        Returns:
            List[Dict]: List of chunks with IDs and documents
        """
        t0 = time.time()

        current_chunk_id = ""
        lst_chunks_including_hash = []
        batch_data = []
        relationships = []
        offset = 0

        # Process each chunk
        for i, chunk in enumerate(chunks):
            page_content = ''.join(chunk)
            position = i + 1
            current_chunk_id = generate_hash(f"{file_name}_{position}")
            previous_chunk_id = current_chunk_id if i == 0 else lst_chunks_including_hash[-1]['chunk_id']

            if i > 0:
                last_page_content = ''.join(chunks[i-1])
                offset += len(last_page_content)

            firstChunk = (i == 0)

            # Create metadata and Document object
            metadata = {
                "position": position,
                "length": len(page_content),
                "content_offset": offset,
                "tokens": len(chunk)
            }
            chunk_document = Document(page_content=page_content, metadata=metadata)

            # Prepare batch data
            chunk_data = {
                "id": current_chunk_id,
                "pg_content": chunk_document.page_content,
                "position": position,
                "length": chunk_document.metadata["length"],
                "f_name": file_name,
                "previous_id": previous_chunk_id,
                "content_offset": offset,
                "tokens": len(chunk)
            }
            batch_data.append(chunk_data)

            lst_chunks_including_hash.append({
                'chunk_id': current_chunk_id,
                'chunk_doc': chunk_document
            })

            # Build relationship data
            if firstChunk:
                relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
            else:
                relationships.append({
                    "type": "NEXT_CHUNK",
                    "previous_chunk_id": previous_chunk_id,
                    "current_chunk_id": current_chunk_id
                })

            # Flush batch when accumulated data reaches the threshold
            if len(batch_data) >= self.batch_size:
                self._process_batch(file_name, batch_data, relationships)
                batch_data = []
                relationships = []

        # Process remaining data
        if batch_data:
            self._process_batch(file_name, batch_data, relationships)

        t1 = time.time()
        print(f"Relationship creation time: {t1-t0:.2f}s")

        return lst_chunks_including_hash

    def _process_batch(self, file_name: str, batch_data: List[Dict], relationships: List[Dict]):
        """
        Process a batch of chunks and relationships.

        Args:
            file_name: File name
            batch_data: Batch data
            relationships: Relationship data
        """
        if not batch_data:
            return

        # Separate FIRST_CHUNK and NEXT_CHUNK relationships
        first_relationships = [r for r in relationships if r.get("type") == "FIRST_CHUNK"]
        next_relationships = [r for r in relationships if r.get("type") == "NEXT_CHUNK"]

        # Use optimized database operations
        self._create_chunks_and_relationships_optimized(file_name, batch_data, first_relationships, next_relationships)

    def _create_chunks_and_relationships_optimized(self, file_name: str, batch_data: List[Dict],
                                                  first_relationships: List[Dict], next_relationships: List[Dict]):
        """
        Optimized query for creating chunks and relationships — reduces database round-trips.

        Args:
            file_name: File name
            batch_data: Batch data
            first_relationships: List of FIRST_CHUNK relationships
            next_relationships: List of NEXT_CHUNK relationships
        """
        # Combined query: create Chunk nodes and PART_OF relationships
        query_chunks_and_part_of = """
        UNWIND $batch_data AS data
        MERGE (c:`__Chunk__` {id: data.id})
        SET c.text = data.pg_content,
            c.position = data.position,
            c.length = data.length,
            c.fileName = data.f_name,
            c.content_offset = data.content_offset,
            c.tokens = data.tokens
        WITH c, data
        MATCH (d:`__Document__` {fileName: data.f_name})
        MERGE (c)-[:PART_OF]->(d)
        """
        self.graph.query(query_chunks_and_part_of, params={"batch_data": batch_data})

        # Handle FIRST_CHUNK relationships
        if first_relationships:
            query_first_chunk = """
            UNWIND $relationships AS relationship
            MATCH (d:`__Document__` {fileName: $f_name})
            MATCH (c:`__Chunk__` {id: relationship.chunk_id})
            MERGE (d)-[:FIRST_CHUNK]->(c)
            """
            self.graph.query(query_first_chunk, params={
                "f_name": file_name,
                "relationships": first_relationships
            })

        # Handle NEXT_CHUNK relationships
        if next_relationships:
            query_next_chunk = """
            UNWIND $relationships AS relationship
            MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
            MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
            MERGE (pc)-[:NEXT_CHUNK]->(c)
            """
            self.graph.query(query_next_chunk, params={"relationships": next_relationships})

    def parallel_process_chunks(self, file_name: str, chunks: List, max_workers=None) -> List[Dict]:
        """
        Process chunks in parallel to improve throughput for large datasets.

        Args:
            file_name: File name
            chunks: List of text chunks
            max_workers: Number of parallel worker threads

        Returns:
            List[Dict]: List of chunks with IDs and documents
        """
        max_workers = max_workers or DEFAULT_MAX_WORKERS

        if len(chunks) < 100:  # Use standard method for small datasets
            return self.create_relation_between_chunks(file_name, chunks)

        # Split chunks into multiple batches
        chunk_batches = []
        batch_size = max(10, len(chunks) // max_workers)

        for i in range(0, len(chunks), batch_size):
            chunk_batches.append(chunks[i:i+batch_size])

        print(f"Processing {len(chunks)} chunks in parallel: {batch_size} per batch, {len(chunk_batches)} batches total")

        # Define processing function for each batch
        def process_chunk_batch(batch, start_index):
            results = []
            current_chunk_id = ""
            batch_data = []
            relationships = []
            offset = 0

            if start_index > 0 and start_index < len(chunks):
                # Use the previous chunk's ID as the starting point
                current_chunk_id = generate_hash(f"{file_name}_{start_index}")
                # Calculate offset for all preceding chunks
                for j in range(start_index):
                    offset += len(''.join(chunks[j]))

            # Process each chunk within the batch
            for i, chunk in enumerate(batch):
                abs_index = start_index + i
                page_content = ''.join(chunk)
                previous_chunk_id = current_chunk_id
                position = abs_index + 1
                current_chunk_id = generate_hash(f"{file_name}_{position}")

                if i > 0:
                    last_page_content = ''.join(batch[i-1])
                    offset += len(last_page_content)

                firstChunk = (abs_index == 0)

                # Create metadata and Document object
                metadata = {
                    "position": position,
                    "length": len(page_content),
                    "content_offset": offset,
                    "tokens": len(chunk)
                }
                chunk_document = Document(page_content=page_content, metadata=metadata)

                # Prepare batch data
                chunk_data = {
                    "id": current_chunk_id,
                    "pg_content": chunk_document.page_content,
                    "position": position,
                    "length": chunk_document.metadata["length"],
                    "f_name": file_name,
                    "previous_id": previous_chunk_id,
                    "content_offset": offset,
                    "tokens": len(chunk)
                }
                batch_data.append(chunk_data)

                results.append({
                    'chunk_id': current_chunk_id,
                    'chunk_doc': chunk_document
                })

                # Build relationship data
                if firstChunk:
                    relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
                else:
                    relationships.append({
                        "type": "NEXT_CHUNK",
                        "previous_chunk_id": previous_chunk_id,
                        "current_chunk_id": current_chunk_id
                    })

            return {
                "batch_data": batch_data,
                "relationships": relationships,
                "results": results
            }

        # Process all batches in parallel
        start_time = time.time()
        all_batch_data = []
        all_relationships = []
        all_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(process_chunk_batch, batch, i * batch_size): i
                for i, batch in enumerate(chunk_batches)
            }

            # Collect all results
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    result = future.result()
                    all_batch_data.extend(result["batch_data"])
                    all_relationships.extend(result["relationships"])
                    all_results.extend(result["results"])
                except Exception as e:
                    print(f"Error processing batch: {e}")

        # Write to database
        print(f"Parallel processing complete: {len(all_batch_data)} chunks, writing to database")

        # Write to database in batches
        db_batch_size = 500
        for i in range(0, len(all_batch_data), db_batch_size):
            batch = all_batch_data[i:i+db_batch_size]
            rel_batch = [r for r in all_relationships
                         if r.get("type") == "FIRST_CHUNK" and any(b["id"] == r["chunk_id"] for b in batch)
                         or r.get("type") == "NEXT_CHUNK" and any(b["id"] == r["current_chunk_id"] for b in batch)]

            self._create_chunks_and_relationships(file_name, batch, rel_batch)
            print(f"Wrote batch {i//db_batch_size + 1}/{(len(all_batch_data) + db_batch_size - 1) // db_batch_size}")

        end_time = time.time()
        print(f"Database write complete, time: {end_time - start_time:.2f}s")

        return all_results

    def _create_chunks_and_relationships(self, file_name: str, batch_data: List[Dict], relationships: List[Dict]):
        """
        Execute queries to create chunks and relationships.

        Args:
            file_name: File name
            batch_data: Batch data
            relationships: Relationship data
        """
        # Create Chunk nodes and PART_OF relationships
        query_chunk_part_of = """
            UNWIND $batch_data AS data
            MERGE (c:`__Chunk__` {id: data.id})
            SET c.text = data.pg_content,
                c.position = data.position,
                c.length = data.length,
                c.fileName = data.f_name,
                c.content_offset = data.content_offset,
                c.tokens = data.tokens
            WITH data, c
            MATCH (d:`__Document__` {fileName: data.f_name})
            MERGE (c)-[:PART_OF]->(d)
        """
        self.graph.query(query_chunk_part_of, params={"batch_data": batch_data})

        # Create FIRST_CHUNK relationships
        query_first_chunk = """
            UNWIND $relationships AS relationship
            MATCH (d:`__Document__` {fileName: $f_name})
            MATCH (c:`__Chunk__` {id: relationship.chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                    MERGE (d)-[:FIRST_CHUNK]->(c))
        """
        self.graph.query(query_first_chunk, params={
            "f_name": file_name,
            "relationships": relationships
        })

        # Create NEXT_CHUNK relationships
        query_next_chunk = """
            UNWIND $relationships AS relationship
            MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
            WITH c, relationship
            MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
            FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                    MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
        self.graph.query(query_next_chunk, params={"relationships": relationships})
