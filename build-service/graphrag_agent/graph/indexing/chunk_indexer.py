import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Neo4jVector

from graphrag_agent.models.get_models import get_embeddings_model
from graphrag_agent.graph.core import BaseIndexer, connection_manager
from graphrag_agent.config.settings import CHUNK_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class ChunkIndexManager(BaseIndexer):
    """
    Chunk index manager responsible for creating and managing vector indexes for text chunks
    in the Neo4j database. Handles embedding vector computation for __Chunk__ nodes and index
    creation to support subsequent vector-similarity-based RAG queries.
    """

    def __init__(self, refresh_schema: bool = True, batch_size: int = 100, max_workers: int = 4):
        """
        Initialize the chunk index manager.

        Args:
            refresh_schema: Whether to refresh the Neo4j graph database schema
            batch_size: Batch processing size
            max_workers: Number of parallel worker threads
        """
        batch_size = batch_size or CHUNK_BATCH_SIZE
        max_workers = max_workers or DEFAULT_MAX_WORKERS

        super().__init__(batch_size, max_workers)

        # Initialize graph database connection
        self.graph = connection_manager.get_connection()

        # Initialize embedding model
        self.embeddings = get_embeddings_model()

        # Create necessary indexes
        self._create_indexes()

    def _create_indexes(self) -> None:
        """Create necessary indexes to optimize query performance."""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.fileName)",
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.position)"
        ]

        connection_manager.create_multiple_indexes(index_queries)

    def clear_existing_index(self) -> None:
        """Clear any existing standard indexes."""
        connection_manager.drop_index("chunk_embedding")

    def create_chunk_index(self,
                         node_label: str = '__Chunk__',
                         text_property: str = 'text',
                         embedding_property: str = 'embedding') -> Optional[Neo4jVector]:
        """
        Generate embeddings for text chunk nodes and create a vector store interface.

        Args:
            node_label: Label for text chunk nodes
            text_property: Text property used to compute embeddings
            embedding_property: Property name used to store embeddings

        Returns:
            Neo4jVector: Created vector store object
        """
        start_time = time.time()

        # Clear any existing indexes first
        self.clear_existing_index()

        # Fetch all text chunk nodes that need processing
        chunks = self.graph.query(
            f"""
            MATCH (c:`{node_label}`)
            WHERE c.{text_property} IS NOT NULL AND c.{embedding_property} IS NULL
            RETURN id(c) AS neo4j_id, c.id AS chunk_id
            """
        )

        if not chunks:
            print("No text chunk nodes found that need processing")
            # Even if no nodes need processing, try to connect to the existing vector store
            try:
                vector_store = Neo4jVector.from_existing_graph(
                    self.embeddings,
                    node_label=node_label,
                    text_node_properties=[text_property],
                    embedding_node_property=embedding_property
                )

                print("Successfully connected to existing vector index")
                return vector_store
            except Exception as e:
                print(f"Error connecting to vector store: {e}")
                return None

        print(f"Generating embeddings for {len(chunks)} text chunks")

        # Process all text chunks in batches
        self._process_embeddings_in_batches(chunks, node_label, text_property, embedding_property)

        # Connect to the existing vector index rather than creating a new one
        try:
            # Create vector store object
            vector_store = Neo4jVector.from_existing_graph(
                self.embeddings,
                node_label=node_label,
                text_node_properties=[text_property],
                embedding_node_property=embedding_property
            )

            end_time = time.time()
            print(f"Index created successfully, total time: {end_time - start_time:.2f}s")
            print(f"  Embedding computation: {self.embedding_time:.2f}s, database operations: {self.db_time:.2f}s")

            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def _process_embeddings_in_batches(self, chunks: List[Dict[str, Any]],
                                      node_label: str, text_property: str,
                                      embedding_property: str) -> None:
        """
        Process chunk embedding generation in batches.

        Args:
            chunks: List of text chunks
            node_label: Node label
            text_property: Text property
            embedding_property: Embedding property name
        """
        # Get optimal batch size
        chunk_count = len(chunks)
        optimal_batch_size = self.get_optimal_batch_size(chunk_count)

        def process_batch(batch, batch_index):
            # Get text for all chunks in the batch
            chunk_texts = self._get_chunk_texts_batch(batch, text_property)

            # Compute embeddings
            embedding_start = time.time()
            embeddings = self._compute_embeddings_batch(chunk_texts)
            embedding_end = time.time()
            self.embedding_time += (embedding_end - embedding_start)

            # Update the database
            db_start = time.time()
            self._update_embeddings_batch(batch, embeddings, embedding_property)
            db_end = time.time()
            self.db_time += (db_end - db_start)

        # Use the generic batch processing method
        self.batch_process_with_progress(
            chunks,
            process_batch,
            optimal_batch_size,
            "Processing chunk embeddings"
        )

    def _compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embedding vectors for a batch of texts.

        Args:
            texts: List of texts

        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Pre-build embedding tasks
            embedding_tasks = []
            for text in texts:
                # Robustness: ensure text is not empty
                safe_text = text if text and text.strip() else "empty chunk"
                embedding_tasks.append(safe_text)

            # Determine optimal sub-batch size
            embed_batch_size = min(32, len(embedding_tasks))

            # Execute embedding tasks in sub-batches
            for i in range(0, len(embedding_tasks), embed_batch_size):
                sub_batch = embedding_tasks[i:i+embed_batch_size]
                try:
                    # Try using the batch embedding method if available
                    if hasattr(self.embeddings, 'embed_documents'):
                        sub_batch_embeddings = self.embeddings.embed_documents(sub_batch)
                        embeddings.extend(sub_batch_embeddings)
                    else:
                        # Fall back to individual embeddings
                        futures = [executor.submit(self.embeddings.embed_query, text) for text in sub_batch]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                embeddings.append(future.result())
                            except Exception as e:
                                print(f"Embedding computation failed: {e}")
                                # Use zero vector as fallback
                                if hasattr(self.embeddings, 'embedding_size'):
                                    embeddings.append([0.0] * self.embeddings.embedding_size)
                                else:
                                    # Assume a common embedding size
                                    embeddings.append([0.0] * 1536)
                except Exception as e:
                    print(f"Batch embedding processing failed: {e}")
                    # Try individual embeddings as fallback
                    for text in sub_batch:
                        try:
                            embeddings.append(self.embeddings.embed_query(text))
                        except Exception as e2:
                            print(f"Single embedding computation failed: {e2}")
                            # Use zero vector as fallback
                            if hasattr(self.embeddings, 'embedding_size'):
                                embeddings.append([0.0] * self.embeddings.embedding_size)
                            else:
                                embeddings.append([0.0] * 1536)

        return embeddings

    def _get_chunk_texts_batch(self, chunks: List[Dict[str, Any]], text_property: str) -> List[str]:
        """
        Retrieve text content for a batch of text chunks.

        Args:
            chunks: List of text chunks
            text_property: Text property

        Returns:
            List[str]: List of chunk text strings
        """
        # Build query parameters
        chunk_ids = [chunk['neo4j_id'] for chunk in chunks]

        # Use an efficient text extraction query
        query = f"""
        UNWIND $chunk_ids AS id
        MATCH (c) WHERE id(c) = id
        RETURN id, c.{text_property} AS chunk_text
        """

        results = self.graph.query(query, params={"chunk_ids": chunk_ids})

        # Extract text content
        chunk_texts = []
        for row in results:
            text = row.get("chunk_text", "")
            # Ensure text is not empty
            if not text:
                text = f"chunk_{row['id']}"

            chunk_texts.append(text)

        return chunk_texts

    def _update_embeddings_batch(self, chunks: List[Dict[str, Any]],
                                embeddings: List[List[float]],
                                embedding_property: str) -> None:
        """
        Batch update chunk embeddings.

        Args:
            chunks: List of text chunks
            embeddings: Corresponding list of embeddings
            embedding_property: Embedding property name
        """
        # Build update data
        update_data = []
        for i, chunk in enumerate(chunks):
            if i < len(embeddings) and embeddings[i] is not None:
                update_data.append({
                    "id": chunk['neo4j_id'],
                    "embedding": embeddings[i]
                })

        # Batch update
        if update_data:
            try:
                query = f"""
                UNWIND $updates AS update
                MATCH (c) WHERE id(c) = update.id
                SET c.{embedding_property} = update.embedding
                """
                self.graph.query(query, params={"updates": update_data})
            except Exception as e:
                print(f"Batch embedding update failed: {e}")
                # Fall back to individual updates
                for update in update_data:
                    try:
                        single_query = f"""
                        MATCH (c) WHERE id(c) = $id
                        SET c.{embedding_property} = $embedding
                        """
                        self.graph.query(single_query, params={
                            "id": update["id"],
                            "embedding": update["embedding"]
                        })
                    except Exception as e2:
                        print(f"Single embedding update failed (ID: {update['id']}): {e2}")
