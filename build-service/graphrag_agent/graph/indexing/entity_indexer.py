import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Neo4jVector

from graphrag_agent.models.get_models import get_embeddings_model, get_llm_model
from graphrag_agent.graph.core import BaseIndexer, connection_manager
from graphrag_agent.config.settings import ENTITY_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class EntityIndexManager(BaseIndexer):
    """
    Entity index manager responsible for creating and managing vector indexes for entities
    in the Neo4j database. Handles embedding vector computation for entity nodes and index
    creation to support subsequent vector-similarity-based entity queries.
    """

    def __init__(self, refresh_schema: bool = True, batch_size: int = 100, max_workers: int = 4):
        """
        Initialize the entity index manager.

        Args:
            refresh_schema: Whether to refresh the Neo4j graph database schema
            batch_size: Batch processing size
            max_workers: Number of parallel worker threads
        """
        batch_size = batch_size or ENTITY_BATCH_SIZE
        max_workers = max_workers or DEFAULT_MAX_WORKERS

        super().__init__(batch_size, max_workers)

        # Initialize graph database connection
        self.graph = connection_manager.get_connection()

        # Initialize models
        self.embeddings = get_embeddings_model()
        self.llm = get_llm_model()

        # Create necessary indexes
        self._create_indexes()

    def _create_indexes(self) -> None:
        """Create necessary indexes to optimize query performance."""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)"
        ]

        connection_manager.create_multiple_indexes(index_queries)

    def clear_existing_index(self) -> None:
        """Clear existing entity embedding indexes to avoid issues when switching embedding models."""
        connection_manager.drop_index("entity_embedding")
        connection_manager.drop_index("vector")

    def create_entity_index(self,
                          node_label: str = '__Entity__',
                          text_properties: List[str] = ['id', 'description'],
                          embedding_property: str = 'embedding') -> Optional[Neo4jVector]:
        """
        Create a vector index for entities with batch processing and parallel optimization.

        Args:
            node_label: Label for entity nodes
            text_properties: List of text properties used to compute embeddings
            embedding_property: Property name used to store embeddings

        Returns:
            Neo4jVector: Created vector store object
        """
        start_time = time.time()

        # Clear any existing indexes first
        self.clear_existing_index()

        # Fetch all entity nodes that need processing
        entities = self.graph.query(
            f"""
            MATCH (e:`{node_label}`)
            WHERE e.{embedding_property} IS NULL
            RETURN id(e) AS neo4j_id, e.id AS entity_id
            """
        )

        if not entities:
            print("No entity nodes found that need processing")
            return None

        print(f"Generating embeddings for {len(entities)} entities")

        # Process all entities in batches
        self._process_embeddings_in_batches(entities, node_label, text_properties, embedding_property)

        # Create the new vector index
        try:
            vector_store = Neo4jVector.from_existing_graph(
                self.embeddings,
                node_label=node_label,
                text_node_properties=text_properties,
                embedding_node_property=embedding_property
            )

            end_time = time.time()
            print(f"Index created successfully, total time: {end_time - start_time:.2f}s")
            print(f"  Embedding computation: {self.embedding_time:.2f}s, database operations: {self.db_time:.2f}s")

            return vector_store
        except Exception as e:
            print(f"Error creating vector index: {e}")
            return None

    def _process_embeddings_in_batches(self, entities: List[Dict[str, Any]],
                                      node_label: str, text_properties: List[str],
                                      embedding_property: str) -> None:
        """
        Process entity embedding generation in batches.

        Args:
            entities: List of entities
            node_label: Entity node label
            text_properties: Text properties
            embedding_property: Embedding property name
        """
        # Get optimal batch size
        entity_count = len(entities)
        optimal_batch_size = self.get_optimal_batch_size(entity_count)

        def process_batch(batch, batch_index):
            # Get text for all entities in the batch
            entity_texts = self._get_entity_texts_batch(batch, text_properties)

            # Compute embeddings
            embedding_start = time.time()
            embeddings = self._compute_embeddings_batch(entity_texts)
            embedding_end = time.time()
            self.embedding_time += (embedding_end - embedding_start)

            # Update the database
            db_start = time.time()
            self._update_embeddings_batch(batch, embeddings, embedding_property)
            db_end = time.time()
            self.db_time += (db_end - db_start)

        # Use the generic batch processing method
        self.batch_process_with_progress(
            entities,
            process_batch,
            optimal_batch_size,
            "Processing entity embeddings"
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
                safe_text = text if text and text.strip() else "unknown entity"
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

    def _get_entity_texts_batch(self, entities: List[Dict[str, Any]], text_properties: List[str]) -> List[str]:
        """
        Retrieve text content for a batch of entities.

        Args:
            entities: List of entities
            text_properties: List of text properties

        Returns:
            List[str]: List of entity text strings
        """
        # Build query parameters
        entity_ids = [entity['neo4j_id'] for entity in entities]

        # Use an efficient text extraction query
        property_selections = ", ".join([
            f"CASE WHEN e.{prop} IS NOT NULL THEN e.{prop} ELSE '' END AS {prop}_text"
            for prop in text_properties
        ])

        query = f"""
        UNWIND $entity_ids AS id
        MATCH (e) WHERE id(e) = id
        RETURN id, {property_selections}
        """

        results = self.graph.query(query, params={"entity_ids": entity_ids})

        # Combine text properties
        entity_texts = []
        for row in results:
            text_parts = []
            for prop in text_properties:
                prop_text = row.get(f"{prop}_text", "")
                if prop_text:
                    text_parts.append(prop_text)

            # Combine all text properties; ensure there is some content
            combined_text = " ".join(text_parts).strip()
            if not combined_text:
                combined_text = f"entity_{row['id']}"

            entity_texts.append(combined_text)

        return entity_texts

    def _update_embeddings_batch(self, entities: List[Dict[str, Any]],
                                embeddings: List[List[float]],
                                embedding_property: str) -> None:
        """
        Batch update entity embeddings.

        Args:
            entities: List of entities
            embeddings: Corresponding list of embeddings
            embedding_property: Embedding property name
        """
        # Build update data
        update_data = []
        for i, entity in enumerate(entities):
            if i < len(embeddings) and embeddings[i] is not None:
                update_data.append({
                    "id": entity['neo4j_id'],
                    "embedding": embeddings[i]
                })

        # Batch update
        if update_data:
            try:
                query = f"""
                UNWIND $updates AS update
                MATCH (e) WHERE id(e) = update.id
                SET e.{embedding_property} = update.embedding
                """
                self.graph.query(query, params={"updates": update_data})
            except Exception as e:
                print(f"Batch embedding update failed: {e}")
                # Fall back to individual updates
                for update in update_data:
                    try:
                        single_query = f"""
                        MATCH (e) WHERE id(e) = $id
                        SET e.{embedding_property} = $embedding
                        """
                        self.graph.query(single_query, params={
                            "id": update["id"],
                            "embedding": update["embedding"]
                        })
                    except Exception as e2:
                        print(f"Single embedding update failed (ID: {update['id']}): {e2}")
