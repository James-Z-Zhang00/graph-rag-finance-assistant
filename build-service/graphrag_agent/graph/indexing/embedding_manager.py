import time
import concurrent.futures
from typing import List, Dict, Any, Optional

from rich.console import Console

from graphrag_agent.models.get_models import get_embeddings_model
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import EMBEDDING_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class EmbeddingManager:
    """
    Embedding manager that supports incremental embedding updates.

    Main capabilities:
    1. Process only entities and chunks whose embeddings need updating
    2. Efficient batch processing and parallel computation
    3. Maintain embedding update state
    """

    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        """
        Initialize the embedding manager.

        Args:
            batch_size: Batch processing size
            max_workers: Number of parallel worker threads
        """
        self.console = Console()
        self.graph = get_db_manager().graph
        self.embeddings_model = get_embeddings_model()

        self.batch_size = batch_size or EMBEDDING_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS

        # Performance monitoring
        self.embedding_time = 0
        self.db_time = 0
        self.total_time = 0

        # Processing statistics
        self.stats = {
            "entity_updates": 0,
            "chunk_updates": 0,
            "total_updates": 0,
            "errors": 0
        }

    def setup_embedding_tracking(self):
        """Set up embedding update tracking."""
        try:
            # Add entity creation time tracking
            self.graph.query("""
                MATCH (e:`__Entity__`)
                WHERE e.created_at IS NULL
                SET e.created_at = datetime()
            """)

            # Add chunk creation time tracking
            self.graph.query("""
                MATCH (c:`__Chunk__`)
                WHERE c.created_at IS NULL
                SET c.created_at = datetime()
            """)

            self.console.print("[green]Embedding update tracking set up successfully[/green]")

        except Exception as e:
            self.console.print(f"[yellow]Error setting up embedding tracking: {e}[/yellow]")

    def get_entities_needing_update(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get entities whose embeddings need updating.

        Args:
            limit: Maximum number of entities to return

        Returns:
            List[Dict]: List of entities that need updating
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE e.embedding IS NULL
        OR (e.needs_reembedding IS NOT NULL AND e.needs_reembedding = true)
        RETURN elementId(e) AS neo4j_id,
            e.id AS entity_id,
            CASE WHEN e.description IS NOT NULL THEN e.description ELSE e.id END AS text
        LIMIT $limit
        """

        result = self.graph.query(query, params={"limit": limit})
        return result if result else []

    def get_chunks_needing_update(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get chunks whose embeddings need updating.

        Args:
            limit: Maximum number of chunks to return

        Returns:
            List[Dict]: List of chunks that need updating
        """
        query = """
        MATCH (c:`__Chunk__`)
        WHERE c.embedding IS NULL
            OR c.needs_reembedding = true
            OR (c.last_updated IS NOT NULL AND
                (c.last_embedded IS NULL OR c.last_updated > c.last_embedded))
        RETURN elementId(c) AS neo4j_id,
               c.id AS chunk_id,
               c.text AS text
        LIMIT $limit
        """

        result = self.graph.query(query, params={"limit": limit})
        return result if result else []

    def update_entity_embeddings(self, entity_ids: Optional[List[str]] = None) -> int:
        """
        Update entity embeddings.

        Args:
            entity_ids: List of entity IDs to update; if None, auto-detect

        Returns:
            int: Number of entities updated
        """
        start_time = time.time()

        # Fetch entities that need updating
        if entity_ids:
            # If a specific list of entity IDs was provided
            id_list = ", ".join([f"'{eid}'" for eid in entity_ids])
            query = f"""
            MATCH (e:`__Entity__`)
            WHERE e.id IN [{id_list}]
            RETURN elementId(e) AS neo4j_id,
                   e.id AS entity_id,
                   CASE WHEN e.description IS NOT NULL THEN e.description ELSE e.id END AS text
            """
            entities = self.graph.query(query)
        else:
            # Auto-detect entities that need updating
            entities = self.get_entities_needing_update(limit=self.batch_size * 5)

        if not entities:
            self.console.print("[yellow]No entities need embedding updates[/yellow]")
            return 0

        self.console.print(f"[cyan]Starting embedding update for {len(entities)} entities...[/cyan]")

        # Process entities in batches
        updated_count = 0
        for i in range(0, len(entities), self.batch_size):
            batch = entities[i:i+self.batch_size]

            # Extract texts and IDs
            texts = [entity["text"] for entity in batch]
            entity_ids = [entity["entity_id"] for entity in batch]
            neo4j_ids = [entity["neo4j_id"] for entity in batch]

            # Compute embeddings
            embedding_start = time.time()
            try:
                embeddings = self._compute_embeddings_batch(texts)
                self.embedding_time += time.time() - embedding_start

                # Prepare update data
                updates = []
                for j, entity_id in enumerate(entity_ids):
                    if j < len(embeddings) and embeddings[j] is not None:
                        updates.append({
                            "neo4j_id": neo4j_ids[j],
                            "embedding": embeddings[j]
                        })

                # Update the database
                db_start = time.time()
                if updates:
                    query = """
                    UNWIND $updates AS update
                    MATCH (e) WHERE elementId(e) = update.neo4j_id
                    SET e.embedding = update.embedding,
                        e.last_embedded = datetime(),
                        e.needs_reembedding = false
                    RETURN count(e) AS updated
                    """

                    result = self.graph.query(query, params={"updates": updates})
                    batch_updated = result[0]["updated"] if result else 0
                    updated_count += batch_updated

                self.db_time += time.time() - db_start

                self.console.print(f"[green]Batch {i//self.batch_size + 1} complete: "
                                  f"processed {len(batch)} entities, "
                                  f"successfully updated {batch_updated}[/green]")

            except Exception as e:
                self.console.print(f"[red]Error updating entity embeddings: {e}[/red]")
                self.stats["errors"] += 1

        # Update statistics
        self.stats["entity_updates"] += updated_count
        self.stats["total_updates"] += updated_count

        # Calculate total time
        self.total_time += time.time() - start_time

        self.console.print(f"[blue]Entity embedding update complete: updated {updated_count} entities, "
                          f"time: {time.time() - start_time:.2f}s[/blue]")

        return updated_count

    def update_chunk_embeddings(self, chunk_ids: Optional[List[str]] = None) -> int:
        """
        Update chunk embeddings.

        Args:
            chunk_ids: List of chunk IDs to update; if None, auto-detect

        Returns:
            int: Number of chunks updated
        """
        start_time = time.time()

        # Fetch chunks that need updating
        if chunk_ids:
            # If a specific list of chunk IDs was provided
            id_list = ", ".join([f"'{cid}'" for cid in chunk_ids])
            query = f"""
            MATCH (c:`__Chunk__`)
            WHERE c.id IN [{id_list}]
            RETURN elementId(c) AS neo4j_id,
                   c.id AS chunk_id,
                   c.text AS text
            """
            chunks = self.graph.query(query)
        else:
            # Auto-detect chunks that need updating
            chunks = self.get_chunks_needing_update(limit=self.batch_size * 5)

        if not chunks:
            self.console.print("[yellow]No chunks need embedding updates[/yellow]")
            return 0

        self.console.print(f"[cyan]Starting embedding update for {len(chunks)} chunks...[/cyan]")

        # Process chunks in batches
        updated_count = 0
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]

            # Extract texts and IDs
            texts = [chunk["text"] for chunk in batch]
            chunk_ids = [chunk["chunk_id"] for chunk in batch]
            neo4j_ids = [chunk["neo4j_id"] for chunk in batch]

            # Compute embeddings
            embedding_start = time.time()
            try:
                embeddings = self._compute_embeddings_batch(texts)
                self.embedding_time += time.time() - embedding_start

                # Prepare update data
                updates = []
                for j, chunk_id in enumerate(chunk_ids):
                    if j < len(embeddings) and embeddings[j] is not None:
                        updates.append({
                            "neo4j_id": neo4j_ids[j],
                            "embedding": embeddings[j]
                        })

                # Update the database
                db_start = time.time()
                if updates:
                    query = """
                    UNWIND $updates AS update
                    MATCH (c) WHERE elementId(c) = update.neo4j_id
                    SET c.embedding = update.embedding,
                        c.last_embedded = datetime(),
                        c.needs_reembedding = false
                    RETURN count(c) AS updated
                    """

                    result = self.graph.query(query, params={"updates": updates})
                    batch_updated = result[0]["updated"] if result else 0
                    updated_count += batch_updated

                self.db_time += time.time() - db_start

                self.console.print(f"[green]Batch {i//self.batch_size + 1} complete: "
                                  f"processed {len(batch)} chunks, "
                                  f"successfully updated {batch_updated}[/green]")

            except Exception as e:
                self.console.print(f"[red]Error updating chunk embeddings: {e}[/red]")
                self.stats["errors"] += 1

        # Update statistics
        self.stats["chunk_updates"] += updated_count
        self.stats["total_updates"] += updated_count

        # Calculate total time
        self.total_time += time.time() - start_time

        self.console.print(f"[blue]Chunk embedding update complete: updated {updated_count} chunks, "
                          f"time: {time.time() - start_time:.2f}s[/blue]")

        return updated_count

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
                safe_text = text if text and text.strip() else "empty content"
                embedding_tasks.append(safe_text)

            # Determine optimal sub-batch size
            embed_batch_size = min(32, len(embedding_tasks))

            # Execute embedding tasks in sub-batches
            for i in range(0, len(embedding_tasks), embed_batch_size):
                sub_batch = embedding_tasks[i:i+embed_batch_size]
                try:
                    # Try using the batch embedding method if available
                    if hasattr(self.embeddings_model, 'embed_documents'):
                        sub_batch_embeddings = self.embeddings_model.embed_documents(sub_batch)
                        embeddings.extend(sub_batch_embeddings)
                    else:
                        # Fall back to individual embeddings
                        futures = [executor.submit(self.embeddings_model.embed_query, text) for text in sub_batch]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                embeddings.append(future.result())
                            except Exception as e:
                                self.console.print(f"[yellow]Embedding computation failed: {e}[/yellow]")
                                # Use zero vector as fallback
                                if hasattr(self.embeddings_model, 'embedding_size'):
                                    embeddings.append([0.0] * self.embeddings_model.embedding_size)
                                else:
                                    # Assume a common embedding size
                                    embeddings.append([0.0] * 1536)
                except Exception as e:
                    self.console.print(f"[yellow]Batch embedding processing failed: {e}[/yellow]")
                    # Try individual embeddings as fallback
                    for text in sub_batch:
                        try:
                            embeddings.append(self.embeddings_model.embed_query(text))
                        except Exception as e2:
                            self.console.print(f"[yellow]Single embedding computation failed: {e2}[/yellow]")
                            # Use zero vector as fallback
                            if hasattr(self.embeddings_model, 'embedding_size'):
                                embeddings.append([0.0] * self.embeddings_model.embedding_size)
                            else:
                                embeddings.append([0.0] * 1536)

        return embeddings

    def mark_entities_for_update(self, entity_ids: List[str]) -> int:
        """
        Mark entities as needing an embedding update.

        Args:
            entity_ids: List of entity IDs

        Returns:
            int: Number of entities marked
        """
        if not entity_ids:
            return 0

        query = """
        UNWIND $entity_ids AS entity_id
        MATCH (e:`__Entity__` {id: entity_id})
        SET e.needs_reembedding = true,
            e.last_updated = datetime()
        RETURN count(e) AS marked
        """

        result = self.graph.query(query, params={"entity_ids": entity_ids})
        marked = result[0]["marked"] if result else 0

        self.console.print(f"[blue]Marked {marked} entities for embedding update[/blue]")

        return marked

    def mark_chunks_for_update(self, chunk_ids: List[str]) -> int:
        """
        Mark chunks as needing an embedding update.

        Args:
            chunk_ids: List of chunk IDs

        Returns:
            int: Number of chunks marked
        """
        if not chunk_ids:
            return 0

        query = """
        UNWIND $chunk_ids AS chunk_id
        MATCH (c:`__Chunk__` {id: chunk_id})
        SET c.needs_reembedding = true,
            c.last_updated = datetime()
        RETURN count(c) AS marked
        """

        result = self.graph.query(query, params={"chunk_ids": chunk_ids})
        marked = result[0]["marked"] if result else 0

        self.console.print(f"[blue]Marked {marked} chunks for embedding update[/blue]")

        return marked

    def mark_document_chunks_for_update(self, filename: str) -> int:
        """
        Mark all chunks of a document as needing an embedding update.

        Args:
            filename: File name

        Returns:
            int: Number of chunks marked
        """
        query = """
        MATCH (d:`__Document__` {fileName: $filename})<-[:PART_OF]-(c:`__Chunk__`)
        SET c.needs_reembedding = true,
            c.last_updated = datetime()
        RETURN count(c) AS marked
        """

        result = self.graph.query(query, params={"filename": filename})
        marked = result[0]["marked"] if result else 0

        self.console.print(f"[blue]Marked {marked} chunks from file '{filename}' for embedding update[/blue]")

        return marked

    def mark_changed_files_chunks(self, changed_files: List[str]) -> int:
        """
        Mark all chunks of changed files as needing an embedding update.

        Args:
            changed_files: List of changed file paths

        Returns:
            int: Number of chunks marked
        """
        if not changed_files:
            return 0

        total_marked = 0
        for filename in changed_files:
            # Extract file name (without path)
            file_name = filename.split("/")[-1]
            marked = self.mark_document_chunks_for_update(file_name)
            total_marked += marked

        return total_marked

    def display_stats(self):
        """Display statistics."""
        self.console.print("\n[bold cyan]Embedding Update Statistics[/bold cyan]")
        self.console.print(f"[blue]Entity updates: {self.stats['entity_updates']}[/blue]")
        self.console.print(f"[blue]Chunk updates: {self.stats['chunk_updates']}[/blue]")
        self.console.print(f"[blue]Total updates: {self.stats['total_updates']}[/blue]")
        self.console.print(f"[blue]Errors: {self.stats['errors']}[/blue]")

        self.console.print(f"[blue]Total time: {self.total_time:.2f}s — "
                          f"embedding computation: {self.embedding_time:.2f}s ({self.embedding_time/self.total_time*100:.1f}%), "
                          f"database operations: {self.db_time:.2f}s ({self.db_time/self.total_time*100:.1f}%)[/blue]")

    def process(self, entity_limit: int = 500, chunk_limit: int = 500) -> Dict[str, Any]:
        """
        Execute the full embedding update pipeline.

        Args:
            entity_limit: Maximum number of entities to process
            chunk_limit: Maximum number of chunks to process

        Returns:
            Dict: Processing result statistics
        """
        start_time = time.time()

        try:
            # Set up embedding tracking
            self.setup_embedding_tracking()

            # Update entity embeddings
            entity_count = self.update_entity_embeddings(limit=entity_limit)

            # Update chunk embeddings
            chunk_count = self.update_chunk_embeddings(limit=chunk_limit)

            # Display statistics
            self.display_stats()

            # Calculate total time
            self.total_time = time.time() - start_time

            return {
                "entity_updates": entity_count,
                "chunk_updates": chunk_count,
                "total_updates": entity_count + chunk_count,
                "total_time": self.total_time,
                "embedding_time": self.embedding_time,
                "db_time": self.db_time
            }

        except Exception as e:
            self.console.print(f"[red]Error during embedding update process: {e}[/red]")
            raise
