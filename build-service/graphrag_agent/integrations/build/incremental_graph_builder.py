import time
from typing import Dict, List, Any
from pathlib import Path
import os
import tempfile
import shutil

from rich.console import Console
from rich.table import Table

from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.config.prompts import system_template_build_graph, human_template_build_graph
from graphrag_agent.config.settings import (
    entity_types, relationship_types, CHUNK_SIZE, OVERLAP, MAX_WORKERS, BATCH_SIZE,
    FILE_REGISTRY_PATH
)
from graphrag_agent.pipelines.ingestion.document_processor import DocumentProcessor
from graphrag_agent.graph import EntityRelationExtractor, GraphWriter, GraphStructureBuilder
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.integrations.build.incremental.file_change_manager import FileChangeManager
from graphrag_agent.graph.indexing.embedding_manager import EmbeddingManager

class IncrementalGraphUpdater:
    """
    Incremental graph updater, implementing efficient incremental updates based on LightRAG principles.

    Key functions:
    1. Seamlessly integrate new data into the existing graph structure
    2. Update only changed portions, avoiding a full index rebuild
    3. Efficiently merge new and existing graph structures
    4. Protect the integrity of the existing graph
    """

    def __init__(self, files_dir: str, registry_path: str = None):
        """
        Initialize the incremental graph updater.

        Args:
            files_dir: File directory
            registry_path: File registry path; defaults to the path in config
        """
        if registry_path is None:
            registry_path = str(FILE_REGISTRY_PATH)

        self.console = Console()
        self.graph = get_db_manager().graph

        # Initialize file change manager
        self.file_manager = FileChangeManager(files_dir, registry_path)

        # Initialize optimized embedding manager
        self.embedding_manager = EmbeddingManager(batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)

        # Save file directory path
        self.files_dir = files_dir

        # Initialize LLM model
        self.llm = get_llm_model()

        # Initialize document processor
        self.document_processor = DocumentProcessor(files_dir, CHUNK_SIZE, OVERLAP)

        # Initialize graph structure builder
        self.struct_builder = GraphStructureBuilder(batch_size=BATCH_SIZE)

        # Initialize entity-relation extractor
        self.entity_extractor = EntityRelationExtractor(
            self.llm,
            system_template_build_graph,
            human_template_build_graph,
            entity_types,
            relationship_types,
            max_workers=MAX_WORKERS
        )

        # Initialize graph writer
        self.graph_writer = GraphWriter(self.graph, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)

        # Processing statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_time": 0,
            "files_processed": 0,
            "entities_integrated": 0,
            "relations_integrated": 0,
            "entities_updated": 0,
            "chunks_updated": 0
        }

    def detect_changes(self) -> Dict[str, List[str]]:
        """
        Detect file changes.

        Returns:
            Dict: File change information
        """
        return self.file_manager.detect_changes()

    def process_new_files(self, added_files: List[str]) -> Dict[str, Any]:
        """
        Run the full processing pipeline for new files (chunking, entity extraction, relationship creation).

        Args:
            added_files: List of new file paths

        Returns:
            Dict: Processing result statistics
        """
        if not added_files:
            return {"files_processed": 0, "entities_extracted": 0, "relations_created": 0}

        results = {
            "files_processed": 0,
            "entities_extracted": 0,
            "relations_created": 0
        }

        self.console.print(f"[bold cyan]Processing {len(added_files)} new files...[/bold cyan]")

        # Print file paths for debugging
        for file_path in added_files:
            self.console.print(f"[blue]Processing file path: {file_path}[/blue]")
            if not os.path.exists(file_path):
                self.console.print(f"[red]Warning: file does not exist: {file_path}[/red]")

        # Use a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Copy files to temporary directory
                copy_success = False
                for file_path in added_files:
                    try:
                        if os.path.exists(file_path):
                            file_name = os.path.basename(file_path)
                            dest_path = os.path.join(temp_dir, file_name)
                            shutil.copy2(file_path, dest_path)
                            self.console.print(f"[green]Copied {file_path} to temp directory[/green]")
                            copy_success = True
                        elif os.path.exists(os.path.join(self.files_dir, file_path)):
                            # Try treating the file path as relative to files_dir
                            full_path = os.path.join(self.files_dir, file_path)
                            file_name = os.path.basename(file_path)
                            dest_path = os.path.join(temp_dir, file_name)
                            shutil.copy2(full_path, dest_path)
                            self.console.print(f"[green]Copied {full_path} to temp directory[/green]")
                            copy_success = True
                        else:
                            # Last attempt: use just the filename
                            file_name = os.path.basename(file_path)
                            full_path = os.path.join(self.files_dir, file_name)
                            if os.path.exists(full_path):
                                dest_path = os.path.join(temp_dir, file_name)
                                shutil.copy2(full_path, dest_path)
                                self.console.print(f"[green]Copied {full_path} to temp directory[/green]")
                                copy_success = True
                            else:
                                self.console.print(f"[red]Failed to copy file — not found: {file_path}[/red]")
                    except Exception as e:
                        self.console.print(f"[red]Error copying {file_path} to temp directory: {e}[/red]")

                if not copy_success:
                    self.console.print("[red]No files were successfully copied to the temp directory; cannot continue[/red]")
                    return results

                # Save original directory and temporarily reassign
                original_dir = self.document_processor.directory_path
                self.document_processor.directory_path = temp_dir
                self.document_processor.file_reader.directory_path = temp_dir

                # Process files in the temporary directory
                processed_documents = self.document_processor.process_directory()

                # Restore original directory
                self.document_processor.directory_path = original_dir
                self.document_processor.file_reader.directory_path = original_dir

                # Record number of processed files
                if processed_documents:
                    results["files_processed"] = len(processed_documents)
                    self.console.print(f"[green]Successfully processed {len(processed_documents)} files[/green]")

                    # Build graph structure
                    for doc in processed_documents:
                        if "chunks" in doc and doc["chunks"]:
                            # Create document node
                            self.console.print(f"[blue]Creating document node for {doc['filename']}[/blue]")
                            self.struct_builder.create_document(
                                type="local",
                                uri=str(self.files_dir),
                                file_name=doc["filename"],
                                domain="document"
                            )

                            # Create chunk nodes and relationships
                            chunks_count = len(doc['chunks']) if doc['chunks'] else 0
                            self.console.print(f"[blue]Creating {chunks_count} chunk nodes for {doc['filename']}[/blue]")
                            doc["graph_result"] = self.struct_builder.create_relation_between_chunks(
                                doc["filename"],
                                doc["chunks"]
                            )

                    # Prepare data for entity extraction
                    file_contents_format = []
                    for doc in processed_documents:
                        if "chunks" in doc and doc["chunks"]:
                            file_contents_format.append([
                                doc["filename"],
                                doc["content"],
                                doc["chunks"]
                            ])

                    # Extract entities and relationships
                    if file_contents_format:
                        self.console.print(f"[cyan]Starting entity/relationship extraction for {len(file_contents_format)} files[/cyan]")

                        total_chunks = sum(len(content[2]) for content in file_contents_format)
                        self.console.print(f"[blue]Total chunks to process: {total_chunks}[/blue]")

                        processed_chunk_count = 0
                        def progress_callback(i):
                            nonlocal processed_chunk_count
                            processed_chunk_count += 1
                            if processed_chunk_count % 5 == 0 or processed_chunk_count == total_chunks:
                                self.console.print(f"[blue]Processed {processed_chunk_count}/{total_chunks} chunks[/blue]")

                        # Disable cache to ensure new files are processed fresh
                        original_cache_setting = getattr(self.entity_extractor, 'enable_cache', True)
                        self.entity_extractor.enable_cache = False

                        try:
                            processed_contents = self.entity_extractor.process_chunks(
                                file_contents_format,
                                progress_callback=progress_callback
                            )

                            # Restore cache setting
                            self.entity_extractor.enable_cache = original_cache_setting

                            # Log processing results
                            if processed_contents:
                                self.console.print(f"[green]Entity extraction complete, processed {len(processed_contents)} files[/green]")

                                # Debug output
                                for i, content in enumerate(processed_contents):
                                    if len(content) > 3:
                                        entity_data = content[3]
                                        entity_count = sum(1 for data in entity_data if '("entity"' in str(data))
                                        relation_count = sum(1 for data in entity_data if '("relationship"' in str(data))
                                        self.console.print(f"[blue]File {i+1}: {content[0]}, extracted {entity_count} entities and {relation_count} relationships[/blue]")
                                    else:
                                        self.console.print(f"[yellow]File {i+1}: {content[0]}, no entity data returned[/yellow]")

                                # Process results and write to graph database
                                graph_writer_data = []
                                for doc in processed_documents:
                                    if "chunks" in doc and doc["chunks"] and "graph_result" in doc:
                                        # Find corresponding processing result
                                        entity_data = None
                                        for content in processed_contents:
                                            if content[0] == doc["filename"] and len(content) > 3:
                                                entity_data = content[3]
                                                break

                                        if entity_data:
                                            # Estimate entity and relationship counts
                                            entity_count = sum(1 for data in entity_data if '("entity"' in str(data))
                                            relation_count = sum(1 for data in entity_data if '("relationship"' in str(data))
                                            self.console.print(f"[green]Identified {entity_count} entities and {relation_count} relationships in {doc['filename']}[/green]")

                                            # Add to write data
                                            graph_writer_data.append([
                                                doc["filename"],
                                                doc["content"],
                                                doc["chunks"],
                                                doc["graph_result"],
                                                entity_data
                                            ])

                                            # Update statistics
                                            results["entities_extracted"] += entity_count
                                            results["relations_created"] += relation_count

                                # Write to graph database
                                if graph_writer_data:
                                    self.console.print(f"[cyan]Writing graph data for {len(graph_writer_data)} files...[/cyan]")
                                    self.graph_writer.process_and_write_graph_documents(graph_writer_data)
                                    self.console.print(f"[green]Graph data write complete[/green]")
                                else:
                                    self.console.print("[yellow]No valid graph data to write[/yellow]")
                            else:
                                self.console.print("[yellow]Entity extraction returned no valid results[/yellow]")

                        except Exception as e:
                            self.console.print(f"[red]Error during entity extraction: {e}[/red]")
                            import traceback
                            self.console.print(f"[red]{traceback.format_exc()}[/red]")
                    else:
                        self.console.print("[yellow]No text chunks available for entity extraction[/yellow]")
                else:
                    self.console.print("[yellow]No files were processed[/yellow]")

            except Exception as e:
                self.console.print(f"[red]Error processing new files: {e}[/red]")
                import traceback
                self.console.print(f"[red]{traceback.format_exc()}[/red]")

        self.console.print(f"[green]Finished processing {results['files_processed']} new files[/green]")
        if results["entities_extracted"] > 0 or results["relations_created"] > 0:
            self.console.print(f"[green]Extracted {results['entities_extracted']} entities and {results['relations_created']} relationships[/green]")

        return results

    def integrate_new_entities(self, new_entities: List[Dict[str, Any]]) -> int:
        """
        Seamlessly integrate new entities into the existing graph structure.

        Args:
            new_entities: List of new entities

        Returns:
            int: Number of entities integrated
        """
        if not new_entities:
            return 0

        # Merge entities
        query = """
        UNWIND $entities AS entity
        MERGE (e:`__Entity__` {id: entity.id})
        ON CREATE
            SET e += entity.properties,
                e.created_at = datetime(),
                e.needs_reembedding = true
        ON MATCH
            SET e += entity.properties,
                e.last_updated = datetime(),
                e.needs_reembedding = true
        RETURN count(e) AS entity_count
        """

        # Prepare entity data
        entities_data = []
        for entity in new_entities:
            entity_data = {
                "id": entity.get("id", ""),
                "properties": {
                    k: v for k, v in entity.items() if k != "id"
                }
            }
            entities_data.append(entity_data)

        # Execute query
        result = self.graph.query(query, params={"entities": entities_data})
        entity_count = result[0]["entity_count"] if result else 0

        # Update statistics
        self.stats["entities_integrated"] += entity_count

        self.console.print(f"[green]Integrated {entity_count} entities[/green]")

        return entity_count

    def integrate_new_relationships(self, new_relationships: List[Dict[str, Any]]) -> int:
        """
        Seamlessly integrate new relationships into the existing graph structure.

        Args:
            new_relationships: List of new relationships

        Returns:
            int: Number of relationships integrated
        """
        if not new_relationships:
            return 0

        # Merge relationships
        query = """
        UNWIND $relationships AS rel
        MATCH (s:`__Entity__` {id: rel.source_id})
        MATCH (t:`__Entity__` {id: rel.target_id})
        CALL apoc.merge.relationship(s, rel.type,
            {},
            rel.properties,
            t,
            {
                onMatch: {
                    properties: rel.properties,
                    last_updated: datetime()
                },
                onCreateProperties: {
                    created_at: datetime()
                }
            }
        )
        YIELD rel as created
        RETURN count(created) AS rel_count
        """

        try:
            # Execute query
            result = self.graph.query(query, params={"relationships": new_relationships})
            rel_count = result[0]["rel_count"] if result else 0

            # Update statistics
            self.stats["relations_integrated"] += rel_count

            self.console.print(f"[green]Integrated {rel_count} relationships[/green]")

            return rel_count
        except Exception as e:
            self.console.print(f"[yellow]Error integrating relationships (APOC may not be installed): {e}[/yellow]")

            # Fallback method
            integrated = 0
            for rel in new_relationships:
                source_id = rel.get("source_id", "")
                target_id = rel.get("target_id", "")
                rel_type = rel.get("type", "RELATED_TO")
                properties = rel.get("properties", {})

                if source_id and target_id:
                    try:
                        backup_query = """
                        MATCH (s:`__Entity__` {id: $source_id})
                        MATCH (t:`__Entity__` {id: $target_id})
                        MERGE (s)-[r:`%s`]->(t)
                        ON CREATE SET r += $properties, r.created_at = datetime()
                        ON MATCH SET r += $properties, r.last_updated = datetime()
                        RETURN count(r) AS created
                        """ % rel_type

                        backup_result = self.graph.query(
                            backup_query,
                            params={
                                "source_id": source_id,
                                "target_id": target_id,
                                "properties": properties
                            }
                        )

                        if backup_result and backup_result[0]["created"] > 0:
                            integrated += 1
                    except Exception as e2:
                        self.console.print(f"[red]Error integrating relationship via fallback: {e2}[/red]")

            # Update statistics
            self.stats["relations_integrated"] += integrated

            self.console.print(f"[green]Integrated {integrated} relationships via fallback method[/green]")

            return integrated

    def merge_graph_structures(self, old_graph: Dict[str, Any], new_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge the existing graph structure with a new graph structure.

        Args:
            old_graph: Existing graph structure
            new_graph: New graph structure

        Returns:
            Dict: Merged graph structure
        """
        # Merge node sets
        merged_nodes = {**old_graph.get("nodes", {})}
        for node_id, node in new_graph.get("nodes", {}).items():
            if node_id in merged_nodes:
                # If node exists, decide whether to update based on timestamp
                old_timestamp = merged_nodes[node_id].get("last_updated", 0)
                new_timestamp = node.get("last_updated", time.time())

                if new_timestamp > old_timestamp:
                    # Newer data: update node
                    merged_nodes[node_id] = {**merged_nodes[node_id], **node}
                    merged_nodes[node_id]["last_updated"] = new_timestamp
                    merged_nodes[node_id]["needs_reembedding"] = True
            else:
                # New node: add directly
                merged_nodes[node_id] = node
                if "needs_reembedding" not in merged_nodes[node_id]:
                    merged_nodes[node_id]["needs_reembedding"] = True

        # Merge edge sets, avoiding duplicate relationships
        merged_edges = {}

        # Add edges from the old graph first
        for edge in old_graph.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("type", "")

            key = f"{source}_{rel_type}_{target}"
            merged_edges[key] = edge

        # Add or update edges from the new graph
        for edge in new_graph.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("type", "")

            key = f"{source}_{rel_type}_{target}"

            if key in merged_edges:
                # If relationship exists, decide whether to update based on timestamp
                old_timestamp = merged_edges[key].get("last_updated", 0)
                new_timestamp = edge.get("last_updated", time.time())

                if new_timestamp > old_timestamp:
                    merged_edges[key] = {**merged_edges[key], **edge}
                    merged_edges[key]["last_updated"] = new_timestamp
            else:
                merged_edges[key] = edge

        return {
            "nodes": merged_nodes,
            "edges": list(merged_edges.values())
        }

    def update_changed_file_embeddings(self, changed_files: List[str]) -> Dict[str, int]:
        """
        Update embeddings for changed files.

        Args:
            changed_files: List of changed files

        Returns:
            Dict: Update statistics
        """
        if not changed_files:
            return {"entities": 0, "chunks": 0}

        # Mark chunks from changed files as needing embedding updates
        marked_chunks = self.embedding_manager.mark_changed_files_chunks(changed_files)

        # Find entities linked to those chunks and mark them for re-embedding
        query = """
        MATCH (c:`__Chunk__`)-[:MENTIONS]->(e:`__Entity__`)
        WHERE c.fileName IN $filenames OR c.needs_reembedding = true
        SET e.needs_reembedding = true,
            e.last_updated = datetime()
        RETURN count(DISTINCT e) AS entity_count
        """

        # Use filenames without path prefix
        filenames = [Path(file).name for file in changed_files]

        result = self.graph.query(query, params={"filenames": filenames})
        marked_entities = result[0]["entity_count"] if result else 0

        self.console.print(f"[blue]Marked {marked_chunks} chunks and {marked_entities} entities for re-embedding[/blue]")

        # Execute updates
        updated_entities = self.embedding_manager.update_entity_embeddings()
        updated_chunks = self.embedding_manager.update_chunk_embeddings()

        # Update statistics
        self.stats["entities_updated"] += updated_entities
        self.stats["chunks_updated"] += updated_chunks

        return {
            "entities": updated_entities,
            "chunks": updated_chunks
        }

    def process_deleted_files(self, deleted_files: List[str]) -> int:
        """
        Handle deleted files.

        Args:
            deleted_files: List of deleted files

        Returns:
            int: Number of nodes deleted
        """
        if not deleted_files:
            return 0

        self.console.print(f"[cyan]Handling {len(deleted_files)} deleted files...[/cyan]")

        deleted_count = 0
        for file_path in deleted_files:
            file_name = Path(file_path).name

            # Find all Chunk nodes associated with the file
            chunk_query = """
            MATCH (d:`__Document__` {fileName: $fileName})<-[:PART_OF]-(c:`__Chunk__`)
            RETURN collect(c.id) AS chunk_ids, count(c) AS chunk_count
            """

            chunk_result = self.graph.query(chunk_query, params={"fileName": file_name})

            if not chunk_result or not chunk_result[0]["chunk_ids"]:
                self.console.print(f"[yellow]No data to delete for file {file_name}[/yellow]")
                continue

            chunk_ids = chunk_result[0]["chunk_ids"]
            chunk_count = chunk_result[0]["chunk_count"]

            # Find entities linked to those chunks, excluding entities referenced by other chunks
            entity_query = """
            MATCH (c:`__Chunk__`)-[:MENTIONS]->(e:`__Entity__`)
            WHERE c.id IN $chunk_ids
            WITH e, count(c) AS references
            MATCH (chunk:`__Chunk__`)-[:MENTIONS]->(e)
            WITH e, references, count(chunk) AS total_references
            WHERE references = total_references
              AND NOT e.manual_edit = true  // exclude manually edited entities
              AND NOT e.protected = true    // exclude protected entities
            RETURN collect(e.id) AS entity_ids, count(e) AS entity_count
            """

            entity_result = self.graph.query(entity_query, params={"chunk_ids": chunk_ids})

            entity_ids = []
            entity_count = 0
            if entity_result and entity_result[0]["entity_ids"]:
                entity_ids = entity_result[0]["entity_ids"]
                entity_count = entity_result[0]["entity_count"]

            # Delete all data associated with the file
            delete_query = """
            // Delete document node and its relationships
            MATCH (d:`__Document__` {fileName: $fileName})
            OPTIONAL MATCH (d)-[r]-()
            DELETE r

            // Delete Chunk nodes and their relationships
            WITH d
            MATCH (c:`__Chunk__`)-[r1:PART_OF]->(d)
            OPTIONAL MATCH (c)-[r2]-()
            WHERE NOT type(r2) = 'PART_OF'
            DELETE r2

            // Delete orphaned entity nodes
            WITH d, collect(c.id) as chunk_ids
            UNWIND $entity_ids AS entity_id
            MATCH (e:`__Entity__` {id: entity_id})
            WHERE NOT e.manual_edit = true AND NOT e.protected = true
            DELETE e

            // Delete Chunk nodes
            WITH d, chunk_ids
            MATCH (c:`__Chunk__`)
            WHERE c.id IN chunk_ids
            DELETE c

            // Finally delete the document node
            DELETE d

            RETURN count(d) AS deleted_docs
            """

            delete_result = self.graph.query(delete_query, params={
                "fileName": file_name,
                "entity_ids": entity_ids
            })

            deleted_docs = delete_result[0]["deleted_docs"] if delete_result else 0
            file_deleted_count = chunk_count + entity_count + deleted_docs
            deleted_count += file_deleted_count

            self.console.print(f"[blue]Deleted data for {file_name}: "
                              f"{chunk_count} chunk nodes, "
                              f"{entity_count} entity nodes, "
                              f"{deleted_docs} document nodes[/blue]")

        self.console.print(f"[green]Deleted file processing complete, {deleted_count} nodes removed[/green]")
        return deleted_count

    def export_graph_structure(self) -> Dict[str, Any]:
        """
        Export the current graph structure.

        Returns:
            Dict: Graph structure data
        """
        # Query all entity nodes
        node_query = """
        MATCH (e:`__Entity__`)
        RETURN e.id AS id,
               labels(e) AS labels,
               properties(e) AS properties,
               e.last_updated AS last_updated
        """

        node_result = self.graph.query(node_query)

        # Query all relationships
        edge_query = """
        MATCH (s:`__Entity__`)-[r]->(t:`__Entity__`)
        RETURN s.id AS source,
               t.id AS target,
               type(r) AS type,
               properties(r) AS properties,
               CASE WHEN r.last_updated IS NOT NULL THEN r.last_updated ELSE null END AS last_updated
        """

        edge_result = self.graph.query(edge_query)

        # Build graph structure
        nodes = {}
        for node in node_result:
            node_id = node["id"]
            nodes[node_id] = {
                "id": node_id,
                "labels": node["labels"],
                "properties": node["properties"],
                "last_updated": node["last_updated"] if node["last_updated"] else time.time()
            }

        edges = []
        for edge in edge_result:
            edges.append({
                "source": edge["source"],
                "target": edge["target"],
                "type": edge["type"],
                "properties": edge["properties"],
                "last_updated": edge["last_updated"] if edge["last_updated"] else time.time()
            })

        return {
            "nodes": nodes,
            "edges": edges
        }

    def import_graph_structure(self, graph_structure: Dict[str, Any]) -> Dict[str, int]:
        """
        Import a graph structure.

        Args:
            graph_structure: Graph structure data

        Returns:
            Dict: Import statistics
        """
        if not graph_structure:
            return {"nodes": 0, "edges": 0}

        # Import nodes
        nodes = list(graph_structure.get("nodes", {}).values())
        node_count = 0

        if nodes:
            # Prepare node data
            node_data = []
            for node in nodes:
                node_data.append({
                    "id": node["id"],
                    "labels": node.get("labels", ["__Entity__"]),
                    "properties": node.get("properties", {})
                })

            # Execute import
            node_query = """
            UNWIND $nodes AS node
            CALL apoc.merge.node(
                node.labels,
                {id: node.id},
                node.properties,
                node.properties
            )
            YIELD node as n
            RETURN count(n) AS node_count
            """

            try:
                node_result = self.graph.query(node_query, params={"nodes": node_data})
                node_count = node_result[0]["node_count"] if node_result else 0
            except Exception as e:
                self.console.print(f"[yellow]Error importing nodes (APOC may not be installed): {e}[/yellow]")

                # Fallback method
                simple_node_query = """
                UNWIND $nodes AS node
                MERGE (n:`__Entity__` {id: node.id})
                SET n += node.properties
                RETURN count(n) AS node_count
                """

                node_result = self.graph.query(simple_node_query, params={"nodes": node_data})
                node_count = node_result[0]["node_count"] if node_result else 0

        # Import edges
        edges = graph_structure.get("edges", [])
        edge_count = 0

        if edges:
            self.integrate_new_relationships(edges)
            edge_count = len(edges)

        return {
            "nodes": node_count,
            "edges": edge_count
        }

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Dict: Statistics
        """
        # Node statistics
        node_query = """
        MATCH (n)
        RETURN
            count(n) AS total_nodes,
            sum(CASE WHEN n:`__Document__` THEN 1 ELSE 0 END) AS doc_count,
            sum(CASE WHEN n:`__Chunk__` THEN 1 ELSE 0 END) AS chunk_count,
            sum(CASE WHEN n:`__Entity__` THEN 1 ELSE 0 END) AS entity_count
        """

        node_result = self.graph.query(node_query)

        # Relationship statistics
        rel_query = """
        MATCH ()-[r]->()
        RETURN count(r) AS total_relations,
               count(DISTINCT type(r)) AS relation_types
        """

        rel_result = self.graph.query(rel_query)

        # Embedding statistics
        embedding_query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN
            count(n) AS nodes_with_embedding,
            sum(CASE WHEN n:`__Entity__` THEN 1 ELSE 0 END) AS entities_with_embedding,
            sum(CASE WHEN n:`__Chunk__` THEN 1 ELSE 0 END) AS chunks_with_embedding
        """

        embedding_result = self.graph.query(embedding_query)

        # Combine statistics
        stats = {
            "total_nodes": node_result[0]["total_nodes"] if node_result else 0,
            "document_count": node_result[0]["doc_count"] if node_result else 0,
            "chunk_count": node_result[0]["chunk_count"] if node_result else 0,
            "entity_count": node_result[0]["entity_count"] if node_result else 0,
            "total_relations": rel_result[0]["total_relations"] if rel_result else 0,
            "relation_types": rel_result[0]["relation_types"] if rel_result else 0,
            "nodes_with_embedding": embedding_result[0]["nodes_with_embedding"] if embedding_result else 0,
            "entities_with_embedding": embedding_result[0]["entities_with_embedding"] if embedding_result else 0,
            "chunks_with_embedding": embedding_result[0]["chunks_with_embedding"] if embedding_result else 0
        }

        return stats

    def display_graph_statistics(self):
        """Display graph statistics"""
        stats = self.get_graph_statistics()

        # Create statistics table
        stats_table = Table(title="Graph Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", justify="right")

        # Node statistics
        stats_table.add_row("Total nodes", str(stats["total_nodes"]))
        stats_table.add_row("Document nodes", str(stats["document_count"]))
        stats_table.add_row("Chunk nodes", str(stats["chunk_count"]))
        stats_table.add_row("Entity nodes", str(stats["entity_count"]))

        # Relationship statistics
        stats_table.add_row("Total relationships", str(stats["total_relations"]))
        stats_table.add_row("Relationship types", str(stats["relation_types"]))

        # Embedding statistics
        stats_table.add_row("Nodes with embeddings", str(stats["nodes_with_embedding"]))
        stats_table.add_row("Entities with embeddings", str(stats["entities_with_embedding"]))
        stats_table.add_row("Chunks with embeddings", str(stats["chunks_with_embedding"]))

        self.console.print(stats_table)

    def process_incremental_update(self) -> Dict[str, Any]:
        """
        Execute the incremental update pipeline.

        Returns:
            Dict: Update result statistics
        """
        start_time = time.time()
        self.stats["start_time"] = start_time

        try:
            # 1. Detect file changes
            self.console.print("[bold cyan]Detecting file changes...[/bold cyan]")
            changes = self.detect_changes()

            # Separate added, modified, and deleted files
            added_files = changes.get("added", [])
            modified_files = changes.get("modified", [])
            deleted_files = changes.get("deleted", [])

            changed_files = modified_files  # Only modified files need embedding updates
            self.stats["files_processed"] = len(added_files) + len(modified_files) + len(deleted_files)

            if not added_files and not changed_files and not deleted_files:
                self.console.print("[yellow]No file changes detected[/yellow]")
                return self.stats

            # 2. Handle deleted files
            if deleted_files:
                self.console.print("[bold cyan]Processing deleted files...[/bold cyan]")
                self.process_deleted_files(deleted_files)

            # 3. Process new files — run full pipeline
            if added_files:
                self.console.print("[bold cyan]Processing new files...[/bold cyan]")
                new_file_results = self.process_new_files(added_files)
                self.stats["entities_integrated"] += new_file_results.get("entities_extracted", 0)
                self.stats["relations_integrated"] += new_file_results.get("relations_created", 0)

            # 4. Update embeddings for changed files
            if changed_files:
                self.console.print("[bold cyan]Updating embeddings for changed files...[/bold cyan]")
                embedding_stats = self.update_changed_file_embeddings(changed_files)

                self.console.print(f"[green]Updated entity embeddings: {embedding_stats['entities']}[/green]")
                self.console.print(f"[green]Updated chunk embeddings: {embedding_stats['chunks']}[/green]")

            # 5. Update file registry
            self.file_manager.update_registry()

            # 6. Display graph statistics
            self.console.print("[bold cyan]Graph Statistics[/bold cyan]")
            self.display_graph_statistics()

            # Compute total time
            end_time = time.time()
            self.stats["end_time"] = end_time
            self.stats["total_time"] = end_time - start_time

            self.console.print("\n[bold green]Incremental update complete![/bold green]")
            self.console.print(f"[green]Total time: {self.stats['total_time']:.2f}s[/green]")
            self.console.print(f"[green]Files processed: {self.stats['files_processed']}[/green]")
            if added_files:
                self.console.print(f"[green]New entities: {self.stats['entities_integrated']}[/green]")
                self.console.print(f"[green]New relationships: {self.stats['relations_integrated']}[/green]")

            return self.stats

        except Exception as e:
            self.console.print(f"[red]Error during incremental update: {e}[/red]")

            end_time = time.time()
            self.stats["end_time"] = end_time
            self.stats["total_time"] = end_time - start_time

            raise
