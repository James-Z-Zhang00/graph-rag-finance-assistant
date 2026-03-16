import time
from typing import Dict, List, Any, Tuple
from rich.console import Console
from rich.table import Table
from graphrag_agent.config.neo4jdb import get_db_manager

class GraphConsistencyValidator:
    """
    Graph consistency validator responsible for verifying the structural and content
    integrity of the knowledge graph.

    Main capabilities:
    1. Check for orphan nodes
    2. Verify relationship chain integrity
    3. Repair common consistency issues
    """

    def __init__(self):
        """Initialize the graph consistency validator."""
        self.console = Console()
        self.graph = get_db_manager().graph

        # Performance timers
        self.validation_time = 0
        self.repair_time = 0

        # Validation statistics
        self.validation_stats = {
            "orphan_entities": 0,
            "dangling_chunks": 0,
            "empty_chunks": 0,
            "broken_doc_links": 0,
            "broken_chunk_chains": 0,
            "total_issues": 0,
            "repaired_issues": 0
        }

    def check_orphan_entities(self) -> Tuple[List[str], int]:
        """
        Check for orphan entity nodes (not referenced by any chunk).

        Returns:
            Tuple: List of orphan entity IDs and their count
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE NOT (e)<-[:MENTIONS]-()
          AND NOT e.manual_edit = true
          AND NOT e.protected = true
        RETURN e.id AS entity_id, count(e) AS count
        """

        result = self.graph.query(query)

        orphan_ids = []
        orphan_count = 0

        if result:
            orphan_count = result[0]["count"]
            # Fetch up to 1000 orphan entity IDs
            id_query = """
            MATCH (e:`__Entity__`)
            WHERE NOT (e)<-[:MENTIONS]-()
              AND NOT e.manual_edit = true
              AND NOT e.protected = true
            RETURN e.id AS entity_id
            LIMIT 1000
            """
            id_result = self.graph.query(id_query)
            orphan_ids = [r["entity_id"] for r in id_result]

        self.validation_stats["orphan_entities"] = orphan_count

        return orphan_ids, orphan_count

    def check_dangling_chunks(self) -> Tuple[List[str], int]:
        """
        Check for dangling chunk nodes (not linked to any document).

        Returns:
            Tuple: List of dangling chunk IDs and their count
        """
        query = """
        MATCH (c:`__Chunk__`)
        WHERE NOT (c)-[:PART_OF]->()
        RETURN c.id AS chunk_id, count(c) AS count
        """

        result = self.graph.query(query)

        dangling_ids = []
        dangling_count = 0

        if result:
            dangling_count = result[0]["count"]
            # Fetch up to 1000 dangling chunk IDs
            id_query = """
            MATCH (c:`__Chunk__`)
            WHERE NOT (c)-[:PART_OF]->()
            RETURN c.id AS chunk_id
            LIMIT 1000
            """
            id_result = self.graph.query(id_query)
            dangling_ids = [r["chunk_id"] for r in id_result]

        self.validation_stats["dangling_chunks"] = dangling_count

        return dangling_ids, dangling_count

    def check_empty_chunks(self) -> Tuple[List[str], int]:
        """
        Check for empty chunk nodes (no text content).

        Returns:
            Tuple: List of empty chunk IDs and their count
        """
        query = """
        MATCH (c:`__Chunk__`)
        WHERE c.text IS NULL OR c.text = ''
        RETURN c.id AS chunk_id, count(c) AS count
        """

        result = self.graph.query(query)

        empty_ids = []
        empty_count = 0

        if result:
            empty_count = result[0]["count"]
            # Fetch up to 1000 empty chunk IDs
            id_query = """
            MATCH (c:`__Chunk__`)
            WHERE c.text IS NULL OR c.text = ''
            RETURN c.id AS chunk_id
            LIMIT 1000
            """
            id_result = self.graph.query(id_query)
            empty_ids = [r["chunk_id"] for r in id_result]

        self.validation_stats["empty_chunks"] = empty_count

        return empty_ids, empty_count

    def check_broken_doc_links(self) -> int:
        """
        Check whether document links are intact (each Document should have a FIRST_CHUNK relationship).

        Returns:
            int: Number of documents with broken links
        """
        query = """
        MATCH (d:`__Document__`)
        WHERE NOT (d)-[:FIRST_CHUNK]->()
        RETURN count(d) AS count
        """

        result = self.graph.query(query)

        count = result[0]["count"] if result else 0
        self.validation_stats["broken_doc_links"] = count

        return count

    def check_broken_chunk_chains(self) -> int:
        """
        Check whether text chunk chains are intact (predecessor/successor relationships).

        Returns:
            int: Number of broken chains
        """
        query = """
        MATCH (c:`__Chunk__`)-[:PART_OF]->(d:`__Document__`)
        WHERE c.position > 1 AND NOT (c)<-[:NEXT_CHUNK]-()
        RETURN count(c) AS count
        """

        result = self.graph.query(query)

        count = result[0]["count"] if result else 0
        self.validation_stats["broken_chunk_chains"] = count

        return count

    def validate_graph(self) -> Dict[str, Any]:
        """
        Perform a comprehensive graph validation.

        Returns:
            Dict: Validation result statistics
        """
        start_time = time.time()

        # 1. Check for orphan entities
        orphan_ids, orphan_count = self.check_orphan_entities()
        if orphan_count > 0:
            self.console.print(f"[yellow]Found {orphan_count} orphan entity nodes[/yellow]")

        # 2. Check for dangling chunks
        dangling_ids, dangling_count = self.check_dangling_chunks()
        if dangling_count > 0:
            self.console.print(f"[yellow]Found {dangling_count} dangling chunk nodes[/yellow]")

        # 3. Check for empty chunks
        empty_ids, empty_count = self.check_empty_chunks()
        if empty_count > 0:
            self.console.print(f"[yellow]Found {empty_count} empty chunk nodes[/yellow]")

        # 4. Check document links
        broken_doc_count = self.check_broken_doc_links()
        if broken_doc_count > 0:
            self.console.print(f"[yellow]Found {broken_doc_count} documents without a FIRST_CHUNK relationship[/yellow]")

        # 5. Check chunk chain integrity
        broken_chain_count = self.check_broken_chunk_chains()
        if broken_chain_count > 0:
            self.console.print(f"[yellow]Found {broken_chain_count} broken chunk chains[/yellow]")

        # Calculate total issues
        total_issues = (orphan_count + dangling_count + empty_count +
                       broken_doc_count + broken_chain_count)
        self.validation_stats["total_issues"] = total_issues

        # Record validation time
        self.validation_time = time.time() - start_time

        self.console.print(f"[blue]Graph validation complete in {self.validation_time:.2f}s[/blue]")
        self.console.print(f"[blue]Total consistency issues found: {total_issues}[/blue]")

        return {
            "validation_time": self.validation_time,
            "validation_stats": self.validation_stats,
            "orphan_ids": orphan_ids,
            "dangling_ids": dangling_ids,
            "empty_ids": empty_ids
        }

    def repair_orphan_entities(self, orphan_ids: List[str] = None) -> int:
        """
        Repair orphan entity nodes (delete or mark them).

        Args:
            orphan_ids: List of orphan entity IDs to repair; if None, auto-detect

        Returns:
            int: Number of nodes repaired
        """
        if orphan_ids is None:
            orphan_ids, _ = self.check_orphan_entities()

        if not orphan_ids:
            return 0

        # Delete orphan entities
        delete_query = """
        UNWIND $orphan_ids AS entity_id
        MATCH (e:`__Entity__` {id: entity_id})
        WHERE NOT (e)<-[:MENTIONS]-()
          AND NOT e.manual_edit = true
          AND NOT e.protected = true
        DELETE e
        RETURN count(*) AS deleted
        """

        result = self.graph.query(delete_query, params={"orphan_ids": orphan_ids})

        deleted = result[0]["deleted"] if result else 0

        self.console.print(f"[green]Deleted {deleted} orphan entity nodes[/green]")

        return deleted

    def repair_dangling_chunks(self, dangling_ids: List[str] = None) -> int:
        """
        Repair dangling chunk nodes (delete them).

        Args:
            dangling_ids: List of dangling chunk IDs to repair; if None, auto-detect

        Returns:
            int: Number of nodes repaired
        """
        if dangling_ids is None:
            dangling_ids, _ = self.check_dangling_chunks()

        if not dangling_ids:
            return 0

        # Delete dangling chunks
        delete_query = """
        UNWIND $dangling_ids AS chunk_id
        MATCH (c:`__Chunk__` {id: chunk_id})
        WHERE NOT (c)-[:PART_OF]->()
        DETACH DELETE c
        RETURN count(*) AS deleted
        """

        result = self.graph.query(delete_query, params={"dangling_ids": dangling_ids})

        deleted = result[0]["deleted"] if result else 0

        self.console.print(f"[green]Deleted {deleted} dangling chunk nodes[/green]")

        return deleted

    def repair_empty_chunks(self, empty_ids: List[str] = None) -> int:
        """
        Repair empty chunk nodes (add placeholder text or delete them).

        Args:
            empty_ids: List of empty chunk IDs to repair; if None, auto-detect

        Returns:
            int: Number of nodes repaired
        """
        if empty_ids is None:
            empty_ids, _ = self.check_empty_chunks()

        if not empty_ids:
            return 0

        # Add placeholder text to empty chunks
        repair_query = """
        UNWIND $empty_ids AS chunk_id
        MATCH (c:`__Chunk__` {id: chunk_id})
        WHERE c.text IS NULL OR c.text = ''
        SET c.text = '[Empty Chunk]', c.repaired = true
        RETURN count(*) AS repaired
        """

        result = self.graph.query(repair_query, params={"empty_ids": empty_ids})

        repaired = result[0]["repaired"] if result else 0

        self.console.print(f"[green]Repaired {repaired} empty chunk nodes[/green]")

        return repaired

    def repair_broken_doc_links(self) -> int:
        """
        Repair broken document links (create missing FIRST_CHUNK relationships).

        Returns:
            int: Number of relationships repaired
        """
        repair_query = """
        MATCH (d:`__Document__`)
        WHERE NOT (d)-[:FIRST_CHUNK]->()

        MATCH (c:`__Chunk__`)-[:PART_OF]->(d)
        WHERE c.position = 1 OR c.position IS NULL

        WITH d, c ORDER BY c.position ASC LIMIT 1
        MERGE (d)-[r:FIRST_CHUNK]->(c)

        RETURN count(r) AS repaired
        """

        result = self.graph.query(repair_query)

        repaired = result[0]["repaired"] if result else 0

        self.console.print(f"[green]Repaired {repaired} broken document links[/green]")

        return repaired

    def repair_broken_chunk_chains(self) -> int:
        """
        Repair broken chunk chains (rebuild NEXT_CHUNK relationships).

        Returns:
            int: Number of relationships repaired
        """
        repair_query = """
        MATCH (d:`__Document__`)
        WITH d
        MATCH (c1:`__Chunk__`)-[:PART_OF]->(d)
        WHERE c1.position IS NOT NULL
        WITH d, c1 ORDER BY c1.position ASC
        WITH d, collect(c1) AS chunks
        UNWIND range(0, size(chunks)-2) AS i
        WITH d, chunks[i] AS current, chunks[i+1] AS next
        WHERE NOT (current)-[:NEXT_CHUNK]->(next)
        MERGE (current)-[r:NEXT_CHUNK]->(next)
        RETURN count(r) AS repaired
        """

        result = self.graph.query(repair_query)

        repaired = result[0]["repaired"] if result else 0

        self.console.print(f"[green]Repaired {repaired} broken chunk chain links[/green]")

        return repaired

    def repair_graph(self) -> Dict[str, Any]:
        """
        Execute graph repair operations.

        Returns:
            Dict: Repair result statistics
        """
        start_time = time.time()

        # Run full validation first
        validation_result = self.validate_graph()

        # Perform repairs based on validation results
        repairs = {
            "orphan_entities": self.repair_orphan_entities(validation_result.get("orphan_ids", [])),
            "dangling_chunks": self.repair_dangling_chunks(validation_result.get("dangling_ids", [])),
            "empty_chunks": self.repair_empty_chunks(validation_result.get("empty_ids", [])),
            "broken_doc_links": self.repair_broken_doc_links(),
            "broken_chunk_chains": self.repair_broken_chunk_chains()
        }

        # Calculate total repaired count
        total_repaired = sum(repairs.values())
        self.validation_stats["repaired_issues"] = total_repaired

        # Record repair time
        self.repair_time = time.time() - start_time

        self.console.print(f"[blue]Graph repair complete in {self.repair_time:.2f}s[/blue]")
        self.console.print(f"[blue]Total consistency issues repaired: {total_repaired}[/blue]")

        return {
            "validation_time": self.validation_time,
            "repair_time": self.repair_time,
            "validation_stats": self.validation_stats,
            "repairs": repairs
        }

    def display_graph_stats(self):
        """Display graph statistics."""
        # Get node and relationship statistics
        stats_query = """
        MATCH (n)
        RETURN
            count(n) AS total_nodes,
            sum(CASE WHEN n:`__Document__` THEN 1 ELSE 0 END) AS doc_count,
            sum(CASE WHEN n:`__Chunk__` THEN 1 ELSE 0 END) AS chunk_count,
            sum(CASE WHEN n:`__Entity__` THEN 1 ELSE 0 END) AS entity_count
        """

        stats_result = self.graph.query(stats_query)

        if not stats_result:
            self.console.print("[yellow]Unable to retrieve graph statistics[/yellow]")
            return

        node_stats = stats_result[0]

        # Get relationship statistics
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
        """

        rel_result = self.graph.query(rel_query)

        # Display node statistics table
        node_table = Table(title="Graph Node Statistics")
        node_table.add_column("Node Type", style="cyan")
        node_table.add_column("Count", justify="right")

        node_table.add_row("__Document__", str(node_stats["doc_count"]))
        node_table.add_row("__Chunk__", str(node_stats["chunk_count"]))
        node_table.add_row("__Entity__", str(node_stats["entity_count"]))
        node_table.add_row("Total", str(node_stats["total_nodes"]), style="bold")

        self.console.print(node_table)

        # Display relationship statistics table
        if rel_result:
            rel_table = Table(title="Graph Relationship Statistics")
            rel_table.add_column("Relationship Type", style="cyan")
            rel_table.add_column("Count", justify="right")

            total_rels = 0
            for rel in rel_result:
                rel_table.add_row(rel["rel_type"], str(rel["count"]))
                total_rels += rel["count"]

            rel_table.add_row("Total", str(total_rels), style="bold")

            self.console.print(rel_table)

    def process(self, repair: bool = True) -> Dict[str, Any]:
        """
        Execute the full graph consistency validation and repair pipeline.

        Args:
            repair: Whether to execute repair operations

        Returns:
            Dict: Processing result statistics
        """
        try:
            # Display basic graph statistics
            self.display_graph_stats()

            # Validate graph consistency
            validation_result = self.validate_graph()

            # If repair is requested and issues were found, run repairs
            if repair and validation_result["validation_stats"]["total_issues"] > 0:
                repair_result = self.repair_graph()
                return {
                    "validation_result": validation_result,
                    "repair_result": repair_result,
                    "total_time": self.validation_time + self.repair_time
                }

            return {
                "validation_result": validation_result,
                "total_time": self.validation_time
            }

        except Exception as e:
            self.console.print(f"[red]Error during graph consistency validation: {e}[/red]")
            raise
