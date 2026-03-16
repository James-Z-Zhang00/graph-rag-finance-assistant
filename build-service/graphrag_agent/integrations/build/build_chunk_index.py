import os
import time
import psutil
from typing import Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from graphrag_agent.graph import ChunkIndexManager
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import MAX_WORKERS, CHUNK_BATCH_SIZE

import shutup
shutup.please()

class ChunkIndexBuilder:
    """
    Chunk index builder, responsible for creating vector indexes on Chunk nodes
    after the base graph is built, to support naive RAG queries.

    Key functions:
    1. Creation and management of Chunk node indexes
    2. Vector index performance statistics
    """

    def __init__(self):
        """Initialize the chunk index builder"""
        # Initialize terminal interface
        self.console = Console()

        # Stage performance statistics
        self.performance_stats = {
            "Initialization": 0,
            "Index Creation": 0
        }

        # Timers
        self.start_time = None
        self.end_time = None

        # Initialize components
        self._initialize_components()

    def _create_progress(self):
        """Create progress display"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )

    def _initialize_components(self):
        """Initialize all required components"""
        init_start = time.time()

        with self._create_progress() as progress:
            task = progress.add_task("[cyan]Initializing components...", total=2)

            # Initialize graph database connection
            db_manager = get_db_manager()
            self.graph = db_manager.graph
            progress.advance(task)

            self.index_manager = ChunkIndexManager(
                batch_size=CHUNK_BATCH_SIZE,
                max_workers=MAX_WORKERS
            )

            # Log parameters in use
            self.console.print(f"[blue]Parallel worker threads: {MAX_WORKERS}[/blue]")
            self.console.print(f"[blue]Database batch size: {CHUNK_BATCH_SIZE}[/blue]")

            progress.advance(task)

        self.performance_stats["Initialization"] = time.time() - init_start

    def _display_stage_header(self, title: str):
        """Display the header for a processing stage"""
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")

    def _display_results_table(self, title: str, data: Dict[str, Any]):
        """Display results table"""
        table = Table(title=title, show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        for key, value in data.items():
            table.add_row(key, str(value))

        self.console.print(table)

    def _format_time(self, seconds: float) -> str:
        """Format time as HH:MM:SS.mmm"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int((seconds % 1) * 1000):03d}"

    def build_chunk_index(self):
        """
        Build the chunk index.

        Returns:
            bool: Whether processing succeeded
        """
        self._display_stage_header("Building Chunk Index")

        try:
            # Create Chunk index
            index_start = time.time()
            self.console.print("[cyan]Creating chunk index...[/cyan]")

            # Only compute and store embeddings; do not create a new vector index
            vector_store = self.index_manager.create_chunk_index()

            self.performance_stats["Index Creation"] = time.time() - index_start

            # Display embedding computation performance
            embedding_time = getattr(self.index_manager, 'embedding_time', 0)
            db_time = getattr(self.index_manager, 'db_time', 0)
            index_total = self.performance_stats["Index Creation"]

            if index_total > 0:
                self.console.print(f"[blue]Index creation complete, total time: {index_total:.2f}s[/blue]")
                self.console.print(f"[blue]Breakdown: embedding computation: {embedding_time:.2f}s ({embedding_time/index_total*100:.1f}%), "
                                f"database operations: {db_time:.2f}s ({db_time/index_total*100:.1f}%)[/blue]")

            # Query node count
            try:
                node_count = self.graph.query(
                    """
                    MATCH (c:`__Chunk__`)
                    WHERE c.embedding IS NOT NULL
                    RETURN count(c) as count
                    """
                )

                self._display_results_table(
                    "Index Creation Results",
                    {
                        "Indexed nodes": node_count[0]["count"] if node_count else 0,
                        "Total time": f"{index_total:.2f}s",
                        "Embedding computation": f"{embedding_time:.2f}s ({embedding_time/index_total*100:.1f}%)" if index_total > 0 else "0.00s",
                        "Database operations": f"{db_time:.2f}s ({db_time/index_total*100:.1f}%)" if index_total > 0 else "0.00s"
                    }
                )
            except Exception as e:
                self.console.print(f"[yellow]Error querying index status (ignorable): {e}[/yellow]")

            self.console.print("[green]Chunk index build complete[/green]")

            # Display performance summary
            performance_table = Table(title="Performance Summary")
            performance_table.add_column("Stage", style="cyan")
            performance_table.add_column("Time (s)", justify="right")
            performance_table.add_column("Share (%)", justify="right")

            total_time = sum(self.performance_stats.values())
            for stage, elapsed in self.performance_stats.items():
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                performance_table.add_row(stage, f"{elapsed:.2f}", f"{percentage:.1f}")

            performance_table.add_row("Total", f"{total_time:.2f}", "100.0", style="bold")
            self.console.print(performance_table)

            return True

        except Exception as e:
            self.console.print(f"[red]Chunk index build failed: {str(e)}[/red]")
            raise

    def process(self):
        """Execute the chunk index build pipeline"""
        try:
            # Record start time
            self.start_time = time.time()

            # Display system resource info
            cpu_count = os.cpu_count() or "unknown"
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)

            system_info = f"System info: CPU cores {cpu_count}, memory {memory_gb:.1f}GB"
            self.console.print(f"[blue]{system_info}[/blue]")

            # Display start panel
            start_text = Text("Starting Chunk Index Build Pipeline", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))

            # Build chunk index
            self.build_chunk_index()

            # Record end time
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time

            # Display completion panel
            success_text = Text("Chunk Index Build Pipeline Complete", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))

            # Display total elapsed time
            self.console.print(f"[bold green]Total time: {self._format_time(elapsed_time)}[/bold green]")

            return True

        except Exception as e:
            # Record end time even on error
            self.end_time = time.time()
            if self.start_time is not None:
                elapsed_time = self.end_time - self.start_time
                self.console.print(f"[bold yellow]Time before interruption: {self._format_time(elapsed_time)}[/bold yellow]")

            error_text = Text(f"Error during processing: {str(e)}", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            raise
