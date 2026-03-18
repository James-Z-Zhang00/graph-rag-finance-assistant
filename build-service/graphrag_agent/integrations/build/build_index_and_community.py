import os
import time
import psutil
from typing import Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from graphrag_agent.config.settings import community_algorithm
from graphrag_agent.graph import EntityIndexManager
from graphrag_agent.graph import GDSConfig, SimilarEntityDetector
from graphrag_agent.graph import EntityMerger
from graphrag_agent.graph.processing import EntityQualityProcessor
from graphrag_agent.community import CommunityDetectorFactory
from graphrag_agent.community import CommunitySummarizerFactory
from graphdatascience import GraphDataScience

from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.config.settings import MAX_WORKERS, ENTITY_BATCH_SIZE, GDS_MEMORY_LIMIT, NEO4J_CONFIG

import shutup
shutup.please()

# Aura Graph Analytics is versionless — patch server_version() to return a
# fixed value instead of calling gds.version which Aura rejects.
# Aura rejects gds.version() with "Aura Graph Analytics is versionless", and also
# doesn't support gds.debug.arrow(), causing GraphDataScience() to fail on init.
# The library's aura_ds=True parameter is supposed to skip both checks, but it doesn't
# work reliably across versions. These patches replicate that behavior directly.
try:
    from graphdatascience.query_runner.neo4j_query_runner import Neo4jQueryRunner as _NQR
    from graphdatascience.server_version.server_version import ServerVersion as _SV
    _NQR.server_version = lambda self: _SV(2, 5, 0)
except Exception:
    pass

# Patch both known import paths for ArrowInfo.create — the location changed between
# library versions. Aura doesn't support gds.debug.arrow() so we always return a
# disabled ArrowInfo, skipping Arrow Flight client construction entirely.
import importlib as _il
for _arrow_info_path in (
    "graphdatascience.query_runner.arrow_info",
    "graphdatascience.arrow_client.arrow_info",
):
    try:
        _mod = _il.import_module(_arrow_info_path)
        _cls = _mod.ArrowInfo

        def _disabled_create(_query_runner, _klass=_cls):
            obj = object.__new__(_klass)
            obj.enabled = False
            obj.running = False
            obj.listenAddress = ""
            obj.versions = []
            return obj

        # create may be a staticmethod or classmethod depending on version
        _cls.create = staticmethod(_disabled_create)
    except Exception:
        pass

class IndexCommunityBuilder:
    """
    Index and community builder, responsible for post-processing after the base graph is built.

    Key functions:
    1. Creation and management of entity indexes
    2. Detection and merging of similar entities
    3. Entity disambiguation and alignment
    4. Community detection
    5. Community summary generation
    """

    def __init__(self):
        """Initialize the index and community builder"""
        # Initialize terminal interface
        self.console = Console()

        # Stage performance statistics — must be defined before _initialize_components
        self.performance_stats = {
            "Initialization": 0,
            "Index Creation": 0,
            "Similar Entity Detection": 0,
            "Entity Merging": 0,
            "Entity Quality Enhancement": 0,
            "Community Detection": 0,
            "Community Summarization": 0
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
            task = progress.add_task("[cyan]Initializing components...", total=4)

            # Initialize graph database connection
            self.gds = GraphDataScience(
                NEO4J_CONFIG["uri"],
                auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"]),
                aura_ds=True
            )
            db_manager = get_db_manager()
            self.graph = db_manager.graph
            progress.advance(task)

            self.index_manager = EntityIndexManager(
                batch_size=ENTITY_BATCH_SIZE,
                max_workers=MAX_WORKERS
            )
            self.gds_config = GDSConfig(
                memory_limit=GDS_MEMORY_LIMIT,
                batch_size=ENTITY_BATCH_SIZE
            )
            self.entity_detector = SimilarEntityDetector(self.gds_config)
            self.entity_merger = EntityMerger(
                batch_size=ENTITY_BATCH_SIZE,
                max_workers=MAX_WORKERS
            )
            progress.advance(task)

            # Initialize entity quality processor
            self.quality_processor = EntityQualityProcessor()
            progress.advance(task)

            # Log parameters in use
            self.console.print(f"[blue]Parallel worker threads: {MAX_WORKERS}[/blue]")
            self.console.print(f"[blue]Database batch size: {ENTITY_BATCH_SIZE}[/blue]")
            self.console.print(f"[blue]GDS memory limit: {GDS_MEMORY_LIMIT}GB[/blue]")

            # Initialize results storage
            self.processing_results = {
                'index_process': {}
            }
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

    def build_index_and_communities(self):
        """
        Build indexes and communities.

        Returns:
            bool: Whether processing succeeded
        """
        self._display_stage_header("Building Indexes and Communities")

        try:
            # 1. Create entity index
            index_start = time.time()
            self.console.print("[cyan]Creating entity index...[/cyan]")

            vector_store = self.index_manager.create_entity_index()
            if not vector_store:
                self.console.print("[yellow]Warning: entity index creation may be incomplete[/yellow]")

            self.performance_stats["Index Creation"] = time.time() - index_start

            # Display embedding computation performance
            embedding_time = getattr(self.index_manager, 'embedding_time', 0)
            db_time = getattr(self.index_manager, 'db_time', 0)
            index_total = self.performance_stats["Index Creation"]

            self.console.print(f"[blue]Index creation complete, total time: {index_total:.2f}s[/blue]")
            self.console.print(f"[blue]Breakdown: embedding computation: {embedding_time:.2f}s ({embedding_time/index_total*100:.1f}%), "
                              f"database operations: {db_time:.2f}s ({db_time/index_total*100:.1f}%)[/blue]")

            # 2. Detect and merge similar entities
            similar_start = time.time()
            self.console.print("[cyan]Detecting similar entities...[/cyan]")

            duplicates = self.entity_detector.process_entities()

            self.performance_stats["Similar Entity Detection"] = time.time() - similar_start

            # Display similar entity detection performance
            projection_time = getattr(self.entity_detector, 'projection_time', 0)
            knn_time = getattr(self.entity_detector, 'knn_time', 0)
            wcc_time = getattr(self.entity_detector, 'wcc_time', 0)
            query_time = getattr(self.entity_detector, 'query_time', 0)
            similar_total = self.performance_stats["Similar Entity Detection"]

            self.console.print(f"[blue]Similar entity detection complete, found {len(duplicates)} candidate groups, total time: {similar_total:.2f}s[/blue]")
            self.console.print(f"[blue]Breakdown: projection: {projection_time:.2f}s ({projection_time/similar_total*100:.1f}%), "
                              f"KNN: {knn_time:.2f}s ({knn_time/similar_total*100:.1f}%), "
                              f"WCC: {wcc_time:.2f}s ({wcc_time/similar_total*100:.1f}%), "
                              f"query: {query_time:.2f}s ({query_time/similar_total*100:.1f}%)[/blue]")

            # 3. Merge similar entities
            merge_start = time.time()
            self.console.print("[cyan]Merging similar entities...[/cyan]")

            merged_count = self.entity_merger.process_duplicates(duplicates)

            self.performance_stats["Entity Merging"] = time.time() - merge_start

            # Display entity merging performance
            llm_time = getattr(self.entity_merger, 'llm_time', 0)
            parse_time = getattr(self.entity_merger, 'parse_time', 0)
            db_time = getattr(self.entity_merger, 'db_time', 0)
            merge_total = self.performance_stats["Entity Merging"]

            self._display_results_table(
                "Entity Merging Results",
                {
                    "Merged entity groups": merged_count,
                    "Total time": f"{merge_total:.2f}s",
                    "LLM processing": f"{llm_time:.2f}s ({llm_time/merge_total*100:.1f}%)" if merge_total > 0 else "0.00s (0.0%)",
                    "Result parsing": f"{parse_time:.2f}s ({parse_time/merge_total*100:.1f}%)" if merge_total > 0 else "0.00s (0.0%)",
                    "Database operations": f"{db_time:.2f}s ({db_time/merge_total*100:.1f}%)" if merge_total > 0 else "0.00s (0.0%)"
                }
            )

            # 4. Entity quality enhancement (disambiguation and alignment)
            quality_start = time.time()
            self.console.print("[cyan]Running entity disambiguation and alignment...[/cyan]")

            quality_result = self.quality_processor.process()

            self.performance_stats["Entity Quality Enhancement"] = time.time() - quality_start

            self._display_results_table(
                "Entity Quality Enhancement Results",
                {
                    "Disambiguated entities": quality_result['disambiguated'],
                    "Aligned entities": quality_result['aligned'],
                    "Total time": f"{self.performance_stats['Entity Quality Enhancement']:.2f}s"
                }
            )

            # 5. Community detection
            community_start = time.time()
            self.console.print("[cyan]Running community detection...[/cyan]")

            # Create detector via factory
            detector = CommunityDetectorFactory.create(
                algorithm=community_algorithm,
                gds=self.gds,
                graph=self.graph
            )
            community_results = detector.process()

            self.performance_stats["Community Detection"] = time.time() - community_start

            self.console.print(f"[blue]Community detection complete, time: {self.performance_stats['Community Detection']:.2f}s[/blue]")
            if community_results and community_results.get('status') == 'success':
                community_count = community_results.get('details', {}).get('detection', {}).get('communityCount', 0)
                self.console.print(f"[blue]Detected {community_count} communities[/blue]")

            # 6. Generate community summaries
            summary_start = time.time()
            self.console.print("[cyan]Generating community summaries...[/cyan]")

            # Use summarizer factory
            summarizer = CommunitySummarizerFactory.create_summarizer(
                community_algorithm,
                self.graph
            )
            summaries = summarizer.process_communities()

            self.performance_stats["Community Summarization"] = time.time() - summary_start

            self._display_results_table(
                "Community Summarization Results",
                {
                    "Summaries generated": len(summaries) if summaries else 0,
                    "Time": f"{self.performance_stats['Community Summarization']:.2f}s"
                }
            )

            self.console.print("[green]Index and community build complete[/green]")

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
            self.console.print(f"[red]Index and community build failed: {str(e)}[/red]")
            raise

    def process(self):
        """Execute the index and community build pipeline"""
        try:
            # Record start time
            self.start_time = time.time()

            # Display system resource info
            cpu_count = os.cpu_count() or "unknown"
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)

            system_info = f"System info: CPU cores {cpu_count}, memory {memory_gb:.1f}GB"
            self.console.print(f"[blue]{system_info}[/blue]")

            # Display start panel
            start_text = Text("Starting Index and Community Build Pipeline", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))

            # Build indexes and communities
            self.build_index_and_communities()

            # Record end time
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time

            # Display completion panel
            success_text = Text("Index and Community Build Pipeline Complete", style="bold green")
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

if __name__ == "__main__":
    try:
        builder = IndexCommunityBuilder()
        builder.process()
    except Exception as e:
        console = Console()
        console.print(f"[red]Error during execution: {str(e)}[/red]")
