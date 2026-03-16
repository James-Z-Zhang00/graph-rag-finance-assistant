import time
from typing import Dict, Any
from rich.console import Console
from rich.table import Table

from graphrag_agent.graph.processing.entity_disambiguation import EntityDisambiguator
from graphrag_agent.graph.processing.entity_alignment import EntityAligner

class EntityQualityProcessor:
    """
    Entity quality processor: integrates the disambiguation and alignment pipeline.
    Runs after similar entity detection and merging to further improve entity quality.
    """

    def __init__(self):
        self.console = Console()
        self.disambiguator = EntityDisambiguator()
        self.aligner = EntityAligner()

        # Performance statistics
        self.stats = {
            'total_time': 0,
            'disambig_time': 0,
            'align_time': 0
        }

    def process(self) -> Dict[str, Any]:
        """
        Execute the full entity quality improvement pipeline.

        Returns:
            Dict: Processing result statistics
        """
        start_time = time.time()

        self.console.print("[bold cyan]Starting entity quality improvement pipeline[/bold cyan]")

        # 1. Entity disambiguation
        self.console.print("\n[cyan]Phase 1: Entity Disambiguation[/cyan]")
        disambig_start = time.time()

        disambiguated = self.disambiguator.apply_to_graph()

        self.stats['disambig_time'] = time.time() - disambig_start

        self.console.print(f"[green]Disambiguation complete: processed {disambiguated} entities[/green]")

        # Display disambiguation statistics
        self._display_stats_table("Disambiguation Statistics", {
            'Entities with canonical_id set': disambiguated,
            'Time elapsed': f"{self.stats['disambig_time']:.2f}s"
        })

        # 2. Entity alignment
        self.console.print("\n[cyan]Phase 2: Entity Alignment[/cyan]")
        align_start = time.time()

        align_result = self.aligner.align_all()

        self.stats['align_time'] = time.time() - align_start

        self.console.print(f"[green]Alignment complete: merged {align_result['entities_aligned']} entities[/green]")

        # Display alignment statistics
        self._display_stats_table("Alignment Statistics", {
            'Groups processed': align_result['groups_processed'],
            'Conflicts detected': align_result['conflicts_detected'],
            'Entities merged': align_result['entities_aligned'],
            'Time elapsed': f"{self.stats['align_time']:.2f}s"
        })

        # Total
        self.stats['total_time'] = time.time() - start_time

        self.console.print(f"\n[bold green]Entity quality improvement complete, total time: {self.stats['total_time']:.2f}s[/bold green]")

        return {
            'disambiguated': disambiguated,
            'aligned': align_result['entities_aligned'],
            'stats': self.stats
        }

    def _display_stats_table(self, title: str, data: Dict[str, Any]):
        """Display a statistics table."""
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        for key, value in data.items():
            table.add_row(str(key), str(value))

        self.console.print(table)
