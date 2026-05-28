import time
import os
import psutil
from typing import Dict, Any, List, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from graphrag_agent.models.get_models import get_extraction_llm, get_embeddings_model
from graphrag_agent.config.prompts import (
    system_template_build_graph,
    human_template_build_graph
)
from graphrag_agent.config.settings import (
    entity_types,
    relationship_types,
    theme,
    FILES_DIR,
    CHUNK_SIZE,
    OVERLAP,
    MAX_WORKERS, BATCH_SIZE,
)
from graphrag_agent.config.neo4jdb import get_db_manager
from graphrag_agent.graph import GraphStructureBuilder
from graphrag_agent.graph import EntityRelationExtractor
from graphrag_agent.graph import GraphWriter

import shutup
shutup.please()


class KnowledgeGraphBuilder:
    """
    Knowledge graph builder that handles the base graph build flow.

    Main responsibilities:
    1. File reading and parsing
    2. Text chunking
    3. Entity and relationship extraction
    4. Base graph structure construction
    5. Database writes
    """
    
    def __init__(self):
        """Initialize the knowledge graph builder."""
        # Initialize console UI
        self.console = Console()
        self.processed_documents = []
        
        # Set up timers
        self.start_time = None
        self.end_time = None
        
        # Stage performance stats
        self.performance_stats = {
            "Initialization": 0,
            "File processing": 0,  # Includes reading and chunking
            "Graph structure build": 0,
            "Entity extraction": 0,
            "Database write": 0
        }
        
        # Initialize components
        self._initialize_components()

    def _create_progress(self):
        """Create a progress display."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )

    def _initialize_components(self):
        """Initialize all required components."""
        init_start = time.time()
        
        with self._create_progress() as progress:
            task = progress.add_task("[cyan]Initializing components...", total=4)
            
            # Initialize models
            self.llm = get_extraction_llm()
            self.embeddings = get_embeddings_model()
            progress.advance(task)
            
            # Initialize graph database connection
            db_manager = get_db_manager()
            self.graph = db_manager.graph
            progress.advance(task)
            
            self.document_processor = None
            self.sec_processor = None
            progress.advance(task)
            
            self.struct_builder = GraphStructureBuilder(batch_size=BATCH_SIZE)
            self.entity_extractor = EntityRelationExtractor(
                self.llm,
                system_template_build_graph,
                human_template_build_graph,
                entity_types,
                relationship_types,
                max_workers=MAX_WORKERS,
                batch_size=5  # Keep LLM batch size small for quality
            )
            
            # Log parameters in use
            self.console.print(f"[blue]Parallel worker threads: {MAX_WORKERS}[/blue]")
            self.console.print(f"[blue]Database batch size: {BATCH_SIZE}[/blue]")
            
            progress.advance(task)
        
        self.performance_stats["Initialization"] = time.time() - init_start

    def _display_stage_header(self, title: str):
        """Display a stage header."""
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")

    def _display_results_table(self, title: str, data: Dict[str, Any]):
        """Display a results table."""
        table = Table(title=title, show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        for key, value in data.items():
            table.add_row(key, str(value))
        
        self.console.print(table)
        
    def _format_time(self, seconds: float) -> str:
        """Format time as hours:minutes:seconds.milliseconds."""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int((seconds % 1) * 1000):03d}"

    def build_base_graph(
        self,
        use_sec_pipeline: bool | None = None,
        files_dir: str | None = None,
    ) -> List:
        """
        Build the base knowledge graph.

        Args:
            use_sec_pipeline: Force SEC or generic processing. None = read from env.
            files_dir: Directory to read source files from. None = use settings default.

        Returns:
            List: Processed file contents including file name, raw text, chunks, and results
        """
        self._display_stage_header("Build base knowledge graph")

        if use_sec_pipeline is None:
            use_sec_pipeline = os.getenv("SEC_PIPELINE_ENABLED", "true").lower() in ("1", "true", "yes")

        try:
            from graphrag_agent.pipelines.ingestion.document_processor import DocumentProcessor
            source_dir = files_dir or str(FILES_DIR)
            self.document_processor = DocumentProcessor(source_dir, CHUNK_SIZE, OVERLAP)

            # 1. Process files (read + chunk)
            process_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]Processing files...", total=1)

                if use_sec_pipeline:
                    from graphrag_agent.pipelines.sec.sec_filing_processor import SecFilingProcessor
                    sec_source = files_dir or os.getenv("SEC_FILES_DIR", str(FILES_DIR))
                    self.sec_processor = SecFilingProcessor(sec_source, CHUNK_SIZE, OVERLAP)
                    self.processed_documents = self.sec_processor.process_directory()
                else:
                    self.processed_documents = self.document_processor.process_directory()
                progress.update(task, completed=1)
                
                # Display file info
                table = Table(title="File info")
                table.add_column("File name")
                table.add_column("Type", style="cyan")
                table.add_column("Content length", justify="right")
                table.add_column("Chunk count", justify="right")
                
                for doc in self.processed_documents:
                    file_type = self.document_processor.get_extension_type(doc["extension"])
                    chunks_count = doc.get("chunk_count", 0)
                    table.add_row(
                        doc["filename"], 
                        file_type, 
                        str(doc["content_length"]),
                        str(chunks_count)
                    )
                self.console.print(table)
            
            self.performance_stats["File processing"] = time.time() - process_start
            
            # Display chunk stats
            total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_documents)
            total_length = sum(doc["content_length"] for doc in self.processed_documents)
            avg_chunk_size = sum(sum(doc.get("chunk_lengths", [0])) for doc in self.processed_documents) / total_chunks if total_chunks else 0
            
            self.console.print(f"[blue]Processed {len(self.processed_documents)} files, total {total_length} characters[/blue]")
            self.console.print(f"[blue]Generated {total_chunks} chunks, average {avg_chunk_size:.1f} characters per chunk[/blue]")
            
            # 3. Build graph structure
            struct_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]Building graph structure...", total=3)
                
                # Clear and create Document nodes
                self.struct_builder.clear_database()
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:  # Only process successfully chunked documents
                        self.struct_builder.create_document(
                            type="local",
                            uri=str(FILES_DIR),
                            file_name=doc["filename"],
                            domain=theme
                        )
                progress.advance(task)
                
                # Create Chunk nodes and relations; optimize with parallel processing for large files
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:  # Only process successfully chunked documents
                        # Choose a method based on chunk count
                        chunks = doc["chunks"]
                        if doc.get("chunk_count", 0) > 100:
                            # Use parallel processing for large files
                            result = self.struct_builder.parallel_process_chunks(
                                doc["filename"],
                                chunks,
                                max_workers=os.cpu_count() or 4
                            )
                        else:
                            # Use standard batch processing for small files
                            result = self.struct_builder.create_relation_between_chunks(
                                doc["filename"],
                                chunks
                            )
                        doc["graph_result"] = result
                progress.advance(task)
                progress.advance(task)
            
            self.performance_stats["Graph structure build"] = time.time() - struct_start
            
            # 4. Extract entities and relationships
            extract_start = time.time()
            with self._create_progress() as progress:
                total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_documents)
                task = progress.add_task("[cyan]Extracting entities and relationships...", total=total_chunks)
                
                def progress_callback(chunk_index):
                    progress.advance(task)
                
                # Prepare data in the required format
                file_contents_format = []
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:
                        file_contents_format.append([
                            doc["filename"], 
                            doc["content"], 
                            doc["chunks"]
                        ])
                
                # Choose a processing method based on dataset size
                if total_chunks > 100:
                    # Use batch processing for large datasets
                    processed_file_contents = self.entity_extractor.process_chunks_batch(
                        file_contents_format,
                        progress_callback
                    )
                else:
                    # Use standard parallel processing for small datasets
                    processed_file_contents = self.entity_extractor.process_chunks(
                        file_contents_format,
                        progress_callback
                    )
                
                # Merge results back into document data
                file_content_map = {}
                for processed_file in processed_file_contents:
                    if len(processed_file) >= 4:  # Ensure sufficient elements
                        filename = processed_file[0]
                        entity_data = processed_file[3]
                        file_content_map[filename] = entity_data
                
                # Use the map to return results to the original documents
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:
                        filename = doc["filename"]
                        if filename in file_content_map:
                            doc["entity_data"] = file_content_map[filename]
                        else:
                            self.console.print(f"[yellow]Warning: entity extraction result not found for file {filename}[/yellow]")
            
            self.performance_stats["Entity extraction"] = time.time() - extract_start
            
            # Log cache stats
            cache_hits = getattr(self.entity_extractor, 'cache_hits', 0)
            cache_misses = getattr(self.entity_extractor, 'cache_misses', 0)
            total_requests = cache_hits + cache_misses
            cache_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            self.console.print(f"[blue]LLM cache hit rate: {cache_rate:.1f}% ({cache_hits}/{total_requests})[/blue]")
            
            # 5. Write to database
            write_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]Writing to database...", total=1)
                
                # Convert processed data to GraphWriter format
                graph_writer_data = []
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"] and "entity_data" in doc:
                        # Get graph build results (created chunk nodes)
                        graph_result = doc.get("graph_result", [])
                        entity_data = doc.get("entity_data", [])
                        
                        # Ensure graph_result and entity_data exist and are aligned
                        if not graph_result:
                            self.console.print(f"[yellow]Warning: graph structure result missing for file {doc['filename']}[/yellow]")
                            continue
                            
                        if not entity_data or not isinstance(entity_data, list):
                            self.console.print(f"[yellow]Warning: entity data missing or invalid for file {doc['filename']}[/yellow]")
                            continue
                            
                        # Adjust data format to match GraphWriter expectations
                        graph_writer_data.append([
                            doc["filename"],
                            doc["content"],
                            doc["chunks"],
                            graph_result,  # Expected chunks_with_hash data
                            entity_data,    # Expected entity extraction results
                        ])
                
                # Use optimized GraphWriter
                graph_writer = GraphWriter(
                    self.graph, 
                    batch_size=50,
                    max_workers=os.cpu_count() or 4
                )
                graph_writer.process_and_write_graph_documents(graph_writer_data)

                # Write structured SEC data (numeric facts + tables + sections + metadata) for each document
                for doc in self.processed_documents:
                    if doc.get("numeric_facts"):
                        graph_writer.write_numeric_facts(doc["filename"], doc["numeric_facts"])
                    if doc.get("tables"):
                        graph_writer.write_tables(doc["filename"], doc["tables"])
                    if doc.get("sections"):
                        graph_writer.write_sections(doc["filename"], doc["sections"])
                    sec_meta_keys = ("form_type", "file_path", "extension", "cik", "company_name", "filing_date")
                    if any(doc.get(k) for k in sec_meta_keys):
                        graph_writer.write_document_metadata(
                            doc["filename"],
                            {k: doc.get(k, "") for k in sec_meta_keys},
                        )

                progress.update(task, completed=1)
            
            self.performance_stats["Database write"] = time.time() - write_start
            
            self.console.print("[green]Base knowledge graph build completed[/green]")
            
            # Display performance stats
            performance_table = Table(title="Performance stats")
            performance_table.add_column("Stage", style="cyan")
            performance_table.add_column("Time (s)", justify="right")
            performance_table.add_column("Share (%)", justify="right")
            
            total_time = sum(self.performance_stats.values())
            for stage, elapsed in self.performance_stats.items():
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                performance_table.add_row(stage, f"{elapsed:.2f}", f"{percentage:.1f}")
            
            performance_table.add_row("Total", f"{total_time:.2f}", "100.0", style="bold")
            self.console.print(performance_table)
            
            # Return processed document list
            file_contents_compat = []
            for doc in self.processed_documents:
                if "chunks" in doc and doc["chunks"]:
                    content_list = [
                        doc["filename"],
                        doc["content"],
                        doc["chunks"]
                    ]
                    if "entity_data" in doc:
                        content_list.append(doc["entity_data"])
                    file_contents_compat.append(content_list)
            
            return file_contents_compat
            
        except Exception as e:
            self.console.print(f"[red]Base graph build failed: {str(e)}[/red]")
            raise

    def build_from_documents(self, documents: List[dict]) -> List:
        """
        Build the knowledge graph from pre-processed documents (e.g. from SEC microservice).
        Skips file reading — runs graph structure build, entity extraction, and DB write only.

        Args:
            documents: List of dicts in the same format as process_directory() output.

        Returns:
            List of processed file contents (compatible with build_base_graph return format).
        """
        self._display_stage_header("Build knowledge graph from pre-processed documents")
        self.processed_documents = documents

        try:
            # 1. Build graph structure
            struct_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]Building graph structure...", total=3)

                self.struct_builder.clear_database()
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:
                        self.struct_builder.create_document(
                            type="local",
                            uri=str(FILES_DIR),
                            file_name=doc["filename"],
                            domain=theme
                        )
                progress.advance(task)

                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:
                        chunks = doc["chunks"]
                        if doc.get("chunk_count", 0) > 100:
                            result = self.struct_builder.parallel_process_chunks(
                                doc["filename"], chunks, max_workers=os.cpu_count() or 4
                            )
                        else:
                            result = self.struct_builder.create_relation_between_chunks(
                                doc["filename"], chunks
                            )
                        doc["graph_result"] = result
                progress.advance(task)
                progress.advance(task)

            self.performance_stats["Graph structure build"] = time.time() - struct_start

            # 2. Extract entities and relationships
            extract_start = time.time()
            with self._create_progress() as progress:
                total_chunks = sum(doc.get("chunk_count", 0) for doc in self.processed_documents)
                task = progress.add_task("[cyan]Extracting entities and relationships...", total=total_chunks)

                def progress_callback(chunk_index):
                    progress.advance(task)

                file_contents_format = []
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:
                        file_contents_format.append([doc["filename"], doc["content"], doc["chunks"]])

                if total_chunks > 100:
                    processed_file_contents = self.entity_extractor.process_chunks_batch(
                        file_contents_format, progress_callback
                    )
                else:
                    processed_file_contents = self.entity_extractor.process_chunks(
                        file_contents_format, progress_callback
                    )

                file_content_map = {}
                for processed_file in processed_file_contents:
                    if len(processed_file) >= 4:
                        file_content_map[processed_file[0]] = processed_file[3]

                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"]:
                        doc["entity_data"] = file_content_map.get(doc["filename"])

            self.performance_stats["Entity extraction"] = time.time() - extract_start

            # 3. Write to database
            write_start = time.time()
            with self._create_progress() as progress:
                task = progress.add_task("[cyan]Writing to database...", total=1)

                graph_writer_data = []
                for doc in self.processed_documents:
                    if "chunks" in doc and doc["chunks"] and "entity_data" in doc:
                        graph_result = doc.get("graph_result", [])
                        entity_data = doc.get("entity_data", [])
                        if not graph_result or not entity_data or not isinstance(entity_data, list):
                            continue
                        graph_writer_data.append([
                            doc["filename"], doc["content"], doc["chunks"],
                            graph_result, entity_data,
                        ])

                graph_writer = GraphWriter(self.graph, batch_size=50, max_workers=os.cpu_count() or 4)
                graph_writer.process_and_write_graph_documents(graph_writer_data)

                for doc in self.processed_documents:
                    if doc.get("numeric_facts"):
                        graph_writer.write_numeric_facts(doc["filename"], doc["numeric_facts"])
                    if doc.get("tables"):
                        graph_writer.write_tables(doc["filename"], doc["tables"])
                    if doc.get("sections"):
                        graph_writer.write_sections(doc["filename"], doc["sections"])
                    sec_meta_keys = ("form_type", "file_path", "extension", "cik", "company_name", "filing_date")
                    if any(doc.get(k) for k in sec_meta_keys):
                        graph_writer.write_document_metadata(
                            doc["filename"], {k: doc.get(k, "") for k in sec_meta_keys}
                        )

                progress.update(task, completed=1)

            self.performance_stats["Database write"] = time.time() - write_start
            self.console.print("[green]Knowledge graph build from documents completed[/green]")

            file_contents_compat = []
            for doc in self.processed_documents:
                if "chunks" in doc and doc["chunks"]:
                    content_list = [doc["filename"], doc["content"], doc["chunks"]]
                    if "entity_data" in doc:
                        content_list.append(doc["entity_data"])
                    file_contents_compat.append(content_list)

            return file_contents_compat

        except Exception as e:
            self.console.print(f"[red]Graph build from documents failed: {str(e)}[/red]")
            raise

    def process(self):
        """Run the knowledge graph build flow."""
        try:
            # Record start time
            self.start_time = time.time()
            
            # Display system resource info
            cpu_count = os.cpu_count() or "Unknown"
            memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            
            system_info = f"System info: CPU cores {cpu_count}, memory {memory_gb:.1f}GB"
            self.console.print(f"[blue]{system_info}[/blue]")
            
            # Show start panel
            start_text = Text("Starting knowledge graph build flow", style="bold cyan")
            self.console.print(Panel(start_text, border_style="cyan"))
            
            # Build base graph
            result = self.build_base_graph()
            
            # Record end time
            self.end_time = time.time()
            elapsed_time = self.end_time - self.start_time
            
            # Show completion panel
            success_text = Text("Knowledge graph build flow completed", style="bold green")
            self.console.print(Panel(success_text, border_style="green"))
            
            # Show total elapsed time
            self.console.print(f"[bold green]Total time: {self._format_time(elapsed_time)}[/bold green]")
            
            return result
            
        except Exception as e:
            # Record end time (even on error)
            self.end_time = time.time()
            if self.start_time is not None:
                elapsed_time = self.end_time - self.start_time
                self.console.print(f"[bold yellow]Elapsed before interruption: {self._format_time(elapsed_time)}[/bold yellow]")
                
            error_text = Text(f"Error during build: {str(e)}", style="bold red")
            self.console.print(Panel(error_text, border_style="red"))
            raise

if __name__ == "__main__":
    try:
        builder = KnowledgeGraphBuilder()
        builder.process()
    except Exception as e:
        console = Console()
        console.print(f"[red]Error occurred during execution: {str(e)}[/red]")