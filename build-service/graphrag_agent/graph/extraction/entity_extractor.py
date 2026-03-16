import time
import os
import pickle
import concurrent.futures
from typing import List, Tuple, Optional
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from graphrag_agent.graph.core import retry, generate_hash
from graphrag_agent.config.settings import MAX_WORKERS as DEFAULT_MAX_WORKERS, BATCH_SIZE as DEFAULT_BATCH_SIZE

class EntityRelationExtractor:
    """
    Entity-relationship extractor responsible for extracting entities and relationships from text.
    Uses an LLM to analyze text chunks and generate structured entity and relationship data.
    """

    def __init__(self, llm, system_template, human_template,
             entity_types: List[str], relationship_types: List[str],
             cache_dir="./cache/graph", max_workers=4, batch_size=5):
        """
        Initialize the entity-relationship extractor.

        Args:
            llm: Language model
            system_template: System prompt template
            human_template: Human prompt template
            entity_types: List of entity types
            relationship_types: List of relationship types
            cache_dir: Cache directory
            max_workers: Number of parallel worker threads
            batch_size: Batch processing size
        """
        self.llm = llm
        self.entity_types = entity_types
        self.relationship_types = relationship_types
        self.chat_history = []

        # Delimiters
        self.tuple_delimiter = " : "
        self.record_delimiter = "\n"
        self.completion_delimiter = "\n\n"

        # Create prompt templates
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            MessagesPlaceholder("chat_history"),
            human_message_prompt
        ])

        # Build processing chain
        self.chain = self.chat_prompt | self.llm

        # Cache settings
        self.cache_dir = cache_dir
        self.enable_cache = True

        # Fingerprint of the system prompt so cache is invalidated when the
        # prompt changes (e.g. after adding new extraction examples).
        self._prompt_hash = generate_hash(system_template)[:8]

        # Ensure cache directory exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Parallel processing configuration
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_cache_key(self, text: str) -> str:
        """
        Generate a cache key for the given text.
        Includes a short fingerprint of the system prompt so that changing
        the extraction prompt (e.g. adding examples) invalidates old cache entries.

        Args:
            text: Input text

        Returns:
            str: Cache key
        """
        return f"{self._prompt_hash}_{generate_hash(text)}"

    def _cache_path(self, cache_key: str) -> str:
        """
        Get the cache file path for a given cache key.

        Args:
            cache_key: Cache key

        Returns:
            str: Cache file path
        """
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _save_to_cache(self, cache_key: str, result: str) -> None:
        """
        Save a result to cache.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if not self.enable_cache:
            return

        cache_path = self._cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Cache save error: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """
        Load a result from cache.

        Args:
            cache_key: Cache key

        Returns:
            Optional[str]: Cached result, or None if not found
        """
        if not self.enable_cache:
            return None

        cache_path = self._cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                    self.cache_hits += 1
                    return result
            except Exception as e:
                print(f"Cache load error: {e}")

        self.cache_misses += 1
        return None

    def process_chunks(self, file_contents: List[Tuple], progress_callback=None) -> List[Tuple]:
        """
        Process all chunks from all files in parallel.

        Args:
            file_contents: List of file contents
            progress_callback: Progress callback function

        Returns:
            List[Tuple]: Processing results
        """
        t0 = time.time()
        chunk_index = 0
        total_chunks = sum(len(file_content[2]) for file_content in file_contents)

        # Multi-threaded dispatch strategy
        for i, file_content in enumerate(file_contents):
            chunks = file_content[2]

            # Pre-check cache hit rate
            cache_keys = [self._generate_cache_key(''.join(chunk)) for chunk in chunks]
            cached_results = {key: self._load_from_cache(key) for key in cache_keys}
            non_cached_indices = [idx for idx, key in enumerate(cache_keys) if cached_results[key] is None]

            if len(non_cached_indices) > 0:
                # Only create tasks for uncached chunks
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_chunk = {
                        executor.submit(self._process_single_chunk, ''.join(chunks[idx])): idx
                        for idx in non_cached_indices
                    }

                    # Process completed tasks
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        try:
                            result = future.result()
                            cached_results[cache_keys[chunk_idx]] = result

                            if progress_callback:
                                progress_callback(chunk_index)
                            chunk_index += 1

                        except Exception as exc:
                            print(f'Chunk {chunk_idx} processing exception: {exc}')
                            # Retry logic
                            retry_count = 0
                            while retry_count < 3:
                                try:
                                    print(f'Retrying chunk {chunk_idx}, attempt {retry_count+1}')
                                    result = self._process_single_chunk(''.join(chunks[chunk_idx]))
                                    cached_results[cache_keys[chunk_idx]] = result
                                    break
                                except Exception as retry_exc:
                                    print(f'Retry failed: {retry_exc}')
                                    retry_count += 1
                                    time.sleep(1)

                            if cached_results[cache_keys[chunk_idx]] is None:
                                cached_results[cache_keys[chunk_idx]] = ""

            # Arrange results in original order
            ordered_results = [cached_results[key] for key in cache_keys]
            file_content.append(ordered_results)

            # Output cache statistics
            cache_ratio = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
            print(f"File {i+1}/{len(file_contents)} processed, cache hit rate: {cache_ratio:.1f}%")

        process_time = time.time() - t0
        print(f"All chunks processed, total time: {process_time:.2f}s, average per chunk: {process_time/total_chunks:.2f}s")
        return file_contents

    def process_chunks_batch(self, file_contents: List[Tuple], progress_callback=None) -> List[Tuple]:
        """
        Process chunks in batches to reduce LLM call count.

        Args:
            file_contents: List of file contents
            progress_callback: Progress callback function

        Returns:
            List[Tuple]: Processing results
        """
        for file_content in file_contents:
            chunks = file_content[2]
            results = []

            # Intelligent dynamic batch sizing
            chunk_lengths = [len(''.join(chunk)) for chunk in chunks]
            avg_chunk_size = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0

            # Dynamically adjust batch size based on average chunk size
            dynamic_batch_size = max(1, min(self.batch_size, int(10000 / (avg_chunk_size + 1))))

            # Process in batches
            for i in range(0, len(chunks), dynamic_batch_size):
                batch_chunks = chunks[i:i+dynamic_batch_size]

                # Cache check
                batch_keys = [self._generate_cache_key(''.join(chunk)) for chunk in batch_chunks]
                cached_batch_results = [self._load_from_cache(key) for key in batch_keys]

                # If all results are cached, skip LLM call
                if None not in cached_batch_results:
                    results.extend(cached_batch_results)
                    if progress_callback:
                        for j in range(len(batch_chunks)):
                            progress_callback(i + j)
                    continue

                # Prepare batch input
                batch_inputs = []
                for chunk in batch_chunks:
                    batch_inputs.append(''.join(chunk))

                # Combine multiple text chunks with separator
                batch_text = f"\n{'-'*50}\n".join(batch_inputs)

                try:
                    batch_response = self.chain.invoke({
                        "chat_history": self.chat_history,
                        "entity_types": self.entity_types,
                        "relationship_types": self.relationship_types,
                        "tuple_delimiter": self.tuple_delimiter,
                        "record_delimiter": self.record_delimiter,
                        "completion_delimiter": self.completion_delimiter,
                        "input_text": batch_text
                    })

                    # Parse batch response
                    batch_results = self._parse_batch_response(batch_response.content)

                    # Handle result count mismatch
                    if len(batch_results) != len(batch_chunks):
                        # If batch response cannot be parsed correctly, process each chunk individually
                        batch_results = []
                        for idx, chunk in enumerate(batch_chunks):
                            cached_result = cached_batch_results[idx]
                            if cached_result is not None:
                                batch_results.append(cached_result)
                            else:
                                individual_result = self._process_single_chunk(''.join(chunk))
                                batch_results.append(individual_result)
                    else:
                        # Cache batch results
                        for idx, result in enumerate(batch_results):
                            if cached_batch_results[idx] is None:  # Only cache misses
                                self._save_to_cache(batch_keys[idx], result)

                    results.extend(batch_results)
                except Exception as e:
                    print(f"Batch processing error, falling back to individual processing: {e}")
                    for idx, chunk in enumerate(batch_chunks):
                        try:
                            individual_result = self._process_single_chunk(''.join(chunk))
                            results.append(individual_result)
                        except Exception as e2:
                            print(f"Single chunk processing failed: {e2}")
                            results.append("")

                if progress_callback:
                    for j in range(len(batch_chunks)):
                        progress_callback(i + j)

            file_content.append(results)

        return file_contents

    def _parse_batch_response(self, batch_content: str) -> List[str]:
        """
        Parse a batch response by splitting it into individual results.

        Args:
            batch_content: Batch response content

        Returns:
            List[str]: List of split results
        """
        parts = batch_content.split(f"\n{'-'*50}\n")
        return [part.strip() for part in parts]

    @retry(times=3, exceptions=(Exception,), delay=1.0)
    def _process_single_chunk(self, input_text: str) -> str:
        """
        Process a single text chunk (with caching).

        Args:
            input_text: Input text

        Returns:
            str: Processing result
        """
        # Generate cache key
        cache_key = self._generate_cache_key(input_text)

        # Try loading from cache
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            return cached_result

        # Cache miss — call LLM
        response = self.chain.invoke({
            "chat_history": self.chat_history,
            "entity_types": self.entity_types,
            "relationship_types": self.relationship_types,
            "tuple_delimiter": self.tuple_delimiter,
            "record_delimiter": self.record_delimiter,
            "completion_delimiter": self.completion_delimiter,
            "input_text": input_text
        })

        result = response.content

        # Save result to cache
        self._save_to_cache(cache_key, result)

        return result

    def stream_process_large_files(self, file_path: str, chunk_size: int = 5000,
                                   structure_builder=None, graph_writer=None) -> None:
        """
        Process large files in streaming fashion to avoid loading all content at once.

        Args:
            file_path: File path
            chunk_size: Chunk size in characters
            structure_builder: Graph structure builder
            graph_writer: Graph writer
        """
        if not structure_builder or not graph_writer:
            print("structure_builder and graph_writer are required for streaming processing")
            return

        def text_chunks_iterator(file_path, chunk_size):
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk = []
                chars_count = 0
                for line in f:
                    chunk.append(line)
                    chars_count += len(line)
                    if chars_count >= chunk_size:
                        yield chunk
                        chunk = []
                        chars_count = 0
                if chunk:  # Don't forget the final partial chunk
                    yield chunk

        # File metadata
        file_name = os.path.basename(file_path)
        file_type = os.path.splitext(file_name)[1]

        # Create document node
        structure_builder.create_document(
            type=file_type,
            uri=file_path,
            file_name=file_name,
            domain="document"
        )

        # Stream-process the file
        chunks = []
        for chunk in text_chunks_iterator(file_path, chunk_size):
            chunks.append(chunk)

        # Create relationships between chunks
        chunks_with_hash = structure_builder.create_relation_between_chunks(
            file_name, chunks
        )

        # Process all chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {}
            for chunk_data in chunks_with_hash:
                chunk_text = chunk_data['chunk_doc'].page_content
                cache_key = self._generate_cache_key(chunk_text)
                cached_result = self._load_from_cache(cache_key)

                if cached_result:
                    # Cache hit — process result directly
                    try:
                        graph_document = graph_writer.convert_to_graph_document(
                            chunk_data['chunk_id'],
                            chunk_data['chunk_doc'].page_content,
                            cached_result
                        )

                        if len(graph_document.nodes) > 0 or len(graph_document.relationships) > 0:
                            graph_writer.graph.add_graph_documents(
                                [graph_document],
                                baseEntityLabel=True,
                                include_source=True
                            )
                    except Exception as e:
                        print(f"Error processing cached result: {e}")
                else:
                    # Cache miss — submit task
                    future = executor.submit(self._process_single_chunk, chunk_text)
                    future_to_chunk[future] = chunk_data

            # Process results and write to graph database
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_data = future_to_chunk[future]
                try:
                    result = future.result()

                    # Write chunk result to graph database immediately
                    graph_document = graph_writer.convert_to_graph_document(
                        chunk_data['chunk_id'],
                        chunk_data['chunk_doc'].page_content,
                        result
                    )

                    if len(graph_document.nodes) > 0 or len(graph_document.relationships) > 0:
                        graph_writer.graph.add_graph_documents(
                            [graph_document],
                            baseEntityLabel=True,
                            include_source=True
                        )

                except Exception as exc:
                    print(f"Error processing chunk {chunk_data['chunk_id']}: {exc}")
