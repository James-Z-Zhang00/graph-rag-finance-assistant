import time
import concurrent.futures
from typing import List, Any, Optional

from graphrag_agent.config.settings import MAX_WORKERS as CONFIG_MAX_WORKERS

class BaseIndexer:
    """
    Base indexer class providing common functionality for all indexers.
    Contains batch processing, parallel computation, and performance monitoring logic.
    """

    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        """
        Initialize the base indexer.

        Args:
            batch_size: Batch processing size
            max_workers: Number of parallel worker threads
        """
        # Batch processing and parallelism parameters
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Performance monitoring parameters
        self.embedding_time = 0
        self.db_time = 0

    def _create_indexes(self) -> None:
        """Create necessary indexes to optimize query performance — implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_optimal_batch_size(self, total_items: int) -> int:
        # Use the configured batch size as an upper bound
        optimal_size = min(self.batch_size, max(20, total_items // 10))
        return optimal_size

    def batch_process_with_progress(self,
                                   items: List[Any],
                                   process_func,
                                   batch_size: Optional[int] = None,
                                   desc: str = "Processing") -> None:
        """
        Generic batch processing logic with progress tracking.

        Args:
            items: List of items to process
            process_func: Function to process a single batch
            batch_size: Batch size; uses optimal value if not provided
            desc: Progress description
        """
        if not items:
            print("No items found to process")
            return

        # Calculate batch processing parameters
        item_count = len(items)
        optimal_batch_size = batch_size or self.get_optimal_batch_size(item_count)
        total_batches = (item_count + optimal_batch_size - 1) // optimal_batch_size

        print(f"{desc}: {item_count} items, batch size: {optimal_batch_size}, total batches: {total_batches}")

        # Store processing time per batch
        batch_times = []

        # Batch processing loop
        for batch_index in range(total_batches):
            batch_start = time.time()

            start_idx = batch_index * optimal_batch_size
            end_idx = min(start_idx + optimal_batch_size, item_count)
            batch = items[start_idx:end_idx]

            # Process current batch
            process_func(batch, batch_index)

            # Calculate and display progress
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)

            # Calculate average time and estimated remaining time
            avg_time = sum(batch_times) / len(batch_times)
            remaining_batches = total_batches - (batch_index + 1)
            estimated_remaining = avg_time * remaining_batches

            print(f"Processed batch {batch_index+1}/{total_batches}, "
                  f"batch time: {batch_time:.2f}s, "
                  f"average: {avg_time:.2f}s/batch, "
                  f"estimated remaining: {estimated_remaining:.2f}s")

    def process_in_parallel(self, items: List[Any], process_func) -> List[Any]:
        """
        Process items in parallel.

        Args:
            items: List of items to process
            process_func: Function to process a single item

        Returns:
            List[Any]: List of processing results
        """
        max_workers = min(self.max_workers, CONFIG_MAX_WORKERS)
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_func, item): i
                for i, item in enumerate(items)
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Parallel processing error: {e}")

        return results
