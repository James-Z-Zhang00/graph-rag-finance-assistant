import time
import hashlib
from typing import Callable, Dict, List, Any

def timer(func):
    """
    Timing decorator for measuring function execution time.

    Args:
        func: The function to measure

    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Function {func.__name__} execution time: {elapsed:.2f}s")
        return result
    return wrapper

def generate_hash(text: str) -> str:
    """
    Generate a hash value for the given text.

    Args:
        text: Input text

    Returns:
        str: Hash string
    """
    return hashlib.sha1(text.encode()).hexdigest()

def batch_process(items: List[Any],
                 process_func: Callable,
                 batch_size: int = 100,
                 show_progress: bool = True) -> List[Any]:
    """
    Process items in batches.

    Args:
        items: List of items to process
        process_func: Function that processes a single batch
        batch_size: Batch size
        show_progress: Whether to display progress

    Returns:
        List[Any]: All processing results
    """
    if not items:
        return []

    results = []
    total = len(items)
    batches = (total + batch_size - 1) // batch_size

    if show_progress:
        print(f"Starting batch processing: {total} items in {batches} batches")

    for i in range(0, total, batch_size):
        batch = items[i:i+batch_size]
        batch_results = process_func(batch)

        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)

        if show_progress:
            progress = (i + len(batch)) / total * 100
            print(f"Progress: {progress:.1f}% ({i + len(batch)}/{total})")

    return results

def retry(times: int = 3, exceptions: tuple = (Exception,), delay: float = 1.0):
    """
    Retry decorator.

    Args:
        times: Maximum number of retries
        exceptions: Exception types to catch
        delay: Delay in seconds between retries

    Returns:
        Wrapped function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= times:
                        raise
                    print(f"Function {func.__name__} failed: {e}, retrying ({attempt}/{times})")
                    time.sleep(delay)
        return wrapper
    return decorator

def get_performance_stats(total_time: float,
                         time_records: Dict[str, float]) -> Dict[str, str]:
    """
    Generate a performance statistics summary.

    Args:
        total_time: Total elapsed time
        time_records: Elapsed time records per stage

    Returns:
        Dict[str, str]: Performance statistics summary
    """
    stats = {"Total time": f"{total_time:.2f}s"}

    for name, t in time_records.items():
        percentage = (t/total_time*100) if total_time > 0 else 0
        stats[name] = f"{t:.2f}s ({percentage:.1f}%)"

    return stats

def print_performance_stats(stats: Dict[str, str], title: str = "Performance Statistics Summary") -> None:
    """
    Print the performance statistics summary.

    Args:
        stats: Performance statistics summary
        title: Title
    """
    print(f"\n{title}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
