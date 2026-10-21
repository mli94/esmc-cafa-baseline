import time
import os
import gc
import resource

def measure_memory_and_time(func, *args, **kwargs):
    """
    Measure execution time and peak memory usage of a function
    
    Args:
        func: Function to measure
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        tuple: (result, execution_time_seconds, peak_memory_mb)
    """
    # Force garbage collection before the test
    gc.collect()
    
    # Get initial memory usage
    initial_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    # Measure execution time
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    # Get peak memory usage
    peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - initial_mem
    
    # Convert to MB (system dependent)
    if os.name == 'posix':
        # On Unix, ru_maxrss is in KB
        peak_mem_mb = peak_mem / 1024
    else:
        # On Windows, ru_maxrss is in bytes
        peak_mem_mb = peak_mem / (1024 * 1024)
    
    return result, execution_time, peak_mem_mb