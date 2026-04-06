# extra import to monitor memory and elapsed time
import time
import os, psutil
import numpy as np
import pandas as pd

class Monitoring:
    """
    Class to monitor elapsed time and memory usage.

    This is a builder design pattern, so you can create an instance of the class and call the methods at each step in your code.

    Usage:
    ```python
    monitor = Monitoring()
    # each step in code
    monitor.elapsed_time('step name')
    monitor.check_memory('step name')
    ```
    """
    def __init__(self):
        self.start_time = time.perf_counter()
    
    def elapsed_time(self, step = ''):
        elapsed = time.perf_counter() - self.start_time
        print(f"{step} seconds:", round(elapsed, 3))
    
    def check_memory(self, step = ''):
        """Gets RSS (Resident Set Size), which is the amount of memory currently held in physical RAM by that process, in bytes.
        """
        p = psutil.Process(os.getpid())
        print(f"{step} RSS GB:", round(p.memory_info().rss / (1024**3), 2))

    def reset_timer(self):
        self.start_time = time.perf_counter()

    def check_variable_memory(self, variable, step = ''):
        if isinstance(variable, np.ndarray):
            print(f"{step} GB:", round(variable.nbytes / (1024**3), 2))
        elif isinstance(variable, pd.DataFrame):
            print(f"{step} GB:", round(variable.memory_usage(deep=True).sum() / (1024**3), 2))
        else:
            print(f"{step} variable type not supported for memory check.")
