import os
import time


def atomic_replace(src, dst, retries=8, delay=0.05):
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
