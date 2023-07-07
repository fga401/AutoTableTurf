import functools
import random
import time


def retry_with_backoff(retries=5, backoff_in_seconds=0.1, max_second=10, skipped_exception=None):
    def rwb(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if skipped_exception is not None and isinstance(e, skipped_exception):
                        raise
                    if x == retries:
                        raise

                    else:
                        sleep = (backoff_in_seconds * 2 ** x + random.uniform(0, 1))
                        sleep = min(max_second, sleep)
                    time.sleep(sleep)
                    x += 1

        return wrapper

    return rwb