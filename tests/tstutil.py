import concurrent.futures
import contextlib
import typing
import random
import logging


logger = logging.getLogger("ktoolbox.tstutil")


@contextlib.contextmanager
def maybe_thread_pool_executor() -> typing.Generator[
    typing.Optional[concurrent.futures.ThreadPoolExecutor],
    None,
    None,
]:
    if random.choice([False, True]):
        yield None
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            yield executor
        finally:
            executor.shutdown(wait=True)
