import concurrent.futures
import contextlib
import typing
import random
import logging


T = typing.TypeVar("T")


logger = logging.getLogger("ktoolbox.tstutil")


def rnd_select(*args: T) -> T:
    return args[rnd_select_n(len(args))]


def rnd_select_n(n: int) -> int:
    return random.randint(0, n - 1)


def rnd_one_in(n: int) -> bool:
    return rnd_select_n(n) == 0


def rnd_bool() -> bool:
    return rnd_one_in(2)


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
