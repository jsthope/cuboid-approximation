"""One CPU budget for independent tasks and native nearest-neighbor queries."""

from contextlib import contextmanager
from contextvars import ContextVar
import os
import operator

from threadpoolctl import threadpool_limits


_workers = ContextVar("cuboid_workers", default=1)


def available_cpus():
    """Respect process affinity instead of using the host's total CPU count."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def worker_count(requested=0):
    try:
        requested = operator.index(requested)
    except TypeError as exc:
        raise ValueError("workers must be a nonnegative integer") from exc
    if requested < 0:
        raise ValueError("workers must be nonnegative (0 selects automatic parallelism)")
    available = available_cpus()
    return min(available, requested or 4)


def current_workers():
    return _workers.get()


@contextmanager
def execution_workers(requested=0):
    """Keep inner BLAS serial while outer tasks use the shared CPU budget.

    Restore native limits and the calling context even if a run fails. Pipeline
    runs are sequential within a process; independent runs use subprocesses.
    """
    count = worker_count(requested)
    token = _workers.set(count)
    try:
        with threadpool_limits(limits=1, user_api="blas"):
            yield count
    finally:
        _workers.reset(token)
