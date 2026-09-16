import json
import logging
import os
import subprocess
import time
from functools import wraps
from pathlib import Path

from .exceptions import FileNotUpdatedError, GTError

logger = logging.getLogger(__name__)


def get_git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def sanitize_path(path: Path | str) -> Path:
    return Path(path).resolve()


def retry(
    exceptions: tuple[Exception],
    tries: int = 20,
    delay: float = 5,
    backoff: float = 2,
):
    """Retry calling the decorated function using an exponential backoff.

    Args:
        exceptions: A tuple of exception classes to catch.
        tries: Number of times to try (not retry) before giving up.
        delay: Initial delay between retries in seconds.
        backoff: Backoff multiplier e.g. value of 2 will make the delay double each retry.

    Returns:
        The decorated function.
    """
    if not isinstance(exceptions, tuple):
        raise ValueError('"exceptions" must be a tuple of exceptions')

    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    logger.warning("Retry: %s, Retrying in %s seconds...", e, mdelay)
                    if isinstance(e, GTError):
                        logger.warning("%s", json.dumps(e.gt_info))
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry

    return deco_retry


class FileWatcher:
    def __init__(self, paths: Path | str | list[Path | str]):
        if not isinstance(paths, list):
            paths = [paths]
        self.paths = [Path(p) for p in paths if p is not None]
        self.last_times = {}

    def __enter__(self):
        for p in self.paths:
            try:
                self.last_times[p] = os.path.getmtime(p)
            except FileNotFoundError:
                self.last_times[p] = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            raise exc_value
        for path, last_mod_time in self.last_times.items():
            try:
                new_time = os.path.getmtime(path)
            except FileNotFoundError:
                new_time = None
            if new_time == last_mod_time:
                raise FileNotUpdatedError(
                    f"File {path.name} has not been updated!", path=path
                )
