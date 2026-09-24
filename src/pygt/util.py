import hashlib
import logging
import subprocess
import time
from functools import wraps
from pathlib import Path

from .exceptions import FileNotUpdatedError

logger = logging.getLogger(__name__)


def get_git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def sanitize_path(path: Path | str) -> Path:
    return Path(path).resolve()


def retry(
    exceptions: Exception | tuple[Exception],
    tries: int = 20,
    delay: float = 5,
    backoff: float = 2,
):
    """Retry calling the decorated function using an exponential backoff.

    Args:
        exceptions: An Exception or tuple of exceptions to catch
        tries: Number of times to try (not retry) before giving up.
        delay: Initial delay between retries in seconds.
        backoff: Backoff multiplier e.g. value of 2 will make the delay double each retry.

    Returns:
        The decorated function.
    """
    if isinstance(exceptions, Exception):
        exceptions = (exceptions,)

    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    logger.warning("Retry: %s, Retrying in %s seconds...", e, mdelay)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry

    return deco_retry


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_mod_time(path: Path) -> int:
    if not path.exists():
        return None
    return path.stat().st_mtime_ns


class FileWatcher:
    def __init__(self, paths: Path | str | list[Path | str]):
        if not isinstance(paths, list):
            paths = [paths]
        self.paths = [Path(p) for p in paths if p is not None]
        self.last_times = {}
        self.last_hash = {}

    def __enter__(self):
        for path in self.paths:
            self.last_times[path] = last_mod_time(path)
            self.last_hash[path] = file_hash(path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            raise exc_value
        for path in self.last_times.keys():
            old_time = self.last_times[path]
            old_hash = self.last_hash[path]
            new_time = last_mod_time(path)
            new_hash = file_hash(path)

            if (new_time == old_time) and (new_hash == old_hash):
                raise FileNotUpdatedError(
                    f"File {path.name} has not been updated!", path=path
                )
