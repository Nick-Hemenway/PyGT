import hashlib
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory

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
    stat = path.stat()
    return (
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_size,
    )


class RequireFileTouch:
    def __init__(self, *paths: Path | str):
        self.paths = [Path(p) for p in paths]
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


@dataclass
class _FileState:
    path: Path
    original_existed: bool
    backup_path: Path | None


class RequireFileCreation:
    """
    Ensure one or more files are created (or recreated) within the context.

    Existing files are moved aside before entering the context. New files
    must be created at the original locations while inside the context.

    Examples
    --------
    with RequireFileCreation(
        "results.dat",
        "summary.json",
        "log.txt",
    ):
        simulation.run()
    """

    def __init__(self, *paths: str | Path):
        if not paths:
            raise ValueError("At least one path must be provided.")

        self._paths = [Path(p) for p in paths]
        self._temp_dir: TemporaryDirectory | None = None
        self._states: dict[Path, _FileState] = {}

    def __enter__(self):
        self._temp_dir = TemporaryDirectory()
        temp_dir = Path(self._temp_dir.name)

        for path in self._paths:
            if path.exists():
                backup_path = temp_dir / path.name

                # Avoid collisions if two watched files have the same name.
                counter = 1
                while backup_path.exists():
                    backup_path = temp_dir / f"{counter}_{path.name}"
                    counter += 1

                shutil.move(path, backup_path)

                self._states[path] = _FileState(
                    path=path,
                    original_existed=True,
                    backup_path=backup_path,
                )
            else:
                self._states[path] = _FileState(
                    path=path,
                    original_existed=False,
                    backup_path=None,
                )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            # Restore originals if the body of the context failed.
            if exc_type is not None:
                if exc_type is FileNotFoundError:
                    raise RuntimeError(
                        "Trying to load file that has been moved inside context RequireFileCreation context"
                    ) from exc_value
                self._restore_all()
                return False

            missing = [str(path) for path in self._paths if not path.exists()]

            if missing:
                self._restore_all()

                raise FileNotUpdatedError(
                    "Expected the following files to be created: " + ", ".join(missing)
                )

            return False

        finally:
            if self._temp_dir is not None:
                self._temp_dir.cleanup()

    def _restore_all(self):
        for state in self._states.values():
            if (
                state.original_existed
                and state.backup_path is not None
                and state.backup_path.exists()
            ):
                if state.path.exists():
                    state.path.unlink()

                shutil.move(state.backup_path, state.path)
