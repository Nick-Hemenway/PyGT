import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from pygt.exceptions import FileNotUpdatedError
from pygt.util import RequireFileCreation, RequireFileTouch, retry
from tests.helpers import FunctionSwitch


@pytest.fixture
def file1(tmp_path: Path) -> Path:
    return tmp_path / "test1.txt"


@pytest.fixture
def file2(tmp_path: Path) -> Path:
    return tmp_path / "test2.txt"


def raise_exception(exc: Exception):
    raise exc


# ---------------------------------------------------------------------------- #
#                              Test Retry Wrapper                              #
# ---------------------------------------------------------------------------- #


def test_retry_single_exception_tuple():
    f = FunctionSwitch.raise_exception_n_times(exc=ValueError(), count=4, return_val=10)
    f_retry = retry(exceptions=(ValueError,), tries=20, delay=0)(f)
    assert f_retry() == 10


def test_retry_single_exception():
    f = FunctionSwitch.raise_exception_n_times(exc=ValueError(), count=4, return_val=10)
    f_retry = retry(exceptions=ValueError, tries=20, delay=0)(f)
    assert f_retry() == 10


def test_retry_multiple_exception():
    f = FunctionSwitch.raise_exception_n_times(exc=ValueError(), count=4, return_val=10)
    f_retry = retry(
        exceptions=(RuntimeError, ValueError, AttributeError), tries=20, delay=0
    )(f)
    assert f_retry() == 10


def test_retry_single_exception_tries_too_low():
    f = FunctionSwitch.raise_exception_n_times(exc=ValueError(), count=6, return_val=10)
    f_retry = retry(exceptions=ValueError, tries=4, delay=0)(f)
    with pytest.raises(ValueError):
        f_retry()


def test_retry_with_unexpected_exception():
    f = FunctionSwitch.raise_exception_n_times(exc=RuntimeError, count=3, return_val=10)
    f_retry = retry(exceptions=(ValueError, KeyError), tries=10, delay=0)(f)
    with pytest.raises(RuntimeError):
        f_retry()


# ---------------------------------------------------------------------------- #
#                              Test File Watchers                              #
# ---------------------------------------------------------------------------- #


def touch(path: Path | str):
    path = Path(path)
    try:
        # If the file exists, update its access and modification times to now
        os.utime(path, None)
    except OSError:
        # If the file doesn't exist, open it in append mode to create it
        # 'with' ensures the file is safely closed immediately
        with open(path, "a"):
            pass


def touch_force_future(path: Path | str, offset_seconds: float = 1):
    path = Path(path)
    """Touches a file, forcing the timestamp to advance."""
    # Get current time and add a buffer so it is guaranteed to be unique
    new_time = time.time() + offset_seconds

    if os.path.exists(path):
        os.utime(path, (new_time, new_time))
    else:
        with open(path, "a"):
            pass
        os.utime(path, (new_time, new_time))


type Watcher = RequireFileCreation | RequireFileTouch


@pytest.mark.parametrize("watcher", [RequireFileTouch, RequireFileCreation])
def test_file_watcher_file_created(file1: Path, watcher: Watcher):
    with watcher(file1):
        touch(file1)


@pytest.mark.parametrize("watcher", [RequireFileTouch, RequireFileCreation])
def test_file_watcher_file_overwritten_with_same_content(file1: Path, watcher: Watcher):
    touch(file1)
    with watcher(file1):
        touch_force_future(file1)


def test_file_watcher_file_touched_but_only_read(file1: Path):
    touch(file1)
    with pytest.raises(FileNotUpdatedError):
        with RequireFileTouch(file1):
            _ = file1.read_text()


def test_file_watcher_file_should_not_be_read(file1: Path):
    touch(file1)
    with pytest.raises(RuntimeError):
        with RequireFileCreation(file1):
            _ = file1.read_text()


@pytest.mark.parametrize("watcher", [RequireFileTouch, RequireFileCreation])
def test_file_watcher_file_not_created(file1: Path, watcher: Watcher):
    with pytest.raises(FileNotUpdatedError):
        with watcher(file1):
            pass


@pytest.mark.parametrize("watcher", [RequireFileTouch, RequireFileCreation])
def test_file_watcher_file_created_but_not_updated(file1: Path, watcher: Watcher):
    touch(file1)
    with pytest.raises(FileNotUpdatedError):
        with watcher(file1):
            pass


@pytest.mark.parametrize("watcher", [RequireFileTouch, RequireFileCreation])
def test_file_watcher_one_file_not_created(file1: Path, file2: Path, watcher: Watcher):
    with pytest.raises(FileNotUpdatedError):
        with watcher(file1, file2):
            touch(file1)


@pytest.mark.parametrize("watcher", [RequireFileTouch, RequireFileCreation])
def test_file_watcher_both_file_not_created(file1: Path, file2: Path, watcher: Watcher):
    with pytest.raises(FileNotUpdatedError):
        with watcher(file1, file2):
            pass


@pytest.mark.parametrize("watcher", [RequireFileTouch, RequireFileCreation])
def test_file_watcher_both_file_created(file1: Path, file2: Path, watcher: Watcher):
    with watcher(file1, file2):
        touch(file1)
        touch(file2)


@pytest.mark.parametrize("watcher", [RequireFileTouch, RequireFileCreation])
def test_file_watcher_both_file_overwritten_with_same_content(
    file1: Path, file2: Path, watcher: Watcher
):
    touch(file1)
    touch(file2)
    with watcher(file1, file2):
        touch_force_future(file1)
        touch_force_future(file2)


def test_file_watcher_same_time_stamp_and_same_content(file1: Path):
    # File timestamps are read as the same and content is the same
    # so an error is raised
    touch(file1)
    with patch("pygt.util.last_mod_time", return_value=0):
        with pytest.raises(FileNotUpdatedError):
            with RequireFileTouch(file1):
                touch_force_future(file1)


def test_file_watcher_same_time_stamp_and_same_content_but_file_regenerated(
    file1: Path,
):
    # File timestamps are read as the same and content is the same
    # so an error is raised
    touch(file1)
    with patch("pygt.util.last_mod_time", return_value=0):
        with RequireFileCreation(file1):
            touch_force_future(file1)


@pytest.mark.parametrize("watcher", [RequireFileTouch, RequireFileCreation])
def test_file_watcher_same_time_stamp_but_different_content(
    file1: Path, watcher: Watcher
):
    touch(file1)
    with patch("pygt.util.last_mod_time", return_value=0):
        with watcher(file1):
            file1.write_text("different content")
