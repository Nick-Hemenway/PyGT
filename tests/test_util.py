from pathlib import Path

import pytest

from pygt.exceptions import FileNotUpdatedError
from pygt.util import FileWatcher, retry
from tests.helpers import RaiseNTimesBeforeReturn, write_slowly


@pytest.fixture
def file1(tmp_path: Path) -> Path:
    return tmp_path / "test1.txt"


@pytest.fixture
def file2(tmp_path: Path) -> Path:
    return tmp_path / "test2.txt"


def test_retry_single_exception_tuple():
    f = RaiseNTimesBeforeReturn(exception=ValueError, count=4, val=10)
    f_retry = retry(exceptions=(ValueError,), tries=20, delay=0)(f)
    assert f_retry() == 10


def test_retry_single_exception():
    f = RaiseNTimesBeforeReturn(exception=ValueError, count=4, val=10)
    f_retry = retry(exceptions=ValueError, tries=20, delay=0)(f)
    assert f_retry() == 10


def test_retry_multiple_exception():
    f = RaiseNTimesBeforeReturn(exception=ValueError, count=4, val=10)
    f_retry = retry(
        exceptions=(RuntimeError, ValueError, AttributeError), tries=20, delay=0
    )(f)
    assert f_retry() == 10


def test_retry_single_exception_tries_too_low():
    f = RaiseNTimesBeforeReturn(exception=ValueError, count=6, val=10)
    f_retry = retry(exceptions=ValueError, tries=4, delay=0)(f)
    with pytest.raises(ValueError):
        f_retry()


def test_retry_with_unexpected_exception():
    f = RaiseNTimesBeforeReturn(exception=RuntimeError, count=3, val=10)
    f_retry = retry(exceptions=(ValueError, KeyError), tries=10, delay=0)(f)
    with pytest.raises(RuntimeError):
        f_retry()


def test_file_watcher_file_created(file1: Path):
    with FileWatcher(file1):
        file1.write_text("Test text")


def test_file_watcher_file_overwritten_with_same_content(file1: Path):
    message = "Test text"
    file1.write_text(message)
    with FileWatcher(file1):
        # Add small amount of sleep because filewatcher looks at timestamp
        # of when file was last modified. Test can fail occasionally
        # because file updates are so close to each other. This seems super
        # unlikely to happen in practical scenarios
        write_slowly(file1, message)


def test_file_watcher_file_touched_but_only_read(file1: Path):
    file1.write_text("Test text")
    with pytest.raises(FileNotUpdatedError):
        with FileWatcher(file1):
            _ = file1.read_text()


def test_file_watcher_file_not_created(file1: Path):
    with pytest.raises(FileNotUpdatedError):
        with FileWatcher(file1):
            pass


def test_file_watcher_file_created_but_not_updated(file1: Path):
    file1.write_text("Test text")
    with pytest.raises(FileNotUpdatedError):
        with FileWatcher(file1):
            pass


def test_file_watcher_one_file_not_created(file1: Path, file2: Path):
    with pytest.raises(FileNotUpdatedError):
        with FileWatcher([file1, file2]):
            file1.write_text("Test text")


def test_file_watcher_both_file_not_created(file1: Path, file2: Path):
    with pytest.raises(FileNotUpdatedError):
        with FileWatcher([file1, file2]):
            pass


def test_file_watcher_both_file_created(file1: Path, file2: Path):
    with FileWatcher([file1, file2]):
        file1.write_text("test")
        file2.write_text("test")


def test_file_watcher_both_file_overwritten_with_same_content(file1: Path, file2: Path):
    message = "test"
    file1.write_text(message)
    file2.write_text(message)
    with FileWatcher([file1, file2]):
        write_slowly(file1, message)
        write_slowly(file2, message)
