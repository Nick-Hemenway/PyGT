import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def write_slowly(path: Path | str, text: str, delay: float = 0.001):
    time.sleep(delay)
    Path(path).write_text(text)


@dataclass
class RaiseNTimesBeforeReturn:
    exception: Exception
    count: int
    val: Any
    exc_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        if self.exc_kwargs is None:
            self.exc_kwargs = {}
        self.total_raised = 0

    def __call__(self):
        if self.total_raised < self.count:
            self.total_raised += 1
            raise self.exception(**self.exc_kwargs)
        else:
            return self.val


@dataclass
class DelayedExecution:
    func: Callable
    count: int

    def __post_init__(self):
        self.total = 0

    def __call__(self):
        if self.total < self.count:
            self.total += 1
            pass
        else:
            return self.func()
