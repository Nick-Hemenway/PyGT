import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Self


def write_slowly(path: Path | str, text: str, delay: float = 0.001):
    time.sleep(delay)
    Path(path).write_text(text)


def raise_exception(exc: Exception):
    raise exc


@dataclass
class WriteFile:
    file: Path
    return_val: Any

    def __call__(self):
        self.file.write_text("some text")
        return self.return_val


@dataclass
class FunctionSwitch:
    count: int
    func1: Callable
    func2: Callable

    def __post_init__(self):
        self.total = 0

    def __call__(self):
        if self.total < self.count:
            self.total += 1
            return self.func1()
        else:
            return self.func2()

    @classmethod
    def raise_exception_n_times(
        cls, exc: Exception, count: int, return_val: Any
    ) -> Self:
        def func2():
            return return_val

        return cls(
            count=count,
            func1=lambda: raise_exception(exc),
            func2=func2,
        )
