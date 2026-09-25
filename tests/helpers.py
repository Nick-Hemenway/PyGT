from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class WriteFile:
    file: Path
    return_val: Any

    def __call__(self, *args, **kwargs):
        self.file.write_text("some text")
        return self.return_val


@dataclass
class FunctionSwitch:
    count: int
    func1: Callable
    func2: Callable

    def __post_init__(self):
        self.total = 0

    def __call__(self, *args, **kwargs):
        if self.total < self.count:
            self.total += 1
            return self.func1(*args, **kwargs)
        else:
            return self.func2(*args, **kwargs)

    @classmethod
    def delayed_action(cls, count: int, func: Callable, exc: Exception | None = None):
        def do_nothing(*args, **kwargs):
            pass

        def raise_exception(exc: Exception):
            raise exc

        if exc is None:
            func1 = do_nothing
        else:

            def func1(*args, **kwargs):
                return raise_exception(exc)

        return cls(
            count=count,
            func1=func1,
            func2=func,
        )
