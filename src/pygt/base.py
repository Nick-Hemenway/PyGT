from pathlib import Path
from subprocess import CompletedProcess
from typing import Protocol


class ProcessRunner(Protocol):
    def run(
        self,
        cmds: list[str],
        stdout: str | Path | None,
        stderr: str | Path | None,
    ) -> CompletedProcess: ...
