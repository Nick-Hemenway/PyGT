import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from subprocess import CompletedProcess
from time import perf_counter

from .exceptions import FileNotUpdatedError, GTLicenseError
from .terminal import run_terminal_commands
from .util import FileWatcher, retry
from .workspace import Workspace, WorkspaceRepository, WorkspaceStatus

logger = logging.getLogger(__name__)


class ExecutionStrategy(ABC):
    @abstractmethod
    def run(self, workspace: Workspace):
        pass


@dataclass
class SerialExecutionResult:
    submit: CompletedProcess
    runtime_seconds: float

    @property
    def runtime(self):
        m, s = divmod(self.runtime_seconds, 60)
        return f"{m} minutes, {s} seconds"


@dataclass
class DistributedExecutionResult(SerialExecutionResult):
    fetch: CompletedProcess


@dataclass
class SerialExecution(ExecutionStrategy):
    version: str | None = None
    use_param_file: bool = True
    continue_if_case_fails: bool = True
    other_flags: list | None = None
    retries: int = 500
    delay: float = 5
    backoff: float = 2
    max_delay: float = 600

    def __post_init__(self):
        if self.other_flags is None:
            self.other_flags = []
        if not isinstance(self.other_flags, (list, tuple)):
            raise ValueError("other_flags must be list, tuple, or None")

    def _build_commands(self, workspace: Workspace) -> list:
        cmds = ["gtsuite"]

        # Optional Flags
        if self.version is not None:
            cmds.append(f"-v{self.version}")
        if self.continue_if_case_fails:
            cmds.append("-u:on")
        if self.use_param_file:
            cmds.append(f"-extparamfilearray:{workspace.param_file_path}")

        cmds.extend(self.other_flags)

        # Model path is always appended as last argument after all flags
        cmds.append(workspace.model_path)
        return cmds

    def _run_once(self, workspace: Workspace) -> SerialExecutionResult:
        if self.use_param_file and not workspace.param_file_path.exists():
            raise RuntimeError(
                (
                    "Param file required for simulation but "
                    "it does not exist in the current workspace"
                )
            )
        cmds = self._build_commands(workspace)
        # Watch .glx file and make sure it's updated. Otherwise raise exception
        with FileWatcher(workspace.model_path.with_suffix(".glx")):
            start = perf_counter()
            out = run_terminal_commands(cmds)
            stop = perf_counter()
        return SerialExecutionResult(
            submit=out,
            runtime_seconds=stop - start,
        )

    def run(
        self, workspace: Workspace, overwrite: bool = False
    ) -> SerialExecutionResult:
        if (workspace.status != WorkspaceStatus.READY) and not overwrite:
            return None
        wrapper = retry(
            exceptions=(GTLicenseError, FileNotUpdatedError),
            tries=self.retries,
            delay=self.delay,
            backoff=self.backoff,
        )
        try:
            out = wrapper(self._run_once)(workspace)
        except Exception as exc:
            workspace.mark_failed()
            raise exc
        else:
            workspace.mark_completed()
            return out


def clean_distributed_run(workspace: Workspace) -> CompletedProcess:
    cmds = [
        "gtsuite",
        "-localdistrib:delete",
        workspace.model_path,
    ]
    if workspace.status in [WorkspaceStatus.FAILED, WorkspaceStatus.RUNNING]:
        out = run_terminal_commands(cmds)
        workspace.mark_ready()
        return out


def clean_all_distributed_runs(repo: WorkspaceRepository) -> list[CompletedProcess]:
    out = []
    for workspace in repo._workspaces.values():
        out.append(clean_distributed_run(workspace))


@dataclass
class DistributedExecution(ExecutionStrategy):
    licenses: int
    packet_size: int
    version: str | None = None
    use_param_file: bool = True
    continue_if_case_fails: bool = True
    other_submit_flags: list | None = None
    other_fetch_flags: list | None = None
    retries: int = 500
    delay: float = 5
    backoff: float = 2
    max_delay: float = 600

    def __post_init__(self):
        if self.other_submit_flags is None:
            self.other_submit_flags = []
        if not isinstance(self.other_submit_flags, (list, tuple)):
            raise ValueError("other_submit_flags must be list, tuple, or None")
        if self.other_fetch_flags is None:
            self.other_fetch_flags = []
        if not isinstance(self.other_fetch_flags, (list, tuple)):
            raise ValueError("other_fetch_flags must be list, tuple, or None")

    def _build_submit_commands(self, workspace: Workspace) -> list:
        cmds = ["gtsuite"]

        # Optional Flags (Used for Serial and Distributed)
        if self.version is not None:
            cmds.append(f"-v{self.version}")
        if self.continue_if_case_fails:
            cmds.append("-u:on")
        if self.use_param_file:
            cmds.append(f"-extparamfilearray:{workspace.param_file_path}")

        cmds.extend(self.other_submit_flags)

        # Distributed Specific Flags
        distributed_flags = [
            "-localdistrib:submit",
            "-forcemerge",
            f"-lic:{self.licenses}",
            f"-casemin:{self.packet_size}",
            f"-casemax:{self.packet_size}",
        ]
        cmds.extend(distributed_flags)

        # Model path is always appended as last argument after all flags
        cmds.append(workspace.model_path)
        return cmds

    def _build_fetch_commands(self, workspace: Workspace) -> list:
        cmds = [
            "gtsuite",
            "-localdistrib:fetch",
        ]
        # Optional Flags (Used for Serial and Distributed)
        if self.version is not None:
            cmds.append(f"-v{self.version}")

        cmds.extend(self.other_fetch_flags)
        # Model path is always appended as last argument after all flags
        cmds.append(workspace.model_path)
        return cmds

    def _run_once(self, workspace: Workspace) -> DistributedExecutionResult:
        if self.use_param_file and not workspace.param_file_path.exists():
            raise RuntimeError(
                (
                    "Param file required for simulation but "
                    "it does not exist in the current workspace"
                )
            )
        # Watch .glx file and make sure it's updated. Otherwise raise exception
        with FileWatcher(workspace.model_path.with_suffix(".glx")):
            start = perf_counter()
            submit_result = self.submit(workspace)
            workspace.mark_running()
            fetch_result = self.fetch(workspace)
            stop = perf_counter()
        return DistributedExecutionResult(
            submit=submit_result,
            fetch=fetch_result,
            runtime_seconds=stop - start,
        )

    def run(
        self, workspace: Workspace, overwrite: bool = False
    ) -> DistributedExecutionResult:
        clean_distributed_run(workspace)
        if (workspace.status != WorkspaceStatus.READY) and not overwrite:
            return None
        wrapper = retry(
            exceptions=(GTLicenseError, FileNotUpdatedError),
            tries=self.retries,
            delay=self.delay,
            backoff=self.backoff,
        )
        try:
            out = wrapper(self._run_once)(workspace)
        except Exception as exc:
            workspace.mark_failed()
            raise exc
        else:
            workspace.mark_completed()
            return out

    def submit(self, workspace: Workspace) -> CompletedProcess:
        cmds = self._build_submit_commands(workspace)
        return run_terminal_commands(cmds)

    def fetch(self, workspace: Workspace) -> CompletedProcess:
        cmds = self._build_fetch_commands(workspace)
        return run_terminal_commands(cmds)
