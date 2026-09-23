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


def _runtime_str(seconds: float):
    m, s = divmod(seconds, 60)
    return f"{m} minutes, {s} seconds"


@dataclass
class SerialExecutionResult:
    submit: CompletedProcess
    runtime_seconds: float

    @property
    def runtime(self):
        return _runtime_str(seconds=self.runtime_seconds)


@dataclass
class DistributedExecutionResult:
    submit: CompletedProcess
    fetch: CompletedProcess
    runtime_seconds: float

    @property
    def runtime(self):
        return _runtime_str(seconds=self.runtime_seconds)


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
            out = run_terminal_commands(
                cmds,
                stdout=workspace.logs_dir / "stdout_serial_run.log",
                stderr=workspace.logs_dir / "stderr_serial_run.log",
            )
            stop = perf_counter()

        return SerialExecutionResult(
            submit=out,
            runtime_seconds=stop - start,
        )

    def run(
        self, workspace: Workspace, overwrite: bool = False
    ) -> SerialExecutionResult | None:
        logger.info("Serial simulation run requested")
        already_complete = workspace.status in [
            WorkspaceStatus.COMPLETED,
            WorkspaceStatus.EXPORTED,
        ]
        if (workspace.status != WorkspaceStatus.READY) and not overwrite:
            if already_complete:
                logger.info(
                    'Simulation not being run because it is already complete and "overwrite"=False'
                )
            else:
                logger.info(
                    'Simulation not being run because status is not "ready" and "overwrite"=False'
                )
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
            logger.exception(
                "Simulation was not successful. Check logs at %s for more info",
                workspace.logs_dir,
            )
            workspace.mark_failed()
            raise exc
        else:
            logger.info("Simulation completed in %s", out.runtime)
            workspace.mark_completed()
            return out


def clean_distributed_run(workspace: Workspace) -> CompletedProcess:
    cmds = [
        "gtsuite",
        "-localdistrib:delete",
        workspace.model_path,
    ]
    logger.info("Distributed run clean invoked for workspace %s", workspace.run_id)
    needs_clean_status = [WorkspaceStatus.FAILED, WorkspaceStatus.RUNNING]
    if workspace.status in needs_clean_status:
        logger.info("Starting deletion of local distributed simulation")
        out = run_terminal_commands(
            cmds,
            stdout=workspace.logs_dir / "stdout_distributed_clean.log",
            stderr=workspace.logs_dir / "stderr_distributed_clean.log",
        )
        logger.info(
            "Local distributed for workspace %s has been deleted", workspace.run_id
        )
        workspace.mark_ready()
        return out
    else:
        logger.info(
            "Local distributed for workspace %s not deleted because workspace status not in %s",
            workspace.run_id,
            [st.value for st in needs_clean_status],
        )


def clean_all_distributed_runs(repo: WorkspaceRepository) -> list[CompletedProcess]:
    out = []
    for workspace in repo._workspaces.values():
        out.append(clean_distributed_run(workspace))
    return out


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
    ) -> DistributedExecutionResult | None:
        logger.info("Distributed simulation run called")
        clean_distributed_run(workspace)
        already_complete = workspace.status in [
            WorkspaceStatus.COMPLETED,
            WorkspaceStatus.EXPORTED,
        ]
        if (workspace.status != WorkspaceStatus.READY) and not overwrite:
            if already_complete:
                logger.info(
                    'Simulation not being run because it is already complete and "overwrite"=False'
                )
            else:
                logger.info(
                    'Simulation not being run because status is not "ready" and "overwrite"=False'
                )
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
            logger.exception(
                "Simulation was not successful. Check logs at %s for more info",
                workspace.logs_dir,
            )
            workspace.mark_failed()
            raise exc
        else:
            logger.info("Simulation completed in: %s", out.runtime)
            workspace.mark_completed()
            return out

    def submit(self, workspace: Workspace) -> CompletedProcess:
        cmds = self._build_submit_commands(workspace)
        logger.info(
            "Submitting distributed simulation with %s licenses requested",
            self.licenses,
        )
        out = run_terminal_commands(
            cmds,
            stdout=workspace.logs_dir / "stdout_distributed_submit.log",
            stderr=workspace.logs_dir / "stderr_distributed_submit.log",
        )
        logger.info("Distributed simulation submission complete")
        return out

    def fetch(self, workspace: Workspace) -> CompletedProcess:
        logger.info("Fetching distributed simulation")
        cmds = self._build_fetch_commands(workspace)
        out = run_terminal_commands(
            cmds,
            stdout=workspace.logs_dir / "stdout_distributed_fetch.log",
            stderr=workspace.logs_dir / "stderr_distributed_fetch.log",
        )
        logger.info("Distributed simulation results fetch complete")
        return out
