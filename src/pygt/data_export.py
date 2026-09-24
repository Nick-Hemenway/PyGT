import logging
from dataclasses import dataclass
from subprocess import CompletedProcess

from .base import ProcessRunner
from .exceptions import FileNotUpdatedError, GTLicenseError
from .output_spec import ExportSpec
from .terminal import SubprocessRunner
from .util import FileWatcher, retry
from .workspace import Workspace, WorkspaceStatus

logger = logging.getLogger(__name__)


@dataclass
class DataExporter:
    retries: int = 500
    delay: float = 5
    backoff: float = 2
    max_delay: float = 600
    runner: ProcessRunner | None = None

    def __post_init__(self):
        if self.runner is None:
            self.runner = SubprocessRunner()

    @staticmethod
    def _build_commands(workspace: Workspace, export_spec: ExportSpec) -> list:
        cmds = [
            "gtexport",
            workspace.export_config_path(export_spec),
            workspace.model_path.with_suffix(".glx"),
            workspace.export_output_path(export_spec),
        ]
        return cmds

    def _export_once(
        self, workspace: Workspace, export_spec: ExportSpec
    ) -> CompletedProcess:
        cmds = self._build_commands(workspace, export_spec)
        with FileWatcher(workspace.export_output_path(export_spec)):
            out = self.runner.run(
                cmds,
                stdout=workspace.logs_dir
                / f"stdout_export_spec_{export_spec.name}.log",
                stderr=workspace.logs_dir
                / f"stderr_export_spec_{export_spec.name}.log",
            )
        return out

    def export(
        self,
        workspace: Workspace,
        export_spec: ExportSpec,
        overwrite: bool = False,
    ) -> CompletedProcess:
        logger.info('Export for ExportSpec "%s" called', export_spec.name)
        if (workspace.status != WorkspaceStatus.COMPLETED) and not overwrite:
            logger.info(
                'Not invoking gtexport. Workspace status must be "completed" or "overwrite" must be True'
            )
            logger.info("Current workspace status: %s", workspace.status)
            return None

        wrapper = retry(
            exceptions=(GTLicenseError, FileNotUpdatedError),
            tries=self.retries,
            delay=self.delay,
            backoff=self.backoff,
        )
        try:
            out = wrapper(self._export_once)(workspace, export_spec)
        except Exception as e:
            logger.exception("Exporting of spec %s failed", export_spec.name)
            raise e
        else:
            logger.info("Data successfully exported")
            workspace.mark_exported()
        return out
