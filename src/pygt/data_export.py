from dataclasses import dataclass
from subprocess import CompletedProcess

from pycam.exceptions import FileNotUpdatedError, GTLicenseError
from pycam.output_spec import ExportSpec
from pycam.terminal import run_terminal_commands
from pycam.util import FileWatcher, retry
from pycam.workspace import Workspace, WorkspaceStatus


@dataclass
class DataExporter:
    retries: int = 500
    delay: float = 5
    backoff: float = 2
    max_delay: float = 600

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
            out = run_terminal_commands(cmds)
        return out

    def export(
        self,
        workspace: Workspace,
        export_spec: ExportSpec,
        overwrite: bool = False,
    ) -> CompletedProcess:
        if (workspace.status != WorkspaceStatus.COMPLETED) and not overwrite:
            return None

        wrapper = retry(
            exceptions=(GTLicenseError, FileNotUpdatedError),
            tries=self.retries,
            delay=self.delay,
            backoff=self.backoff,
        )
        out = wrapper(self._export_once)(workspace, export_spec)
        workspace.mark_exported()
        return out
