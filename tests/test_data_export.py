from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Callable

import pytest

from pygt import DataExporter, ExportSpec, Workspace
from pygt.exceptions import FileNotUpdatedError, GTLicenseError
from tests.helpers import RaiseNTimesBeforeReturn, write_slowly


@dataclass
class GTErrorRunner:
    func: Callable
    path_to_update: Path | None = None

    def run(
        self, cmds: list[str], stdout: str | Path | None, stderr: str | Path | None
    ) -> CompletedProcess:
        if self.path_to_update is not None:
            write_slowly(self.path_to_update, "some_text")
        return self.func()


@pytest.fixture
def license_failures() -> int:
    return 3


@pytest.fixture
def completed() -> CompletedProcess:
    return CompletedProcess(args=(), returncode=1)


@pytest.fixture
def no_license_func(completed: CompletedProcess, license_failures: int) -> Callable:
    return RaiseNTimesBeforeReturn(
        exception=GTLicenseError,
        count=license_failures,
        val=completed,
        exc_kwargs={"message": "test", "gt_info": ()},
    )


def test_export_recovers_from_gt_license_error(
    workspace: Workspace,
    export_spec1: ExportSpec,
    no_license_func: Callable,
    license_failures: int,
    completed: CompletedProcess,
):
    runner = GTErrorRunner(
        func=no_license_func,
        path_to_update=workspace.export_output_path(export_spec1),
    )
    exporter = DataExporter(delay=0, runner=runner, retries=license_failures + 1)
    result = exporter.export(
        workspace=workspace,
        export_spec=export_spec1,
        overwrite=True,
    )
    assert result == completed


def test_export_recovers_if_file_not_immediately_updated(
    workspace: Workspace,
    export_spec1: ExportSpec,
    no_license_func: Callable,
    license_failures: int,
    completed: CompletedProcess,
):
    runner = GTErrorRunner(
        func=no_license_func,
        path_to_update=workspace.export_output_path(export_spec1),
    )
    exporter = DataExporter(delay=0, runner=runner, retries=license_failures + 1)
    result = exporter.export(
        workspace=workspace,
        export_spec=export_spec1,
        overwrite=True,
    )
    assert result == completed


def test_export_raises_error_if_file_never_updated(
    workspace: Workspace,
    export_spec1: ExportSpec,
    no_license_func: Callable,
    license_failures: int,
):
    # Exporter told to retry more times than GTLicenseError is raised,
    # but the file is never updated so it still raises an error
    runner = GTErrorRunner(
        func=no_license_func,
    )
    exporter = DataExporter(delay=0, runner=runner, retries=license_failures + 1)
    with pytest.raises(FileNotUpdatedError):
        exporter.export(
            workspace=workspace,
            export_spec=export_spec1,
            overwrite=True,
        )


# def test_retry_export_on_file_not_updated_error(
#     workspace: Workspace,
#     export_spec1: ExportSpec,
#     no_license_func: Callable,
#     license_failures: int,
#     completed: CompletedProcess,
# ):
#     runner = GTErrorRunner(
#         func=no_license_func,
#         path_to_update=workspace.export_output_path(export_spec1),
#     )
#     exporter = DataExporter(delay=0, runner=runner, retries=license_failures+1)
#     result = exporter.export(
#         workspace=workspace,
#         export_spec=export_spec1,
#         overwrite=True,
#     )
#     assert result == completed


# def test_retry_export_on_file_not_updated_error(
#     workspace: Workspace, export_spec1: ExportSpec
# ):
#     out = CompletedProcess(args=(), returncode=1)
#     f = RaiseNTimesBeforeReturn(
#         exception=GTLicenseError,
#         count=3,
#         val=out,
#         exc_kwargs={"message": "test", "gt_info": ()},
#     )
#     runner = GTErrorRunner(
#         func=f,
#         path_to_update=workspace.export_output_path(export_spec1),
#         update_results_file=True,
#     )
#     exporter = DataExporter(delay=0, runner=runner, retries=10)
#     result = exporter.export(
#         workspace=workspace,
#         export_spec=export_spec1,
#         overwrite=True,
#     )
#     assert result == out


# def test_export_raises_error_if_file_never_updated(
#     workspace: Workspace, export_spec1: ExportSpec
# ):
#     # GTLicenseError only raised 3 times
#     # Exporter told to retry 5 times, but the file
#     # is never updated
#     out = CompletedProcess(args=(), returncode=1)

#     runner = GTErrorRunner(
#         func=f,
#         path_to_update=workspace.export_output_path(export_spec1),
#         update_results_file=False,
#     )
#     exporter = DataExporter(delay=0, runner=runner, retries=5)
#     with pytest.raises(FileNotUpdatedError):
#         exporter.export(
#             workspace=workspace,
#             export_spec=export_spec1,
#             overwrite=True,
#         )
