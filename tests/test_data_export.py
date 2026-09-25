from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Callable

import pytest

from pygt import DataExporter, ExportSpec, Workspace
from pygt.exceptions import FileNotUpdatedError, GTLicenseError
from tests.helpers import FunctionSwitch, WriteFile, raise_exception


@dataclass
class MockGTRunner:
    func: Callable[[], CompletedProcess]

    def run(
        self, cmds: list[str], stdout: str | Path | None, stderr: str | Path | None
    ) -> CompletedProcess:
        return self.func()


@pytest.fixture
def license_failures() -> int:
    return 3


@pytest.fixture
def completed() -> CompletedProcess:
    return CompletedProcess(args=(), returncode=1)


@pytest.fixture
def license_exception(completed) -> GTLicenseError:
    return GTLicenseError(message="", gt_info=completed)


@pytest.fixture
def no_license_runner(
    workspace: Workspace,
    export_spec1: ExportSpec,
    completed: CompletedProcess,
    license_failures: int,
) -> MockGTRunner:
    return MockGTRunner(
        func=FunctionSwitch(
            count=license_failures,
            func1=lambda: raise_exception(
                GTLicenseError(message="", gt_info=completed)
            ),
            func2=WriteFile(
                file=workspace.export_output_path(export_spec1), return_val=completed
            ),
        )
    )


def test_export_recovers_from_gt_license_error(
    workspace: Workspace,
    export_spec1: ExportSpec,
    license_failures: int,
    no_license_runner: MockGTRunner,
    completed: CompletedProcess,
):
    exporter = DataExporter(
        delay=0, runner=no_license_runner, retries=license_failures + 1
    )
    result = exporter.export(
        workspace=workspace,
        export_spec=export_spec1,
        overwrite=True,
    )
    assert result == completed


def test_export_fails_if_more_license_errors_than_retries(
    workspace: Workspace,
    export_spec1: ExportSpec,
    license_failures: int,
    no_license_runner: MockGTRunner,
    completed: CompletedProcess,
):
    exporter = DataExporter(
        delay=0, runner=no_license_runner, retries=license_failures - 1
    )
    with pytest.raises(GTLicenseError):
        exporter.export(
            workspace=workspace,
            export_spec=export_spec1,
            overwrite=True,
        )


def test_export_recovers_if_file_not_immediately_updated(
    workspace: Workspace,
    export_spec1: ExportSpec,
    completed: CompletedProcess,
):
    def do_nothing():
        pass

    file_fail_count = 3
    runner = MockGTRunner(
        func=FunctionSwitch(
            count=file_fail_count,
            func1=do_nothing,
            func2=WriteFile(
                file=workspace.export_output_path(export_spec1), return_val=completed
            ),
        )
    )
    exporter = DataExporter(delay=0, runner=runner, retries=file_fail_count + 1)
    result = exporter.export(
        workspace=workspace,
        export_spec=export_spec1,
        overwrite=True,
    )
    assert result == completed


def test_export_raises_error_if_file_never_updated(
    workspace: Workspace,
    export_spec1: ExportSpec,
    license_failures: int,
    completed: CompletedProcess,
):
    # Eventually aquire GT license but file still never updated
    runner = MockGTRunner(
        func=FunctionSwitch.raise_exception_n_times(
            exc=GTLicenseError(message="", gt_info=completed),
            count=license_failures,
            return_val=completed,
        )
    )
    exporter = DataExporter(delay=0, runner=runner, retries=license_failures + 1)
    with pytest.raises(FileNotUpdatedError):
        exporter.export(
            workspace=workspace,
            export_spec=export_spec1,
            overwrite=True,
        )
