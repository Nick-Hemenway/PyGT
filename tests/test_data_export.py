from contextlib import nullcontext
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from pygt import DataExporter, ExportSpec, Workspace
from pygt.base import ProcessRunner
from pygt.exceptions import FileNotUpdatedError, GTLicenseError
from tests.helpers import FunctionSwitch, WriteFile


@pytest.fixture
def completed() -> CompletedProcess:
    return CompletedProcess(args=(), returncode=1)


@pytest.fixture
def license_exception(completed) -> GTLicenseError:
    return GTLicenseError(message="", gt_info=completed)


@pytest.fixture
def mock_runner() -> MagicMock:
    return MagicMock(spec=ProcessRunner)


def test_export_recovers_from_gt_license_error(
    workspace: Workspace,
    export_spec1: ExportSpec,
    completed: CompletedProcess,
    mock_runner: MagicMock,
    license_exception: GTLicenseError,
):
    mock_runner.run.side_effect = FunctionSwitch.delayed_action(
        count=2,
        func=lambda *args, **kwargs: completed,
        exc=license_exception,
    )
    exporter = DataExporter(delay=0, runner=mock_runner, retries=3)

    with patch("pygt.data_export.RequireFileTouch", return_value=nullcontext()):
        result = exporter.export(
            workspace=workspace,
            export_spec=export_spec1,
            overwrite=True,
        )
    assert result == completed
    assert mock_runner.run.call_count == 3


def test_export_fails_if_more_license_errors_than_retries(
    export_spec1: ExportSpec,
    license_exception: GTLicenseError,
    mock_runner: MagicMock,
    workspace: Workspace,
):
    mock_runner.run.side_effect = license_exception
    exporter = DataExporter(delay=0, runner=mock_runner, retries=3)
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
    mock_runner: MagicMock,
):
    file_fail_count = 2
    mock_runner.run.side_effect = FunctionSwitch.delayed_action(
        count=file_fail_count,
        func=WriteFile(
            file=workspace.export_output_path(export_spec1), return_val=completed
        ),
    )
    exporter = DataExporter(delay=0, runner=mock_runner, retries=file_fail_count + 1)
    result = exporter.export(
        workspace=workspace,
        export_spec=export_spec1,
        overwrite=True,
    )
    assert result == completed


def test_export_raises_error_if_file_never_updated(
    workspace: Workspace,
    export_spec1: ExportSpec,
    mock_runner: MagicMock,
):
    mock_runner.run.return_value = None
    exporter = DataExporter(delay=0, runner=mock_runner, retries=3)
    with pytest.raises(FileNotUpdatedError):
        exporter.export(
            workspace=workspace,
            export_spec=export_spec1,
            overwrite=True,
        )
