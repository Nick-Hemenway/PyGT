import subprocess

import pytest

from pygt import exceptions as exc
from pygt.terminal import raise_on_gt_errors


def make_result(
    stdout="",
    stderr="",
    returncode=0,
):
    return subprocess.CompletedProcess(
        args=["sim"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_empty_output_does_not_raise():
    res = make_result(stdout="", stderr="")
    raise_on_gt_errors(res)


def test_benign_output_does_not_raise():
    res = make_result(stdout="ERROR", stderr="ERROR")
    raise_on_gt_errors(res)


def test_error_pattern_in_stdout_raises():
    res = make_result(stdout="cannot continue to the next case")
    with pytest.raises(exc.GTError):
        raise_on_gt_errors(res)


def test_error_pattern_in_stderr_raises():
    res = make_result(stderr="cannot connect to license server")
    with pytest.raises(exc.GTError):
        raise_on_gt_errors(res)


def test_error_pattern_in_both_stdout_and_stderr_raises():
    res = make_result(
        stderr="cannot connect to license server",
        stdout="cannot continue to the next case",
    )
    with pytest.raises(exc.GTError):
        raise_on_gt_errors(res)
