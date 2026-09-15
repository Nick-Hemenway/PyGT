import logging
import re
import subprocess

import pycam.exceptions as exc

logger = logging.getLogger(__name__)


def validate_gt_outputs(result: subprocess.CompletedProcess):
    stderr = result.stderr
    stdout = result.stdout
    cmds = result.args

    if "no valid license" in stderr.lower():
        raise exc.GTLicenseError("GT did not acquire a license", gt_info=result)

    if "cannot connect to license server" in stderr.lower():
        raise exc.GTLicenseError("GT cannot connect to license server", gt_info=result)

    if "cannot continue to the next case" in stdout.lower() and (
        "Case Number:        2" not in stdout
    ):
        raise exc.GTFirstCaseFailedError("Cannot continue to next case", gt_info=result)

    if "cannot continue to the next case" in stdout.lower():
        # blanket catch all for if cases stop
        raise exc.GTFatalError("Cannot continue to next case", gt_info=result)

    if "ERROR!! Failed to execute gtsuite File" in stderr:
        # can happen if poor args given at command line (e.g. wrong file path)
        raise exc.GTFailedExecution("Failed to execute model!", gt_info=result)

    if ("-localdistrib:submit" in cmds) and (
        "Model submitted successfully" not in stdout
    ):
        raise exc.GTDistributedSubmissionFail(
            message="Local-Distributed submission was not successful!", gt_info=result
        )

    if ("-localdistrib:fetch" in cmds) and (
        re.search(r"Fetched \d+ files from simulation", stdout) is None
    ):
        raise exc.GTDistributedFetchFail(
            "Failed to fetch distributed sim results", gt_info=result
        )

    if ("-localdistrib:fetch" in cmds) and (stdout.strip().endswith("not found")):
        raise exc.GTDistributedFetchFail(
            "Failed to fetch distributed sim results", gt_info=result
        )

    if "'gtsuite' is not recognized as an internal or external command" in stderr:
        raise exc.GTNotFoundError(
            message='"gtsuite" command not available at terminal. This is likely an issue with the path environment variable',
            gt_info=result,
        )


def run_terminal_commands(cmds: list[str]) -> subprocess.CompletedProcess:
    logger.info("Invoking GT at Terminal")
    logger.info(f"Input Commands: {cmds}")
    outputs = subprocess.run(cmds, capture_output=True, text=True, shell=True)
    logger.info("GT call complete")
    logger.info(f"stdout: {outputs.stdout}")
    logger.info(f"stderr: {outputs.stderr}")
    validate_gt_outputs(outputs)
    return outputs
