import logging
import subprocess
from pathlib import Path
from time import perf_counter

from .error_check import raise_on_gt_errors

logger = logging.getLogger(__name__)


class SubprocessRunner:
    def run(
        self,
        cmds: list[str],
        stdout: str | Path | None,
        stderr: str | Path | None,
    ) -> subprocess.CompletedProcess:
        logger.info("Starting subprocess: %s", cmds)
        start = perf_counter()
        outputs = subprocess.run(cmds, capture_output=True, text=True, shell=True)
        stop = perf_counter()
        m, s = divmod(stop - start, 60)
        logger.info("Subprocess complete")
        logger.info("Runtime: %d minutes %.3f seconds", m, s)
        if stdout is not None:
            stdout = Path(stdout)
            stdout.write_text(outputs.stdout)
            logger.debug("Stdout logs written to %s", stdout)
        if stderr is not None:
            stderr = Path(stderr)
            stderr.write_text(outputs.stderr)
            logger.debug("Stderr logs written to %s", stderr)
        raise_on_gt_errors(outputs)
        return outputs
