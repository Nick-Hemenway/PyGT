from pathlib import Path
from subprocess import CompletedProcess


class GTError(Exception):
    def __init__(self, message: str, gt_info: CompletedProcess):
        self.message = message
        self.gt_info = gt_info
        super().__init__(self.message)


class GTLicenseError(GTError):
    def __init__(self, message: str, gt_info: CompletedProcess):
        super().__init__(message, gt_info)


class GTFirstCaseFailedError(GTError):
    def __init__(self, message: str, gt_info: CompletedProcess):
        super().__init__(message, gt_info)


class GTFailedExecution(GTError):
    def __init__(self, message: str, gt_info: CompletedProcess):
        super().__init__(message, gt_info)


class GTFatalError(GTError):
    def __init__(self, message: str, gt_info: CompletedProcess):
        super().__init__(message, gt_info)


class GTDistributedSubmissionFail(GTError):
    def __init__(self, message: str, gt_info: CompletedProcess):
        super().__init__(message, gt_info)


class GTDistributedFetchFail(GTError):
    def __init__(self, message: str, gt_info: CompletedProcess):
        super().__init__(message, gt_info)


class GTNotFoundError(GTError):
    def __init__(self, message: str, gt_info: CompletedProcess):
        super().__init__(message, gt_info)


class FileNotUpdatedError(Exception):
    def __init__(self, message, path: Path | str | None = None):
        self.message = message
        self.path = path
        super().__init__(self.message)


class MissingSignalError(Exception):
    def __init__(self, signal: str):
        self.signal = signal
        message = f"{signal} not present in CaseData"
        super().__init__(message)


class InconsistentParametersError(Exception):
    def __init__(self):
        message = (
            "Existing parameter file is not consistent with cases being provided to DesignWriter. "
            'Either create a new workspace, or explicitly set "overwrite=True"'
        )
        super().__init__(message)
