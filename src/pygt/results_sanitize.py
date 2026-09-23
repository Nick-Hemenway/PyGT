import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from .exceptions import MissingSignalError
from .simdata import CaseData

logger = logging.getLogger(__name__)


class QDAResultsAlign:
    def __init__(
        self,
        timing_param: str,
        intake: bool,
        start_of_cycle_angle: float,
        signals: list | tuple | Callable,
    ):
        self.timing_param = timing_param
        if intake:
            self.sign = 1
        else:
            self.sign = -1
        self.start_of_cycle_angle = start_of_cycle_angle
        self.signals = signals

    def angle_offset(self, case: CaseData):
        # the QDA results are given as a function of elapsed cam angle
        # from the start of the simulation
        offset_from_tdc = self.sign * getattr(case, self.timing_param)  # crank degrees
        # angle from zero crank axis to peak valve lift
        global_timing_angle = 360 + offset_from_tdc  # crank degrees
        # The QDA data is represented in terms total elapsed crank angle from start of simulation.
        # Thus, we must add the additional start of cycle angle offset to the timing angle
        total_qda_offset = (
            global_timing_angle - self.start_of_cycle_angle
        )  # crank degrees

        return total_qda_offset / 2  # convert offset to cam degrees

    def signal_names(self, case: CaseData):
        if isinstance(self.signals, (list, tuple)):
            signals = self.signals
        else:
            signals = [sig for sig in case.signals if self.signals(sig)]
        return signals

    def center_of_mass(self, x: NDArray, y: NDArray) -> float:
        z = np.dot(x, np.abs(y))
        total = np.sum(np.abs(y))
        if total == 0:
            return 0
        else:
            return z / total

    def is_wrapped(self, case: CaseData, signal: str) -> bool:
        return self.center_of_mass(x=case.get(case.index_name), y=case.get(signal)) < 0

    def wrap_offset(self, case: CaseData, signal: str) -> float:
        # Sometimes the QDA results come out wrapped, such that the main profiles
        # occur at negative cam angles. We must check if this is the case and if so
        # add a full cam rotation before subtracting the offset
        if self.is_wrapped(case, signal):
            return 360
        else:
            return 0

    def align_col(
        self,
        x: NDArray,
        y: NDArray,
        wrap_offset: float,
        angle_offset: float,
    ) -> NDArray:
        f = interp1d(x, y, kind="linear", fill_value=0, bounds_error=False)
        return f(x - wrap_offset + angle_offset)

    def __call__(self, case: CaseData) -> CaseData:
        signals = {key: np.asarray(val).copy() for key, val in case.signals.items()}
        angle_offset = self.angle_offset(case)
        try:
            x = case.get(case.index_name)
        except MissingSignalError:
            # The whole case must have failed for the index to not be present
            # We'll just return the original CaseData object directly and log a warning
            logger.warning(
                (
                    "Case does not have index signal present "
                    "and thus QDA alignment cannot be performed. "
                )
            )
            logger.warning("Case Status: %s", case.status)
            return case
        for sig in self.signal_names(case):
            try:
                y = case.get(sig)
            except MissingSignalError:
                continue
            else:
                signals[sig] = self.align_col(
                    x,
                    y=y,
                    wrap_offset=self.wrap_offset(case, sig),
                    angle_offset=angle_offset,
                )
        return CaseData(
            case_id=case.case_id,
            design_id=case.design_id,
            design_variables=case.design_variables,
            operating_point=case.operating_point,
            index_name=case.index_name,
            signals=signals,
            status=case.status,
            required_signals=case.required_signals,
        )
