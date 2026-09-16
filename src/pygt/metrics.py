import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import optwrap.resolvers as rs

from .exceptions import MissingSignalError
from .simdata import CaseData

logger = logging.getLogger(__name__)


@dataclass
class SignalAggregate:
    kind: str

    def __post_init__(self):
        if self.kind.lower() not in ["min", "max"]:
            raise ValueError('Invalid value for "agg" attribute')

    def __call__(
        self,
        index: np.ndarray,
        signal: np.ndarray,
        duration: float | None = None,
    ) -> np.ndarray:
        if duration is not None:
            mask = (index >= (-duration / 2)) & (index <= (duration / 2))
            signal = signal[mask]

        if self.kind.lower() == "min":
            return np.min(signal, axis=0)
        elif self.kind.lower() == "max":
            return np.max(signal, axis=0)
        else:
            raise RuntimeError("Invalid value provided for 'agg' in SimpleMetric")


@dataclass
class MultiCaseEvaluator:
    func: Callable[..., float]
    arg_map: dict[str, str | rs.Resolver]
    op_points: int | None = None

    def __call__(self, cases: list[CaseData]):
        res = []
        for case in cases:
            try:
                kwargs = rs.build_kwargs(case, self.arg_map)
            except MissingSignalError:
                res.append(None)
            else:
                res.append(self.func(**kwargs))
        res = np.asarray(res, dtype=float)
        if self.op_points is not None:
            res = res.reshape(-1, self.op_points)
        return res
