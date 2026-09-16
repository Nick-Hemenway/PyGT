import logging
from dataclasses import dataclass

import numpy as np

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
