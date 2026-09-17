import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def signal_min(signal: NDArray) -> float:
    return np.min(signal)


def signal_max(signal: NDArray) -> float:
    return np.max(signal)
