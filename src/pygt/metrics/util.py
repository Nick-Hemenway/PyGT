import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import optwrap.resolvers as rs

from ..exceptions import MissingSignalError
from ..simdata import CaseData

logger = logging.getLogger(__name__)


@dataclass
class MultiCaseEvaluator:
    func: Callable[..., float]
    arg_map: dict[str, str | rs.Resolver]
    op_points_count: int | None = None

    def __call__(self, cases: list[CaseData]):
        res = []
        for case in cases:
            try:
                kwargs = rs.build_kwargs(case, self.arg_map)
            except MissingSignalError as e:
                logger.warning("%s", e)
                res.append(None)
            else:
                res.append(self.func(**kwargs))
        res = np.asarray(res, dtype=float)
        if self.op_points_count is not None:
            res = res.reshape(-1, self.op_points_count)
        return res
