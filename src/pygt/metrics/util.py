import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import optwrap as opt
import optwrap.resolvers as rs

from ..exceptions import MissingSignalError
from ..simdata import CaseData

logger = logging.getLogger(__name__)


@dataclass
class MultiCaseEvaluator:
    func: Callable[..., float]
    case_arg_map: dict[str, rs.Resolvable[CaseData, Any]]
    context_arg_map: dict[str, rs.Resolvable[opt.Context, Any]] | None = None
    op_points_count: rs.Resolvable[opt.Context, int] | None = None

    def __call__(self, case_data: list[CaseData], context: opt.Context | None = None):
        res = []
        ctx_kwargs_list = self._context_kwargs_list(
            case_count=len(case_data), context=context
        )
        for case, ctx_kwargs in zip(case_data, ctx_kwargs_list):
            try:
                kwargs = rs.build_kwargs(case, self.case_arg_map)
                kwargs = kwargs | ctx_kwargs
            except MissingSignalError as e:
                logger.warning("%s", e)
                res.append(None)
            else:
                res.append(self.func(**kwargs))
        res = np.asarray(res, dtype=float)
        if self.op_points_count is not None:
            count = rs.as_resolver(self.op_points_count)
            res = res.reshape(-1, count.resolve(context))
        return res

    def _context_kwargs_list(
        self, case_count: int, context: opt.Context | None = None
    ) -> list[dict]:
        if (context is None) or (self.context_arg_map is None):
            return [{} for _ in range(case_count)]

        kwargs = rs.build_kwargs(context, arg_map=self.context_arg_map)

        # Ensure all arguments have the correct length
        for key, val in kwargs.items():
            try:
                count = len(val)
            except TypeError as e:
                raise ValueError(
                    (
                        "All attributes must have the same length as number "
                        f"of cases but {key} doesn't have length attribute"
                    )
                ) from e
            else:
                if count != case_count:
                    raise ValueError(
                        (
                            "All attributes must have the same length as number "
                            f"of cases but {key} has length of {count}"
                        )
                    )
        kwargs_list = []
        for i in range(case_count):
            kwargs_list.append({key: val[i] for key, val in kwargs.items()})
        return kwargs_list


def qda_op_point_index(rpms: list[float], qda_rpms: list[float]) -> int:
    idx = None
    for i, rpm in enumerate(rpms):
        if rpm in qda_rpms:
            idx = i
            break
    if idx is None:
        raise ValueError(
            "No rpm matches any of the qda rpms meaning the QDA results indices will be wrong!"
        )
    return idx
