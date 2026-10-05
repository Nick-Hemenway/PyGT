import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import optwrap as opt
import optwrap.resolvers as rs
from optwrap import Context, Dependencies
from optwrap.nodes import (
    ResolvedDependencies,
    resolve_dependencies,
    sanitize_dependencies,
)
from optwrap.resolvers import Resolvable

from ..exceptions import MissingSignalError
from ..simdata import CaseData

logger = logging.getLogger(__name__)

type ContextResolveable = Resolvable[Context, Any]


@dataclass
class MultiCaseEvaluator:
    func: Callable[..., float]
    case_dependencies: (
        Dependencies[CaseData]
        | Resolvable[CaseData, Any]
        | Sequence[Resolvable[CaseData, Any]]
        | Mapping[str, Resolvable[CaseData, Any]]
        | None
    ) = None
    context_dependencies: (
        Dependencies[Context]
        | Resolvable[Context, Sequence]
        | Sequence[Resolvable[Context, Sequence]]
        | Mapping[str, Resolvable[Context, Sequence]]
        | None
    ) = None
    op_points_count: rs.Resolvable[opt.Context, int] | None = None

    def __post_init__(self):
        self.case_dependencies: Dependencies[CaseData] = sanitize_dependencies(
            self.case_dependencies
        )
        self.context_dependencies: Dependencies[Context] = sanitize_dependencies(
            self.context_dependencies
        )
        if (len(self.case_dependencies.args) > 0) and (
            len(self.context_dependencies) > 0
        ):
            raise ValueError(
                "Positional arguments cannot be supplied for both case_dependencies "
                "and context dependencies since the ordering of parameters becomes ambiguous"
            )

    def __call__(self, case_data: list[CaseData], context: opt.Context | None = None):
        results = []
        resolved_ctx_deps = self._context_dependencies(
            case_count=len(case_data), context=context
        )
        for case, ctx_deps in zip(case_data, resolved_ctx_deps):
            try:
                resolved_case_deps = resolve_dependencies(case, self.case_dependencies)
            except MissingSignalError as e:
                logger.warning("%s", e)
                results.append(None)
            else:
                args = (*resolved_case_deps.args, *ctx_deps.args)
                kwargs = resolved_case_deps.kwargs | ctx_deps.kwargs
                results.append(self.func(*args, **kwargs))
        results = np.asarray(results, dtype=float)
        if self.op_points_count is not None:
            count = rs.as_resolver(self.op_points_count)
            results = results.reshape(-1, count.resolve(context))
        return results

    def _context_dependencies(
        self, case_count: int, context: opt.Context | None = None
    ) -> list[ResolvedDependencies]:
        if not self.context_dependencies.has_dependencies:
            return [ResolvedDependencies(args=(), kwargs={}) for _ in range(case_count)]
        if (context is None) and self.context_dependencies.has_dependencies:
            raise ValueError(
                "No context provided even though context dependencies have been declared"
            )

        # Resolve the context dependencies and make sure they're all a
        # Sequence with length equal to the number of cases
        deps_ctx = resolve_dependencies(context, self.context_dependencies)
        self._verify_correct_length(deps_ctx.args, case_count)
        self._verify_correct_length(deps_ctx.kwargs.values(), case_count)
        out_deps: list[ResolvedDependencies] = []
        for i in range(case_count):
            out_deps.append(
                ResolvedDependencies(
                    args=tuple(arg[i] for arg in deps_ctx.args),
                    kwargs={key: val[i] for key, val in deps_ctx.kwargs.items()},
                )
            )
        return out_deps

    def _verify_correct_length(self, args: Sequence, case_count: int):
        for arg in args:
            try:
                count = len(arg)
            except TypeError as e:
                raise ValueError(
                    (
                        "All context attributes must have the same length as number "
                        "of cases but attribute doesn't have length attribute"
                    )
                ) from e
            else:
                if count != case_count:
                    raise ValueError(
                        (
                            "All context attributes must have the same length as number "
                            f"of cases but attribute has length of {count}"
                        )
                    )


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
