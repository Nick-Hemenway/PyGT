import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from optwrap import Context, OperatingPoints, Registry

logger = logging.getLogger(__name__)


@dataclass
class CaseList:
    design_variables: dict[str, NDArray]
    params: dict[str, NDArray] | None = None
    operating_points: dict[str, NDArray] | None = None
    op_point_size: int | None = None
    write_params: bool = True

    def __post_init__(self):
        if (self.operating_points is not None) and (self.op_point_size is None):
            raise ValueError(
                '"op_point_size" must be provided if "operating_point_params" is not None'
            )

        # Ensure all arrays are at least 1D
        self.design_variables = {
            key: np.atleast_1d(val) for key, val in self.design_variables.items()
        }
        if self.params is not None:
            self.params = {key: np.atleast_1d(val) for key, val in self.params.items()}
        if self.operating_points is not None:
            self.operating_points = {
                key: np.atleast_1d(val) for key, val in self.operating_points.items()
            }

        # Ensure that size of arrays provided is divisible by number of operating points
        if self.operating_points is not None:
            design_count = self.size / self.op_point_size
            if int(design_count) != design_count:
                raise ValueError(
                    "Total case count is not divisible by number of operating points"
                )

    @cached_property
    def values(self) -> dict[str, np.ndarray]:
        # Combine and broadcast all arrays
        params = self.params
        if params is None:
            params = {}
        op_points = self.operating_points
        if op_points is None:
            op_points = {}

        if self.write_params:
            combined = self.design_variables | params | op_points
        else:
            combined = self.design_variables | op_points

        out = {}
        try:
            for key, arr in zip(
                combined.keys(), np.broadcast_arrays(*combined.values())
            ):
                out[key] = arr
        except ValueError as e:
            raise ValueError(
                "All operating point columns must have the same length."
            ) from e
        else:
            return out

    @property
    def size(self):
        return len(next(iter(self.values.values())))

    def __getitem__(self, key: str):
        return self.values[key]

    @cached_property
    def index_map(self) -> NDArray:
        return np.arange(self.size).astype(int)

    @cached_property
    def design_map(self) -> NDArray:
        if self.operating_points is None:
            return np.arange(self.size)
        return np.repeat(np.arange(self.design_count), self.op_point_size).astype(int)

    @property
    def design_count(self) -> int:
        if self.operating_points is None:
            return self.size
        else:
            return int(self.size / self.op_point_size)

    def get_design_variables(self, case_id: int):
        params = {}
        for name in self.context.names(vars=True):
            params[name] = self.caselist.values[name][case_id]
        return params

    def get_operating_point(self, case_id: int):
        if self.operating_points is None:
            return {}
        params = {}
        for name in self.context.names(op_point_params=True):
            params[name] = self.caselist.values[name][case_id]
        return params


@dataclass
class ExpandedCaseList:
    caselist: CaseList
    guard_design: dict[str, float]
    index_map: NDArray[np.int_]

    def __post_init__(self):
        self.index_map = np.asarray(self.index_map)

    @property
    def _real_mask(self) -> NDArray:
        return self.index_map >= 0

    @property
    def _guard_mask(self) -> NDArray:
        return self.index_map < 0

    @cached_property
    def values(self):
        out = {}
        for key, arr in self.caselist.values.items():
            expanded = np.empty(len(self.index_map))
            expanded[self._real_mask] = arr[self.index_map[self._real_mask]]
            expanded[self._guard_mask] = self.guard_design[key]
            out[key] = expanded
        return out

    def __getitem__(self, key: str):
        return self.values[key]

    @property
    def size(self):
        return len(next(iter(self.values.values())))

    @classmethod
    def from_packet_size(
        cls, caselist: CaseList, guard_design: dict[str, float], packet_size: int
    ) -> Self:
        return cls(
            caselist=caselist,
            guard_design=guard_design,
            index_map=build_index_map(cases=caselist.size, packet_size=packet_size),
        )

    @cached_property
    def design_map(self) -> NDArray:
        out = np.empty(len(self.index_map))
        out[self._guard_mask] = -1
        out[self._real_mask] = self.caselist.design_map
        return out.astype(int)

    @property
    def design_count(self) -> int:
        return self.caselist.design_count

    def get_design_variables(self, case_id: int):
        return self.caselist.get_design_variables(case_id)

    def get_operating_point(self, case_id: int):
        return self.caselist.get_operating_point(case_id)


def build_caselist(
    context: Context, caselist_kwargs: dict[str, Any] | None = None
) -> CaseList:
    vars = {}
    for key in context.names(vars=True):
        vars[key] = np.asarray(getattr(context, key)).flatten(order="c")

    params = {}
    for key in context.names(params=True):
        params[key] = np.asarray(getattr(context, key)).flatten(order="c")
    if len(params) == 0:
        params = None

    op_points = None
    op_point_size = None
    if context.operating_points is not None:
        op_point_size = context.operating_points.size
        op_points = {}
        for key in context.names(op_point_params=True):
            op_points[key] = getattr(context, key)

    return CaseList(
        design_variables=vars,
        params=params,
        operating_points=op_points,
        op_point_size=op_point_size,
        **caselist_kwargs,
    )


def build_index_map(cases: int, packet_size: int) -> NDArray:
    """
    Guard designs correspond to -1
    Original/real design indices are stored in other slots
    [-1, 0, 1, 2, -1, 3, 4, 5, -1]
    """
    mapping = []
    design_idx = 0
    while design_idx < cases:
        mapping.append(-1)  # guard
        end = min(design_idx + packet_size - 1, cases)
        mapping.extend(range(design_idx, end))
        design_idx = end
    return np.array(mapping).astype(int)


def build_guard_design(
    registry: Registry,
    operating_points: OperatingPoints | None = None,
) -> dict[str, float]:
    # Create default design
    x0 = registry.pack()
    ctx = registry.context_physical(x0, operating_points=operating_points)
    guard_design = {}
    for name in ctx.names(vars=True, params=True):
        guard_design[name] = getattr(ctx, name).flatten()[0]
    if ctx.operating_points is not None:
        for name in ctx.names(op_point_params=True):
            guard_design[name] = getattr(ctx, name).flatten()[0]
    return guard_design
