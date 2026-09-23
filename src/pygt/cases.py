import logging
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from optwrap import Context, OperatingPoints, Registry

logger = logging.getLogger(__name__)


def index_map(cases: int, packet_size: int) -> NDArray:
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
    return np.array(mapping)


@dataclass
class CaseList:
    context: Context

    @property
    def _values(self):
        out = {}
        for key in self.context.names(vars=True, params=True):
            out[key] = getattr(self.context, key)
        if self.context.operating_points is not None:
            for key in self.context.names(op_point_params=True):
                out[key] = getattr(self.context, key)
        return out

    @cached_property
    def values(self):
        return {key: val.flatten(order="c") for key, val in self._values.items()}

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
        if self.context.operating_points is None:
            return np.arange(self.size)
        npoints = self.context.operating_points.size
        return np.repeat(np.arange(int(self.size / npoints)), npoints).astype(int)

    @property
    def design_count(self) -> int:
        npoints = self.context.operating_points.size
        return int(self.size / npoints)

    def get_design_variables(self, case_id: int):
        params = {}
        for name in self.context.names(vars=True):
            params[name] = self.caselist.values[name][case_id]
        return params

    def get_operating_point(self, case_id: int):
        if self.context.operating_points is None:
            return {}
        params = {}
        for name in self.context.names(op_point_params=True):
            params[name] = self.caselist.values[name][case_id]
        return params


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


@dataclass
class ExpandedCaseList:
    caselist: CaseList
    packet_size: int
    guard_design: dict[str, float]

    @property
    def real_mask(self) -> NDArray:
        return self.index_map >= 0

    @property
    def guard_mask(self) -> NDArray:
        return self.index_map < 0

    @cached_property
    def values(self):
        out = {}
        for key, arr in self.caselist.values.items():
            expanded = np.empty(len(self.index_map))
            expanded[self.real_mask] = arr[self.index_map[self.real_mask]]
            expanded[self.guard_mask] = self.guard_design[key]
            out[key] = expanded
        return out

    def __getitem__(self, key: str):
        return self.values[key]

    @property
    def size(self):
        return len(next(iter(self.values.values())))

    @cached_property
    def index_map(self) -> NDArray:
        return index_map(cases=self.caselist.size, packet_size=self.packet_size).astype(
            int
        )

    @cached_property
    def design_map(self) -> NDArray:
        out = np.empty(len(self.index_map))
        out[self.guard_mask] = -1
        out[self.real_mask] = self.caselist.design_map
        return out.astype(int)

    @property
    def design_count(self) -> int:
        return self.caselist.design_count

    def get_design_variables(self, case_id: int):
        return self.caselist.get_design_variables(case_id)

    def get_operating_point(self, case_id: int):
        return self.caselist.get_operating_point(case_id)


# NOTE: This functions are implemented but not used.
# They might be helpful if we decide to support parallel runs
# from directly within python
def prepend_design(caselist: CaseList, design: dict[str, float]) -> CaseList:
    new_data = {}
    for name, values in caselist.parameters.items():
        new_data[name] = np.concatenate([[design[name]], values])
    return CaseList(new_data)


def packetize(caselist: CaseList, packet_size: int):
    n = caselist.size
    for start in range(0, n, packet_size):
        end = min(start + packet_size, n)
        yield caselist.slice(start, end)


def prepare_packets(caselist, packet_size, guard_design):
    for packet in packetize(caselist, packet_size):
        yield prepend_design(packet, guard_design)


@dataclass
class Packet:
    caselist: CaseList
    original_indices: NDArray
