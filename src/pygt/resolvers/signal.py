import abc
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from optwrap.resolvers import Resolver

from ..simdata import CaseData
from .base import CaseDataResolver


class SignalFilter(abc.ABC):
    @abc.abstractmethod
    def resolve(self, case: CaseData) -> NDArray[np.bool_]: ...


@dataclass
class Signal(Resolver):
    name: str
    where: SignalFilter | list[SignalFilter] | None = None
    func: Callable | None = None

    def resolve(self, case: CaseData) -> NDArray:
        sig = case.get(self.name)
        mask = build_filter_mask(case, filters=self.where)
        if mask is None:
            out = sig
        else:
            out = sig[mask]
        if self.func is not None:
            return self.func(out)
        else:
            return out


@dataclass
class Between(SignalFilter):
    name: Signal
    lower: CaseDataResolver | float | int
    upper: CaseDataResolver | float | int

    def __post_init__(self):
        if not isinstance(self.name, Signal):
            raise ValueError(f"{self.name} must be a Signal resolver")

    def _lower(self, case: CaseData) -> float:
        try:
            return self.lower.resolve(case)
        except AttributeError:
            return self.lower

    def _upper(self, case: CaseData) -> float:
        try:
            return self.upper.resolve(case)
        except AttributeError:
            return self.upper

    def resolve(self, case: CaseData) -> NDArray[np.bool_]:
        sig = self.name.resolve(case)
        mask = (sig >= self._lower(case)) & (sig <= self._upper(case))
        return mask


@dataclass
class Window(SignalFilter):
    name: Signal
    duration: CaseDataResolver

    def __post_init__(self):
        if not isinstance(self.name, Signal):
            raise ValueError(f"{self.name} must be a Signal resolver")

    def _value(self, case: CaseData) -> float:
        try:
            return self.duration.resolve(case)
        except AttributeError:
            return self.duration

    def _lower(self, case: CaseData) -> float:
        return -self._value(case) / 2

    def _upper(self, case: CaseData) -> float:
        return self._value(case) / 2

    def resolve(self, case: CaseData) -> NDArray[np.bool_]:
        sig = self.name.resolve(case)
        mask = (sig >= self._lower(case)) & (sig <= self._upper(case))
        return mask


def build_filter_mask(
    case: CaseData, filters: list[SignalFilter] | None = None
) -> NDArray[np.bool_]:
    if filters is None:
        return None
    if not isinstance(filters, (list, tuple)):
        filters = [filters]
    masks = [filt.resolve(case) for filt in filters]
    mask = np.logical_and.reduce(masks)
    if not np.any(mask):
        raise ValueError(
            "Signal filter results in no remaining values (mask is all False)"
        )
    return mask
