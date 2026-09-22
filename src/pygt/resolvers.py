import abc
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from optwrap.resolvers import Ref

from .simdata import CaseData


class SignalFilter(abc.ABC):
    @abc.abstractmethod
    def resolve(self, case: CaseData) -> NDArray[np.bool_]: ...


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


type Resolvable = "CaseDataResolver" | int | float | str


class CaseDataResolver(abc.ABC):
    @abc.abstractmethod
    def resolve(self, case: CaseData) -> float | NDArray: ...

    def __neg__(self):
        return self.__mul__(-1)

    def __pos__(self):
        return self.__mul__(1)

    def __add__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=self, right=other, func=lambda a, b: a + b, op="+"
        )

    def __radd__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return self.__add__(other)

    def __sub__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=self, right=other, func=lambda a, b: a - b, op="-"
        )

    def __rsub__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=other, right=self, func=lambda a, b: a - b, op="-"
        )

    def __mul__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=self, right=other, func=lambda a, b: a * b, op="*"
        )

    def __rmul__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return self.__mul__(other)

    def __truediv__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=self, right=other, func=lambda a, b: a / b, op="/"
        )

    def __rtruediv__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=other, right=self, func=lambda a, b: a / b, op="/"
        )

    def __mod__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=self, right=other, func=lambda a, b: a % b, op="%"
        )

    def __rmod__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=other, right=self, func=lambda a, b: a % b, op="%"
        )

    def __floordiv__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=self, right=other, func=lambda a, b: a // b, op="//"
        )

    def __rfloordiv__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=other, right=self, func=lambda a, b: a // b, op="//"
        )

    def __pow__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=self, right=other, func=lambda a, b: a**b, op="**"
        )

    def __rpow__(self, other: Resolvable) -> "BinaryCaseDataResolver":
        return BinaryCaseDataResolver(
            left=other, right=self, func=lambda a, b: a**b, op="**"
        )


@dataclass
class Literal(CaseDataResolver):
    value: float
    func: Callable | None = None

    def resolve(self, case: CaseData) -> float:
        out = self.value
        if self.func is not None:
            return self.func(out)
        else:
            return out


@dataclass
class OpPoint(CaseDataResolver):
    name: str
    func: Callable | None = None

    def resolve(self, case: CaseData) -> float | NDArray:
        out = getattr(case, self.name)
        if self.func is not None:
            return self.func(out)
        else:
            return out


@dataclass
class DesignVar(CaseDataResolver):
    name: str
    func: Callable | None = None

    def resolve(self, case: CaseData) -> float | NDArray:
        out = getattr(case, self.name)
        if self.func is not None:
            return self.func(out)
        else:
            return out


@dataclass
class Signal(CaseDataResolver):
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


def as_resolver(resolver: Resolvable) -> CaseDataResolver:
    if isinstance(resolver, CaseDataResolver):
        return resolver
    elif isinstance(resolver, str):
        return Ref(resolver)
    else:
        return Literal(resolver)


class BinaryCaseDataResolver(CaseDataResolver):
    def __init__(
        self,
        left: CaseDataResolver,
        right: CaseDataResolver,
        func: Callable,
        op: str,
    ):
        self.left = as_resolver(left)
        self.right = as_resolver(right)
        self.func = func
        self.op = op

    def resolve(self, case: CaseData) -> float | NDArray:
        return self.func(self.left.resolve(case), self.right.resolve(case))


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
class BetweenSymmetric(SignalFilter):
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
