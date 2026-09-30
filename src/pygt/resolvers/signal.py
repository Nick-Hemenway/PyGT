from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from optwrap.resolvers import AbstractResolver, Resolvable, Resolver, as_resolver

from ..simdata import CaseData

type SignalMask = Resolver[CaseData, NDArray[np.bool_]]


@dataclass
class Signal(AbstractResolver[CaseData, NDArray]):
    name: str
    where: SignalMask | list[SignalMask] | None = None

    def resolve(self, case: CaseData) -> NDArray:
        sig = case.get(self.name)
        mask = build_signal_mask(case, masks=self.where)
        if mask is None:
            return sig
        else:
            return sig[mask]


@dataclass
class Between(AbstractResolver[CaseData, NDArray[np.bool_]]):
    name: Signal
    lower: Resolvable[CaseData, float]
    upper: Resolvable[CaseData, float]

    def __post_init__(self):
        if not isinstance(self.name, Signal):
            raise ValueError(f"{self.name} must be a Signal resolver")
        self.lower = as_resolver(self.lower)
        self.upper = as_resolver(self.upper)

    def resolve(self, case: CaseData) -> NDArray[np.bool_]:
        lower = self.lower.resolve(case)
        upper = self.upper.resolve(case)
        sig = self.name.resolve(case)
        mask = (sig >= lower) & (sig <= upper)
        return mask


@dataclass
class Window(AbstractResolver[CaseData, NDArray[np.bool_]]):
    name: Signal
    duration: Resolvable[CaseData, float]

    def __post_init__(self):
        if not isinstance(self.name, Signal):
            raise ValueError(f"{self.name} must be a Signal resolver")
        self.duration = as_resolver(self.duration)

    def resolve(self, case: CaseData) -> NDArray[np.bool_]:
        duration = self.duration.resolve(case)
        lower = -duration / 2
        upper = duration / 2
        sig = self.name.resolve(case)
        mask = (sig >= lower) & (sig <= upper)
        return mask


def build_signal_mask(
    case: CaseData, masks: list[SignalMask] | None = None
) -> NDArray[np.bool_]:
    if masks is None:
        return None
    if not isinstance(masks, (list, tuple)):
        masks = [masks]
    masks = [mask.resolve(case) for mask in masks]
    mask = np.logical_and.reduce(masks)
    if not np.any(mask):
        raise ValueError(
            "Signal filter results in no remaining values (mask is all False)"
        )
    return mask
