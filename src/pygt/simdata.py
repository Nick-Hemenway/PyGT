from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import numpy as np
import pandas as pd

from pycam.cases import CaseList, ExpandedCaseList
from pycam.exceptions import MissingSignalError


class CaseStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class Signal:
    name: str

    def resolve(self, case: "CaseData") -> np.ndarray:
        return case.get(self.name)


@dataclass
class CaseData:
    case_id: int
    design_id: int
    design_variables: dict[str, float | int | np.ndarray]
    operating_point: dict[str, float | int | np.ndarray]
    index_col: str
    signals: dict[str, np.ndarray]
    status: CaseStatus
    required_signals: list[str]

    def get(self, signal: str) -> np.ndarray:
        if signal not in (self.required_signals + [self.index_col]):
            raise ValueError(
                f"Signal {signal} is not an expected signal. Expected one of {self.required_signals}"
            )
        out = self.signals.get(signal, None)
        if out is None:
            raise MissingSignalError(signal)
        return out

    def __getattr__(self, name: str) -> np.ndarray:
        return self.resolve(name)

    def resolve(self, name: str) -> np.ndarray:
        # Design Variable?
        if name in self.design_variables:
            out = self.design_variables[name]
        # Operating Point?
        elif name in self.operating_point:
            out = self.operating_point[name]
        else:
            raise AttributeError(name)
        return np.asarray(out)

    @property
    def df(self):
        return pd.DataFrame(self.signals).set_index(self.index_col)

    @classmethod
    def from_dataframe(
        cls,
        case_id: int,
        design_id: int,
        required_signals: list[str],
        design_variables: dict[str, float],
        operating_point: dict[str, float],
        df: pd.DataFrame | None = None,
    ) -> "CaseData":
        if df is None:
            signals = {}
            index_col = None
        else:
            signals = {col: df[col].to_numpy().squeeze() for col in df}
            signals[df.index.name] = df.index.to_numpy()
            index_col = df.index.name

        # Determine if case is succesful based on presence of all required signals
        mask = (signal in signals for signal in required_signals)
        if all(mask):
            status = CaseStatus.SUCCESS
        elif sum(mask) == 0:
            status = CaseStatus.FAILED
        else:
            status = CaseStatus.PARTIAL

        return cls(
            case_id=int(case_id),
            design_id=int(design_id),
            design_variables=design_variables,
            operating_point=operating_point,
            index_col=index_col,
            signals=signals,
            status=status,
            required_signals=required_signals,
        )


@dataclass
class DesignData:
    design_id: int
    cases: list[CaseData]

    def get(self, signal: str) -> np.ndarray:
        return np.stack([case.get(signal) for case in self.cases], axis=-1)

    @property
    def params(self):
        return self.cases[0].design_variables

    @property
    def operating_points(self):
        out = {}
        for case in self.cases:
            for key, val in case.operating_point.items():
                out.setdefault(key, []).append(val)
        return {key: np.asarray(val) for key, val in out.items()}


@dataclass
class SimulationData:
    sim_data: dict[str, pd.DataFrame]
    caselist: CaseList | ExpandedCaseList
    required_signals: list[str]

    def __post_init__(self):
        if not isinstance(self.caselist, (CaseList, ExpandedCaseList)):
            raise ValueError("caselist must be either CaseList or ExpandedCaseList")

    @cached_property
    def case_data(self) -> list[CaseData]:
        out = []
        for idx, (case_id, design_id) in enumerate(
            zip(self.caselist.index_map, self.caselist.design_map)
        ):
            if case_id == -1:
                continue

            df = self.sim_data.get(idx, None)
            out.append(
                CaseData.from_dataframe(
                    case_id=case_id,
                    design_id=design_id,
                    design_variables=self._get_design_variables(idx),
                    operating_point=self._get_operating_point(idx),
                    df=df,
                    required_signals=self.required_signals,
                )
            )
        return out

    def _get_design_variables(self, idx: int):
        if isinstance(self.caselist, CaseList):
            ctx = self.caselist.context
        else:
            ctx = self.caselist.caselist.context
        params = {}
        for name in ctx.names(vars=True, operating_point_params=False, nodes=False):
            params[name] = self.caselist.values[name][idx]
        return params

    def _get_operating_point(self, idx: int):
        if isinstance(self.caselist, CaseList):
            ctx = self.caselist.context
        else:
            ctx = self.caselist.caselist.context
        params = {}
        for name in ctx.names(vars=False, operating_point_params=True, nodes=False):
            params[name] = self.caselist.values[name][idx]
        return params

    def get_design(self, design_id: int) -> DesignData:
        d = DesignData(
            design_id=design_id,
            cases=[case for case in self.case_data if case.design_id == design_id],
        )
        return d

    @property
    def design_count(self) -> int:
        return self.caselist.design_count

    @cached_property
    def design_data(self) -> list[DesignData]:
        return [self.get_design(id_) for id_ in range(self.design_count)]
