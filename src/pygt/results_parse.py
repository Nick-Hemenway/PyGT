import abc
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd

from pygt.cases import CaseList, ExpandedCaseList
from pygt.output_spec import ExportSpec, SampledOutputSpec
from pygt.simdata import SimulationData
from pygt.workspace import Workspace


def flatten_columns(old_cols: pd.MultiIndex, include_units: bool = True) -> list[str]:
    """
    Convert multi-index where units are second level
    to single level index with format: "col_name [units]"
    """
    if include_units:
        return [f"{tup[0]} [{tup[1]}]" for tup in old_cols]
    else:
        return [f"{tup[0]}" for tup in old_cols]


@dataclass
class FileParserBase(abc.ABC):
    split_text: str
    include_units: bool = False

    def _read(
        self, path: Path | str, name_map: dict[str, str]
    ) -> dict[int, pd.DataFrame]:
        with open(path) as f:
            text = self.sanitize(f.read())

        out = {}
        sections = re.split(self.split_text, text)
        for section_text in sections:
            case = self.parse_case_number(section_text)
            df = pd.read_csv(StringIO(section_text), skiprows=1, header=[0, 1])
            df.columns = flatten_columns(df.columns, include_units=self.include_units)
            df = df.rename(columns=name_map)
            # Add dataframe to results if it is not empty
            if df.shape[0] != 0:
                out[case - 1] = df
        return out

    @staticmethod
    def sanitize(text: str) -> str:
        return text

    @abc.abstractmethod
    def parse_case_number(self, text: str) -> int: ...


class SampledOutputParser(FileParserBase):
    def __init__(self, include_units: bool = False):
        super().__init__(
            split_text=r"\n+\s* CASE",
            include_units=include_units,
        )

    def _read(
        self,
        spec: SampledOutputSpec,
        workspace: Workspace,
    ) -> dict[int, pd.DataFrame]:
        data_path = workspace.path / spec.fname
        df_dict = super()._read(data_path, name_map=spec.name_map)
        out = {}
        for key, df in df_dict.items():
            df.index.name = spec.index_name
            out[key] = df
        return out

    def parse_case_number(self, text: str) -> int:
        # find end of first line so we don't have to split
        # entire text in next step
        idx = text.find("\n")
        # split first line on spaces and grab case at specified location
        return int(text[0:idx].split()[1])

    @staticmethod
    def sanitize(text: str) -> str:
        return text.replace("|", "").strip()[4::]

    def read(
        self,
        spec: SampledOutputSpec,
        workspace: Workspace,
        caselist: CaseList | ExpandedCaseList,
    ) -> SimulationData:
        data = self._read(spec=spec, workspace=workspace)
        return SimulationData(
            sim_data=data,
            caselist=caselist,
            required_signals=list(spec.name_map.values()),
        )


class ExportedOutputParser(FileParserBase):
    def __init__(self, include_units: bool = False):
        super().__init__(
            # Split sections based on 2 or more consecutive newline chars
            # with any amount of additional white space between
            split_text=r"\n\s*\n+",
            include_units=include_units,
        )

    def _read(
        self,
        spec: ExportSpec,
        workspace: Workspace,
    ) -> dict[int, pd.DataFrame]:
        data_path = workspace.export_output_path(spec)
        df_dict = super()._read(data_path, name_map=spec.name_map)
        out = {}
        for key, df in df_dict.items():
            df = df.set_index(df.columns[0])
            df.index.name = spec.index_name
            out[key] = df
        return out

    @staticmethod
    def sanitize(text: str) -> str:
        return text.strip()

    def parse_case_number(self, text: str) -> int:
        # find end of first line so we don't have to split
        # entire text in next step
        idx = text.find("\n")
        # split first line on spaces and grab case at specified location
        return int(text[0:idx].split()[2])

    def read(
        self,
        spec: ExportSpec,
        workspace: Workspace,
        caselist: CaseList | ExpandedCaseList,
    ) -> SimulationData:
        data = self._read(spec=spec, workspace=workspace)
        return SimulationData(
            sim_data=data,
            caselist=caselist,
            required_signals=list(spec.name_map.values()),
        )
