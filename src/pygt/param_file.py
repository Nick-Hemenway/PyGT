import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .cases import CaseList, ExpandedCaseList
from .exceptions import InconsistentParametersError
from .workspace import Workspace

logger = logging.getLogger(__name__)


class DesignWriter:
    @staticmethod
    def cases_to_text(caselist: CaseList | ExpandedCaseList) -> str:
        lines = []
        for name, arr in caselist.values.items():
            lines.append(f"{name} > {' '.join([str(item) for item in arr])} <")
        return "\n".join(lines)

    @staticmethod
    def cases_to_metadata_text(caselist: CaseList | ExpandedCaseList) -> str:
        return json.dumps(
            {
                "design_map": np.asarray(caselist.design_map).tolist(),
                "index_map": np.asarray(caselist.index_map).tolist(),
            },
            indent=2,
        )

    def write_file(
        self,
        caselist: CaseList | ExpandedCaseList,
        workspace: Workspace,
        overwrite: bool = False,
    ):
        # Convert caselist to required text for files
        text = self.cases_to_text(caselist)
        metadata_text = self.cases_to_metadata_text(caselist)

        # Ensure that if param file already exists, cases are consistent with it
        if workspace.param_file_path.exists() and not overwrite:
            c1 = text == workspace.param_file_path.read_text()
            c2 = json.loads(metadata_text) == json.loads(
                workspace.param_metadata_file_path.read_text()
            )
            if not (c1 and c2):
                raise InconsistentParametersError
            return None

        workspace.param_file_path.write_text(text)
        workspace.param_metadata_file_path.write_text(metadata_text)

        if (
            workspace.param_file_path.exists()
            and workspace.param_metadata_file_path.exists()
        ):
            workspace.mark_ready()


def read_param_file(path: Path | str) -> list[dict]:
    out = {}
    with open(path) as f:
        for line in f:
            name, nums = line.split(">")
            out[name.strip()] = [float(n) for n in nums[0:-2].split()]

    df = pd.DataFrame(out)
    return df.to_dict(orient="records")
