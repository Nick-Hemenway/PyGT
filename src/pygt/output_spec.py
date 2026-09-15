from dataclasses import dataclass
from pathlib import Path

from pycam.util import sanitize_path


@dataclass
class ExportSpec:
    name: str
    config_source: Path | str
    output_filename: str
    index_name: str
    name_map: dict[str, str]

    def __post_init__(self):
        self.config_source = sanitize_path(self.config_source)


@dataclass
class SampledOutputSpec:
    fname: str
    index_name: str
    name_map: dict[str, str]
