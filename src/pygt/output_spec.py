import logging
from dataclasses import dataclass
from pathlib import Path

from .util import sanitize_path

logger = logging.getLogger(__name__)


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
    name: str
    fname: str
    index_name: str
    name_map: dict[str, str]
