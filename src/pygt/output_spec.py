import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import sanitize_path

logger = logging.getLogger(__name__)


@dataclass
class FileSource:
    source: Path | str

    def __post_init__(self):
        self.source = sanitize_path(self.source)


@dataclass
class TemplateSource:
    source: Path | str
    context: dict[str, Any]

    def __post_init__(self):
        self.source = sanitize_path(self.source)


@dataclass
class ExportSpec:
    name: str
    source: FileSource | TemplateSource
    output_filename: str
    index_name: str
    name_map: dict[str, str]


@dataclass
class SampledOutputSpec:
    name: str
    fname: str
    index_name: str
    name_map: dict[str, str]
