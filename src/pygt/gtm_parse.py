import xml.etree.ElementTree as ET
from pathlib import Path


class GTMFileParser:
    def __init__(self, path: Path | str) -> None:
        self.path = path
        self.tree = ET.parse(path)

    @property
    def root(self):
        return self.tree.getroot()

    @property
    def cases(self):
        return self.root.findall("cases/case")

    @property
    def case_count(self) -> int:
        return len(self.cases)

    def get_case_params(self) -> dict[str, float | str]:
        out = {}
        for param in self.cases[0]:
            d = param.attrib
            name = d["parameter"]
            val = d["text"]
            try:
                val = float(val)
            except ValueError:
                pass
            out[name] = val
        return out
