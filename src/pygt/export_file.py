from pathlib import Path
from typing import Any

import jinja2
from jinja2 import Environment, meta

# ---------------------------------------------------------------------------- #
#                  Functions to Determine QDA Results Indices                  #
# ---------------------------------------------------------------------------- #


def _sort_dict_by_values(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda item: item[1]))


def build_indices(qda_rpms: dict[str, float], rpm: float) -> dict[str, str]:
    # For any given case, all qda rpms are run, and the single cases rpm is used
    temp = "_tmp"
    if rpm not in qda_rpms.values():
        d = qda_rpms | {temp: rpm}
    else:
        d = qda_rpms.copy()
    d = _sort_dict_by_values(d)
    d = {key: f"{i:03d}" for i, key in enumerate(d, start=1)}
    if temp in d:
        _ = d.pop(temp)
    return d


# ---------------------------------------------------------------------------- #
#                            Jinja Related Functions                           #
# ---------------------------------------------------------------------------- #


def get_variables(path: Path | str) -> tuple[str]:
    env = Environment()
    text = Path(path).read_text()
    ast = env.parse(text)
    return tuple(meta.find_undeclared_variables(ast))


def populate_jinja_template(
    source: Path | str,
    context: dict[str, Any],
) -> str:
    path = Path(source)
    env = Environment(
        loader=jinja2.FileSystemLoader(path.parent),
        undefined=jinja2.StrictUndefined,
    )
    temp = env.get_template(path.name)
    return temp.render(context)
