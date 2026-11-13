from pathlib import Path
from typing import Any

import jinja2
from jinja2 import Environment, meta

# ---------------------------------------------------------------------------- #
#                  Functions to Determine QDA Results Indices                  #
# ---------------------------------------------------------------------------- #


def _sort_dict_by_values(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda item: item[1]))


def build_qda_indices(qda_rpms: dict[str, float], case_rpm: float) -> dict[str, str]:
    """Build indices that need to go in export file for given rpm and qda rpms

    Parameters
    ----------
    qda_rpms : dict[str, float]
        Mapping from gt parameter names for the various qda rpms to their rpm value
    case_rpm : float
        Engine rpm for a particular case. For example, if you want to extract QDA results
        for the first case of each design, you could set this to the first rpm value
        that you're evaluating a design at.

    Returns
    -------
    dict[str, str]
        Mapping from qda rpm name to index to be inserted in export config file.
        E.g. {"qda_erpm_idle": "001", "qda_erpm_max_speed": "003"}

    Examples
    --------

    # Case 1: No Overlap of Case RPMs with QDA RPMs

    >>> qda_rpms = {"qda_erpm_idle": 650, "qda_erpm_1": 4100, "qda_erpm_max_speed": 6800}
    >>> rpms = [4200, 6600] # all case rpms
    >>> build_qda_indices(qda_rpms, rpms[0])
    >>> {"qda_erpm_idle": "001", "qda_erpm_1": "002", "qda_erpm_max_speed": "004"}


    # Case 2: Case RPM Matches One of the QDA RPMs

    >>> qda_rpms = {"qda_erpm_idle": 650, "qda_erpm_1": 4100, "qda_erpm_max_speed": 6800}
    >>> rpms = [4100, 6600] # all case rpms
    >>> build_qda_indices(qda_rpms, rpms[0])
    >>> {"qda_erpm_idle": "001", "qda_erpm_1": "002", "qda_erpm_max_speed": "003"}
    """
    # For any given case, all qda rpms are run, and the single cases rpm is used
    temp = "_tmp"
    if case_rpm not in qda_rpms.values():
        d = qda_rpms | {temp: case_rpm}
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


def get_template_variables(path: Path | str) -> tuple[str]:
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
