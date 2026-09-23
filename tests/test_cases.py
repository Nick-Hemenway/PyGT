import numpy as np
import optwrap as opt
import pytest

import pygt
from pygt import CaseList, ExpandedCaseList


@pytest.fixture
def caselist() -> CaseList:
    return CaseList(
        design_variables={"a": [1, 2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8, 9]},
        operating_points={"rpm": [100, 200, 300, 100, 200, 300]},
        params={"p1": 1, "p2": 5},
        op_point_size=3,
        write_params=True,
    )


@pytest.fixture
def guard_design() -> dict[str, float]:
    return {"a": 1e6, "b": 2e6, "rpm": 3e6}


@pytest.fixture
def expanded_caselist(
    caselist: CaseList, guard_design: dict[str, float]
) -> ExpandedCaseList:
    return ExpandedCaseList.from_packet_size(
        caselist=caselist,
        guard_design=guard_design,
        packet_size=4,
    )


def test_caselist_broadcasting_scalar():
    c = CaseList(design_variables={"a": [1, 2, 3], "b": 4})
    assert np.array_equal(c.b, [4, 4, 4])


def test_caselist_broadcasting_scalar_param():
    c = CaseList(
        design_variables={"a": [1, 2, 3], "b": [4, 4, 4]},
        params={"p1": 1, "p2": 2},
    )
    assert np.array_equal(c.p1, [1, 1, 1])


def test_caselist_broadcasting_arrays():
    c = CaseList(design_variables={"a": [1, 2, 3], "b": [4, 4, 4]})
    assert np.array_equal(c.b, [4, 4, 4])


def test_caselist_broadcasting_invalid():
    with pytest.raises(ValueError):
        CaseList(design_variables={"a": [1, 2, 3], "b": [4, 4]})


def test_caselist_missing_op_size():
    # Should raise a value error because op_point_size is not provided
    with pytest.raises(ValueError):
        CaseList(
            design_variables={"a": [1, 2, 3], "b": [4, 5, 6]},
            operating_points={"rpm": [100, 200, 300]},
        )


@pytest.mark.parametrize("op_point_size", [1, 2, 3, 6])
def test_caselist_valid_op_size(op_point_size: int):
    # None should raise error because size is divisible by op_point_size
    CaseList(
        design_variables={"a": [1, 2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8, 9]},
        operating_points={"rpm": [100, 200, 300, 400, 500, 600]},
        op_point_size=op_point_size,
    )


@pytest.mark.parametrize("op_point_size", [4, 5, 7, 12])
def test_caselist_invalid_op_size(op_point_size: int):
    # Should raise a value error because size is not evenly divisible by op_point_size
    with pytest.raises(ValueError):
        CaseList(
            design_variables={"a": [1, 2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8, 9]},
            operating_points={"rpm": [100, 200, 300, 400, 500, 600]},
            op_point_size=op_point_size,
        )


def test_params_not_present():
    cases = CaseList(
        design_variables={"a": [1, 2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8, 9]},
        operating_points={"rpm": [100, 200, 300, 100, 200, 300]},
        params={"p1": 1, "p2": 5},
        op_point_size=3,
        write_params=False,
    )
    assert set(cases.values.keys()) == {"a", "b", "rpm"}


def test_params_present():
    cases = CaseList(
        design_variables={"a": [1, 2, 3, 4, 5, 6], "b": [4, 5, 6, 7, 8, 9]},
        operating_points={"rpm": [100, 200, 300, 100, 200, 300]},
        params={"p1": 1, "p2": 5},
        op_point_size=3,
        write_params=True,
    )
    assert set(cases.values.keys()) == {"a", "b", "rpm", "p1", "p2"}


def test_caselist_size(caselist: CaseList):
    assert caselist.size == 6


def test_caselist_index_map(caselist: CaseList):
    assert np.array_equal(caselist.index_map, [0, 1, 2, 3, 4, 5])


def test_caselist_design_map(caselist: CaseList):
    assert np.array_equal(caselist.design_map, [0, 0, 0, 1, 1, 1])


def test_caselist_design_count(caselist: CaseList):
    assert caselist.design_count == 2


@pytest.mark.parametrize(
    "case_id, expected", [(0, {"a": 1, "b": 4}), (4, {"a": 5, "b": 8})]
)
def test_caselist_get_design_variables(
    caselist: CaseList, case_id: int, expected: dict[str, float]
):
    vars = caselist.get_design_variables(case_id=case_id)
    assert vars == expected


@pytest.mark.parametrize("case_id, expected", [(0, {"rpm": 100}), (4, {"rpm": 200})])
def test_caselist_get_operating_points(
    caselist: CaseList, case_id: int, expected: dict[str, float]
):
    ops = caselist.get_operating_point(case_id=case_id)
    assert ops == expected


@pytest.mark.parametrize(
    "case_id, expected", [(0, {"a": 1, "b": 4}), (4, {"a": 5, "b": 8})]
)
def test_expanded_caselist_get_design_variables(
    expanded_caselist: ExpandedCaseList, case_id: int, expected: dict[str, float]
):
    vars = expanded_caselist.get_design_variables(case_id=case_id)
    assert vars == expected


@pytest.mark.parametrize("case_id, expected", [(0, {"rpm": 100}), (4, {"rpm": 200})])
def test_expanded_caselist_get_operating_points(
    expanded_caselist: ExpandedCaseList, case_id: int, expected: dict[str, float]
):
    ops = expanded_caselist.get_operating_point(case_id=case_id)
    assert ops == expected


def test_expanded_caselist_size(expanded_caselist: ExpandedCaseList):
    assert expanded_caselist.size == 8


def test_guard_design_sanitize_all_param_missing(
    expanded_caselist: ExpandedCaseList, guard_design: dict[str, float]
):
    expected = guard_design.copy()
    expected["p1"] = 1
    expected["p2"] = 5
    assert expanded_caselist.sanitized_guard_design == expected


def test_guard_design_sanitize_partial_param(
    caselist: CaseList, guard_design: dict[str, float]
):
    guard_design["p1"] = 1e6
    expected = guard_design.copy()
    expected["p2"] = 5
    expanded = ExpandedCaseList.from_packet_size(
        caselist=caselist, guard_design=guard_design, packet_size=4
    )
    assert expanded.sanitized_guard_design == expected


def test_guard_design_sanitize_all_param(
    caselist: CaseList, guard_design: dict[str, float]
):
    guard_design["p1"] = 1e6
    guard_design["p2"] = 1e6
    expanded = ExpandedCaseList.from_packet_size(
        caselist=caselist, guard_design=guard_design, packet_size=4
    )
    assert expanded.sanitized_guard_design == guard_design


def test_expanded_caselist_design_count(expanded_caselist: ExpandedCaseList):
    assert expanded_caselist.design_count == 2


def test_expanded_caselist_index_map(expanded_caselist: ExpandedCaseList):
    assert np.array_equal(expanded_caselist.index_map, [-1, 0, 1, 2, -1, 3, 4, 5])


def test_expanded_caselist_design_map(expanded_caselist: ExpandedCaseList):
    assert np.array_equal(expanded_caselist.design_map, [-1, 0, 0, 0, -1, 1, 1, 1])


def test_build_guard_design():
    # Build out Registry
    reg = opt.Registry()
    reg.add("a", lower=0, upper=10, default=1)
    reg.add("b", lower=0, upper=100, default=2)
    reg.add_operating_point_param(opt.OperatingPointParameter("rpm"))
    reg.add_param(opt.Parameter("p1", 1000))

    # Construcst Operating Points
    ops = opt.OperatingPoints(rpm=[100, 200, 300])

    guard = pygt.build_guard_design(registry=reg, operating_points=ops)
    assert guard == {"a": 1, "b": 2, "rpm": 100, "p1": 1000}


@pytest.mark.parametrize("licenses", [4, 5])
def test_invalid_license_count(
    caselist: CaseList, guard_design: dict[str, float], licenses: int
):
    with pytest.raises(ValueError):
        ExpandedCaseList.from_license_count(
            caselist=caselist, guard_design=guard_design, licenses=licenses
        )


@pytest.mark.parametrize("licenses, expected_size", [(1, 7), (2, 8), (3, 9), (6, 12)])
def test_case_size_per_license_count(
    caselist: CaseList,
    guard_design: dict[str, float],
    licenses: int,
    expected_size: int,
):
    expanded = ExpandedCaseList.from_license_count(
        caselist=caselist, guard_design=guard_design, licenses=licenses
    )
    assert expanded.size == expected_size


def test_expanded_caselist_index_map_from_licenses(
    caselist: CaseList,
    guard_design: dict[str, float],
):
    expanded = ExpandedCaseList.from_license_count(
        caselist=caselist,
        guard_design=guard_design,
        licenses=3,
    )
    assert np.array_equal(expanded.index_map, [-1, 0, 1, -1, 2, 3, -1, 4, 5])


def test_expanded_caselist_design_map_from_licenses(
    caselist: CaseList,
    guard_design: dict[str, float],
):
    expanded = ExpandedCaseList.from_license_count(
        caselist=caselist,
        guard_design=guard_design,
        licenses=3,
    )
    assert np.array_equal(expanded.design_map, [-1, 0, 0, -1, 0, 1, -1, 1, 1])
