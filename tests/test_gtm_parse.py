from pathlib import Path

import pytest

from pygt import GTMFileParser


@pytest.fixture
def parser() -> GTMFileParser:
    return GTMFileParser(path=Path(__file__).parent / "assets/model_01.gtm")


def test_case_count(parser: GTMFileParser):
    assert parser.case_count == 1


def test_parameter_extraction(parser: GTMFileParser):
    params = parser.get_case_params()
    assert params["start_of_cycle"] == -60
    assert params["bucket_diameter_all"] == 32
