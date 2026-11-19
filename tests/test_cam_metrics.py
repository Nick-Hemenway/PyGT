from pathlib import Path

import pandas as pd
import pytest

from pygt.metrics.cam import duration, lubrication_number


@pytest.mark.parametrize(
    "angle",
    [
        [-2, -1, 0, 1, 2],
        [-2, -1, 0.1, 1, 2],
        [-2, -1, -0.1, 1, 2],
        [-2, -1, -0.01, 0.1, 2],
    ],
)
def test_lubrication_number_exactly_zero(angle):
    ent_vel = [0, -10, -20, -10, 0]
    assert lubrication_number(angle, ent_vel) == pytest.approx(40)


def test_duration():
    df = pd.read_csv(Path(__file__).parent / "assets/lift.csv")
    angle = df["angle"].to_numpy()
    lift = df["lift"].to_numpy()
    expected_duration = 124.31826530150819  # Exact value used to generate profile
    ramp_height = 0.4
    assert duration(angle, lift, ramp_height) == pytest.approx(
        expected_duration, abs=0.5
    )
