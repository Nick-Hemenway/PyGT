from pathlib import Path

import pandas as pd
import pytest

from pygt.metrics.cam import duration, impulse_width, lubrication_number


@pytest.fixture
def cam_profile() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "assets/cam_profile_01.csv")


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


def test_duration(cam_profile: pd.DataFrame):
    angle = cam_profile["angle"].to_numpy()
    lift = cam_profile["int_lift"].to_numpy()
    expected_duration = 124.31826530150819  # Exact value used to generate profile
    ramp_height = 0.4
    assert duration(angle, lift, ramp_height) == pytest.approx(
        expected_duration, abs=0.5
    )


def test_impulse_width(cam_profile: pd.DataFrame):
    angle = cam_profile["angle"].to_numpy()
    acceleration = cam_profile["int_acceleration"].to_numpy()
    duration = 124.31826530150819  # Exact value used to generate profile
    mask = (angle >= -duration / 2) & (angle <= duration / 2)
    angle = angle[mask]
    accel = acceleration[mask]
    expected_impulse = 15.663340154677861  # Summing theta1-theta3 of exact profile
    assert impulse_width(angle, accel) == pytest.approx(expected_impulse, abs=0.5)
