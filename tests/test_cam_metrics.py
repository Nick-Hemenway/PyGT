import pytest

from pygt.metrics.cam import lubrication_number


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
