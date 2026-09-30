import logging

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

logger = logging.getLogger(__name__)


def impulse_width(theta1: float, theta2: float, theta3: float, theta4: float) -> float:
    return sum([theta1, theta2, theta3, theta4])


def stability(impulse_width: float, natural_freq_hz: float, max_rpm: float) -> float:
    return (impulse_width * natural_freq_hz) / (180 * (max_rpm / 60))


def lubrication_number(
    angle: NDArray, entrainment_velocity: NDArray, theta6: float
) -> float:
    angle = np.asarray(angle)
    entrainment_velocity = np.asarray(entrainment_velocity)
    mask = (angle > -theta6) & (angle < theta6)
    return 2 * np.abs(entrainment_velocity[mask]).min()


def lift_area(angle: NDArray, lift: NDArray, ramp_height: float) -> float:
    angle = np.asarray(angle)
    lift = np.asarray(lift)
    f = CubicSpline(angle, lift - ramp_height)
    out = quad(f, angle.min(), angle.max(), full_output=True)
    return out[0]  # area in first index of tuple


def bucket_lateral_position_margin(position: NDArray, bucket_diameter: float) -> float:
    position = np.asarray(position)
    opening = np.abs(position.min())
    closing = np.abs(position.max())
    lateral_position = max(opening, closing)
    return bucket_diameter / 2 - lateral_position


def min_negative_radius_of_curvature(
    roc: NDArray, no_negative_val: float = 1e9
) -> float:
    valid = roc[roc < 0]
    if len(valid) == 0:
        return no_negative_val
    else:
        return np.abs(valid).min()


def _first_lube_node_location(
    angle: NDArray, entrainment_velocity: NDArray
) -> tuple[float, CubicSpline]:
    first_half_mask = angle < 0
    vel_func = CubicSpline(
        x=angle[first_half_mask], y=entrainment_velocity[first_half_mask]
    )
    try:
        result = root_scalar(vel_func, bracket=[angle.min(), 0])
    except ValueError:
        logger.exception(
            "Root solve for first lube node location was not successful -- returning None"
        )
        # we will assume that if root finding fails, e.g. case when we get
        # "f(a) and f(b) must have different signs" error, that the profile is
        # probably not well behaved anyway, so we'll return None assuming the
        # profile is bad
        return None
    if not result.converged:
        logger.warning("First lube node root solve did not converge -- returning None")
        return None
    return result.root, vel_func


def first_lube_node_inclination(angle: NDArray, entrainment_velocity: NDArray) -> float:
    """Compute inclination of entrainment velocity and lubrication node (zero-crossing)

    Parameters
    ----------
    angle : NDArray
        Crank angle in cam-degrees
    entrainment_velocity : NDArray
        Entrainment velocity in mm / crank-radian

    Returns
    -------
    float
        Slope of entrainment velocity as it crosses through zero [mm / cam-rad ^ 2]

    Notes
    -----

    Units:
    - Entrainment velocity units are in mm / crank-radian
    - Angle input units are in cam-deg
    - Taking slope of input data yields ( mm / crank-radian) / cam-deg = mm / (crank-radian * cam-deg)
    - We can convert to mm / cam-rad ^ 2 as follows:
        - mm / (crank-radian * cam-deg) * (180 cam-deg / pi cam-radian) * (2 crank-radian / cam-radian)


    """
    angle = np.asarray(angle)
    entrainment_velocity = np.asarray(entrainment_velocity)
    node_angle, vel_func = _first_lube_node_location(angle, entrainment_velocity)
    if node_angle is None:
        return None
    slope = vel_func(node_angle, nu=1)  # Evaluate first derivative
    deg_to_rad = 180 / np.pi
    cam_to_crank_deg = 2
    # mm/deg/crank-rad
    return np.abs(slope) * deg_to_rad * cam_to_crank_deg


def first_lube_node_contact_pressure(
    angle: NDArray, entrainment_velocity: NDArray, pressure: NDArray
) -> float:
    # Sanitize inputs
    angle = np.asarray(angle)
    entrainment_velocity = np.asarray(entrainment_velocity)
    pressure = np.asarray(pressure)

    # Find location of first lube node
    node_angle, _ = _first_lube_node_location(angle, entrainment_velocity)

    # Evaluate pressure at first lube node
    pressure_func = CubicSpline(angle, pressure)
    return pressure_func(node_angle)


def theta6(
    duration: NDArray,
    theta1: NDArray,
    theta2: NDArray,
    theta3: NDArray,
    theta4: NDArray,
    theta5: NDArray,
) -> NDArray:
    return duration / 2 - (theta1 + theta2 + theta3 + theta4 + theta5)
