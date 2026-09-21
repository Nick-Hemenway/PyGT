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


# class CamTorqueMetrics(AbstractMetric):
#     def __init__(
#         self,
#         bank_to_bank_angle_param: str,
#         int_timing_param: str,
#         exh_timing_param: str,
#         int_qda_col: str,
#         exh_qda_col: str,
#         cams_per_cyl_per_shaft: int = 2,
#         name: str | None = None,
#         label: str | None = None,
#         method_name: str = "max_energy_harmonic",
#     ):
#         self.bank_to_bank_angle_param = bank_to_bank_angle_param
#         self.int_timing_param = int_timing_param
#         self.exh_timing_param = exh_timing_param
#         self.int_qda_col = int_qda_col
#         self.exh_qda_col = exh_qda_col
#         self.cams_per_cyl_per_shaft = cams_per_cyl_per_shaft
#         self.name = name
#         self.label = label
#         self.method_name = method_name

#     def __call__(
#         self, sim_data: pd.DataFrame | None, params: dict[str, float]
#     ) -> float:
#         if sim_data is None:
#             return None
#         return getattr(self, self.method_name)(sim_data, params)

#     def max_energy_harmonic(
#         self, sim_data: pd.DataFrame | None, params: dict[str, float]
#     ) -> float:
#         fft = self.get_fft(sim_data, params)
#         if fft is None:
#             return None
#         fft = fft.squeeze()
#         return fft[fft.index <= 10].max()

#     def get_fft(self, df: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
#         df = self.process_sim_data(df, params=params)
#         if df is None:
#             return None
#         fft = FFT_of_df(df["primary_delta_integrated"].to_frame()).abs()
#         fft.index *= 720
#         return fft

#     def process_sim_data(
#         self, sim_data: pd.DataFrame, params: dict[str, float]
#     ) -> float:
#         df = self.get_cam_torques(df=sim_data, params=params)
#         if df is None:
#             return None
#         df = self.add_opposite_bank(df=df, params=params)
#         df = self.sum_banks(df)
#         df = df.assign(primary_delta=df.bank_sum - df.op_bank_sum)
#         df = self.add_energy(df)
#         return df

#     def add_energy(self, df: pd.DataFrame) -> pd.DataFrame:
#         for col in ["primary_delta"]:
#             name = f"{col}_integrated"
#             df[name] = cumulative_simpson(y=df[col], x=df.index, initial=0)
#             df[name] -= df[name].mean()
#         return df

#     def sum_banks(self, df: pd.DataFrame):
#         return df.assign(
#             bank_sum=df[self.exh_qda_col] + df[self.int_qda_col],
#             op_bank_sum=df[f"op_bank_{self.exh_qda_col}"]
#             + df[f"op_bank_{self.int_qda_col}"],
#         )

#     def get_cam_torques(self, df: pd.DataFrame, params: dict[str, float]) -> float:
#         df = self.get_sub_df(df, params)
#         if df is None:
#             return None
#         df *= self.cams_per_cyl_per_shaft
#         df = self.shift_cols(df, params)
#         df = self.compute_total_cam_torque(df)
#         return df

#     def add_opposite_bank(
#         self, df: pd.DataFrame, params: dict[str, float]
#     ) -> pd.DataFrame:
#         df1 = df.copy()
#         df1.index += params[self.bank_to_bank_angle_param]
#         df1.index %= 720
#         df1 = df1.sort_index()
#         df1 = df1.rename(columns=lambda col: f"op_bank_{col}")
#         out = pd.concat([df, df1], axis=1)
#         return out[~out.index.duplicated()]

#     def get_sub_df(self, df: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
#         cols = [self.int_qda_col, self.exh_qda_col]
#         try:
#             df_sub = df[cols]
#         except KeyError:
#             logger.warning(
#                 f'CamTorqueMetrics: One or more of columns "{cols}" are not in dataframe'
#             )
#             return None
#         return df_sub

#     def shift_cols(self, df: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
#         return pd.concat([self.shift_col(df, col, params) for col in df], axis=1)

#     def compute_total_cam_torque(self, df):
#         df_total = df.copy()
#         for i in range(1, 6):
#             df_temp = df.copy()
#             df_temp.index = (df.index + i * 120) % 720
#             # Sort index and add only values (not index aligned) because
#             # then we don't need to add by index values where the could be
#             # small floating point differences in angle after doing the add and mod
#             # operations. This allows us to avoid perfoming an interpolation onto the
#             # same axis as the original data. This works because we're simply shifting
#             # the data but the spacing between datapoints stays the same
#             df_temp = df_temp.sort_index()
#             df_total += df_temp.to_numpy()
#         return df_total

#     def shift_col(self, df, col, params) -> pd.DataFrame:
#         s = df[col]
#         if col == self.int_qda_col:
#             s.index += self.int_shift(params)
#         elif col == self.exh_qda_col:
#             s.index += self.exh_shift(params)
#         else:
#             raise ValueError(
#                 f"CamTorqueMetrics: Column needs to be either {self.int_qda_col} or {self.exh_qda_col}"
#             )
#         s.index %= 720
#         s = s.sort_index()
#         s = s[~s.index.duplicated()]
#         interp_func = CubicSpline(s.index, s.to_numpy())
#         new_angles = np.arange(0, 720, 0.5)
#         return pd.DataFrame(
#             data=interp_func(new_angles),
#             columns=[col],
#             index=pd.Index(new_angles, name="Angle"),
#         )

#     def int_shift(self, params: dict[str, float]) -> float:
#         return 360 + params[self.int_timing_param]

#     def exh_shift(self, params: dict[str, float]) -> float:
#         return 360 - params[self.exh_timing_param]
