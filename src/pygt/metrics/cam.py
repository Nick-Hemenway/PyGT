import logging

# from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

# import pandas as pd

# from hydrodata.signal import FFT_of_df
# from pycam.optimize.metrics_base import AbstractMetric
# from scipy.integrate import cumulative_simpson, quad
# from scipy.interpolate import CubicSpline
# from scipy.optimize import root_scalar

logger = logging.getLogger(__name__)


def impulse_width(theta1: float, theta2: float, theta3: float, theta4: float) -> float:
    return sum([theta1, theta2, theta3, theta4])


def stability(impulse_width: float, natural_freq_hz: float, max_rpm: float) -> float:
    return (impulse_width * natural_freq_hz) / (180 * (max_rpm / 60))


def lubrication_number(
    angle: NDArray, entrainment_vel: NDArray, theta6: float
) -> float:
    angle = np.asarray(angle)
    entrainment_vel = np.asarray(entrainment_vel)
    mask = (angle > -theta6) & (angle < theta6)
    return 2 * np.abs(entrainment_vel[mask]).min()


# class FirstLubricationNodeInclination(AbstractMetric):
#     def __init__(
#         self,
#         column: str,
#         duration_param: str,
#         name: Optional[str] = None,
#         label: Optional[str] = None,
#     ):
#         self.column = column
#         self.name = name
#         self.label = label
#         self.duration_param = duration_param

#     def __call__(
#         self, sim_data: Union[pd.DataFrame, None], params: dict[str, float]
#     ) -> float:
#         if sim_data is None:
#             return None

#         try:
#             velocity = sim_data[self.column]
#         except KeyError:
#             logger.warning(
#                 f'FirstLubricationNodeInclination: Column "{self.column}" is not in dataframe'
#             )
#             return None

#         duration = params[self.duration_param]
#         first_half = velocity.loc[-duration / 2 : 0]

#         velocity = CubicSpline(x=first_half.index, y=first_half.to_numpy())
#         try:
#             result = root_scalar(velocity, bracket=[-duration / 2, 0])
#         except ValueError:
#             logger.exception(
#                 "Root solve in FirstLubricationNodeInclination not succesful -- returning None"
#             )
#             # we will assume that if root finding fails, e.g. case when we get
#             # "f(a) and f(b) must have different signs" error, that the profile is
#             # probably not well behaved anyway, so we'll return None assuming the
#             # profile is bad
#             return None
#         deg_to_rad = 180 / np.pi
#         cam_to_crank_deg = 2

#         if not result.converged:
#             logger.error("Lubrication node did not converge properly -- returning None")
#             return None

#         return (
#             velocity(result.root, nu=1) * -1 * deg_to_rad * cam_to_crank_deg
#         )  # mm/deg/crank-rad


# class FirstLubricationNodeContactPressure(AbstractMetric):
#     def __init__(
#         self,
#         column_velocity: str,
#         column_pressure: str,
#         duration_param: str,
#         name: Optional[str] = None,
#         label: Optional[str] = None,
#     ):
#         self.column_velocity = column_velocity
#         self.column_pressure = column_pressure
#         self.name = name
#         self.label = label
#         self.duration_param = duration_param

#     def __call__(
#         self, sim_data: Union[pd.DataFrame, None], params: dict[str, float]
#     ) -> float:
#         if sim_data is None:
#             return None

#         try:
#             velocity = sim_data[self.column_velocity]
#         except KeyError:
#             logger.warning(
#                 f'FirstLubricationNodeContactPressure: Column "{self.column_velocity}" is not in dataframe'
#             )
#             return None

#         duration = params[self.duration_param]
#         first_half = velocity.loc[-duration / 2 : 0]

#         # solve for when entrainment velocity (relative velocity) goes to zero
#         velocity = CubicSpline(x=first_half.index, y=first_half.to_numpy())
#         try:
#             result = root_scalar(velocity, bracket=[-duration / 2, 0])
#         except ValueError:
#             logger.exception(
#                 "Root solve in FirstLubricationNodeContactPressure not succesful -- returning None"
#             )
#             # we will assume that if root finding fails, e.g. case when we get
#             # "f(a) and f(b) must have different signs" error, that the profile is
#             # probably not well behaved anyway, so we'll return None assuming the
#             # profile is bad
#             return None

#         if not result.converged:
#             logger.error("Lubrication node did not converge properly -- returning None")
#             return None

#         node_angle = result.root

#         try:
#             pressure = sim_data[self.column_pressure]
#         except KeyError:
#             logger.warning(
#                 f'FirstLubricationNodeContactPressure: Column "{self.column_pressure}" is not in dataframe'
#             )
#             return None

#         # evaluate pressure at location where velocity goes to zero
#         pressure_continuous = CubicSpline(pressure.index, pressure.to_numpy())
#         return pressure_continuous(node_angle)


# class LubricationNumber(AbstractMetric):
#     def __init__(
#         self,
#         column: str,
#         duration_param: str,
#         theta_template: str,
#         name: Optional[str] = None,
#         label: Optional[str] = None,
#     ):
#         # column represents relative velocity between cam surface and bucket
#         self.column = column  # e.g. "Int_Entrainment Velocity"
#         self.name = name
#         self.label = label
#         self.duration_param = duration_param
#         self.theta_template = theta_template  # e.g. "int_theta{}"

#     def theta6(self, params: dict[str, float]):
#         duration = params[self.duration_param]
#         thetas = [params[self.theta_template.format(i)] for i in range(1, 6)]
#         return duration / 2 - sum(thetas)

#     def __call__(
#         self, sim_data: Union[pd.DataFrame, None], params: dict[str, float]
#     ) -> float:
#         if sim_data is None:
#             return None

#         theta6 = self.theta6(params)
#         try:
#             entrainment_vel = sim_data.loc[-theta6:theta6, self.column]
#         except KeyError:
#             logger.warning(
#                 f'MaxLubricationNumber: Column "{self.column}" is not in dataframe'
#             )
#             return None

#         # minimum magnitude of entrainment velocity over theta6
#         return 2 * entrainment_vel.abs().min()


# class LiftArea(AbstractMetric):
#     def __init__(
#         self,
#         column: str,
#         duration_param: str,
#         ramp_height_param: str,
#         name: Optional[str] = None,
#         label: Optional[str] = None,
#     ):
#         self.name = name
#         self.label = label
#         self.column = column
#         self.duration_param = duration_param
#         self.ramp_height_param = ramp_height_param

#     def __call__(
#         self, sim_data: Union[pd.DataFrame, None], params: dict[str, float]
#     ) -> float:
#         if sim_data is None:
#             return None

#         duration = params[self.duration_param]
#         ramp_height = params[self.ramp_height_param]
#         try:
#             lift = sim_data[self.column]
#         except KeyError:
#             logger.warning(f'LiftArea: Column "{self.column}" is not in dataframe')
#             return None
#         # create continuous function, removing ramp height
#         f = CubicSpline(sim_data.index, lift - ramp_height)
#         # integrate area under lift curve between ramps
#         out = quad(f, -duration / 2, duration / 2, full_output=True)
#         return out[0]  # area in first index of tuple


# class BucketContactLateralPositionMargin(AbstractMetric):
#     def __init__(
#         self,
#         column: str,
#         bucket_diameter_param: str,
#         duration_param: str,
#         name: Optional[str] = None,
#         label: Optional[str] = None,
#     ):
#         self.column = column
#         self.bucket_diameter_param = bucket_diameter_param
#         self.name = name
#         self.label = label
#         self.duration_param = duration_param

#     def __call__(
#         self, sim_data: Union[pd.DataFrame, None], params: dict[str, float]
#     ) -> float:
#         # case where sim failed and there is no data
#         if sim_data is None:
#             return None

#         try:
#             position: pd.Series = sim_data[self.column]  # pd.Series
#         except KeyError:
#             logger.warning(
#                 f'BucketContactLateralPositionMargin: Column "{self.column}" is not in dataframe'
#             )
#             # case where sim data is made but quasidynamic analysis fails and key isn't available
#             return None

#         duration = params[self.duration_param]
#         bucket_diameter = params[self.bucket_diameter_param]

#         opening = np.abs(position.loc[-duration / 2 : duration / 2].min())
#         closing = np.abs(position.loc[-duration / 2 : duration / 2].max())

#         lateral_position = max(opening, closing)
#         return bucket_diameter / 2 - lateral_position


# class MinNegativeRadiusOfCurvature(AbstractMetric):
#     def __init__(
#         self,
#         column: str,
#         duration_param: str,
#         name: Optional[str] = None,
#         label: Optional[str] = None,
#         no_negative_val: float = 1e9,
#     ):
#         self.name = name
#         self.label = label
#         self.column = column
#         self.duration_param = duration_param
#         self.no_negative_val = no_negative_val

#     def __call__(
#         self, sim_data: Union[pd.DataFrame, None], params: dict[str, float]
#     ) -> float:
#         if sim_data is None:
#             return None

#         duration = params[self.duration_param]
#         try:
#             roc = sim_data[self.column]
#         except KeyError:
#             logger.warning(
#                 f'MinNegativeRadiusOfCurvature: Column "{self.column}" is not in dataframe'
#             )
#             return None

#         roc = roc.loc[-duration / 2 : duration / 2]
#         valid = roc[roc.lt(0)]
#         if len(valid) == 0:
#             return self.no_negative_val
#         else:
#             return valid.abs().min()


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
