import logging
from enum import StrEnum

import matplotlib.pyplot as plt
import numpy as np
import optwrap as opt

from pygt.cases import CaseList, ExpandedCaseList, build_guard_design
from pygt.data_export import DataExporter
from pygt.exceptions import MissingSignalError
from pygt.output_spec import ExportSpec, SampledOutputSpec
from pygt.param_file import DesignWriter  # noqa
from pygt.results_parse import (
    ExportedOutputParser,
    SampledOutputParser,
)
from pygt.results_sanitize import QDAResultsAlign
from pygt.simulation import (
    DistributedExecution,
    SerialExecution,
    clean_all_distributed_runs,
)
from pygt.workspace import (
    KeepRecentPolicy,
    WorkspaceFactory,
    WorkspaceRepository,
)

plt.set_loglevel("WARNING")
logging.basicConfig(level=logging.INFO, filename="./logs/qda_align.log")


class Columns(StrEnum):
    INT_LIFT = "int_lift"
    INT_VELOCITY = "int_vel"
    INT_ACCEL = "int_accel"
    INT_JERK = "int_jerk"
    INT_ROC = "int_roc_knife"
    INT_ROC_EFF = "int_roc_effective"
    INT_SPRING_MARGIN = "int_spring_margin"
    INT_CONTACT_PRESSURE_IDLE = "int_contact_pressure_idle"
    INT_CONTACT_PRESSURE_MAX_SPEED = "int_contact_pressure_max_speed"
    EXH_LIFT = "exh_lift"
    EXH_VELOCITY = "exh_vel"
    EXH_ACCEL = "exh_accel"
    EXH_JERK = "exh_jerk"
    EXH_ROC = "exh_roc_knife"
    EXH_ROC_EFF = "exh_roc_effective"
    EXH_SPRING_MARGIN = "exh_spring_margin"
    EXH_CONTACT_PRESSURE_IDLE = "exh_contact_pressure_idle"
    EXH_CONTACT_PRESSURE_MAX_SPEED = "exh_contact_pressure_max_speed"


plt.style.use("bmh")

export_spec = ExportSpec(
    name="export",
    config_source="./gt_files/export_configuration_integrated.exp",
    output_filename="exported.txt",
    index_name="angle",
    name_map={
        "Int_Lift": Columns.INT_LIFT,
        "Int_Velocity": Columns.INT_VELOCITY,
        "Int_Acceleration": Columns.INT_ACCEL,
        "Int_Jerk": Columns.INT_JERK,
        "Int_RoC (Knife Edge)": Columns.INT_ROC,
        "Int_RoC Effective": Columns.INT_ROC_EFF,
        "Int_Spring Margin @6000rpm": Columns.INT_SPRING_MARGIN,
        "Int_Contact Pressure @650rpm": Columns.INT_CONTACT_PRESSURE_IDLE,
        "Int_Contact Pressure @6000rpm": Columns.INT_CONTACT_PRESSURE_MAX_SPEED,
        "Exh_Lift": Columns.EXH_LIFT,
        "Exh_Velocity": Columns.EXH_VELOCITY,
        "Exh_Acceleration": Columns.EXH_ACCEL,
        "Exh_Jerk": Columns.EXH_JERK,
        "Exh_RoC (Knife Edge)": Columns.EXH_ROC,
        "Exh_RoC Effective": Columns.EXH_ROC_EFF,
        "Exh_Spring Margin @6000rpm": Columns.EXH_SPRING_MARGIN,
        "Exh_Contact Pressure @650rpm": Columns.EXH_CONTACT_PRESSURE_IDLE,
        "Exh_Contact Pressure @6000rpm": Columns.EXH_CONTACT_PRESSURE_MAX_SPEED,
    },
)
sampled_spec = SampledOutputSpec(
    name="sampled",
    fname="RLT.trn",
    index_name="count",
    name_map={
        "Brake_Torque": "torque",
        "Brake_Power": "power",
        "Overlap_Factor": "overlap",
    },
)

root = "./runs/runs_integrated"
factory = WorkspaceFactory(model_path="./gt_files/integrated.gtm", root=root)
workspace = factory.build(export_spec=export_spec)

repo = WorkspaceRepository(root=root, retention_policy=KeepRecentPolicy(keep=5))
repo.refresh()
repo.apply_retention_policy()
workspace = repo.latest()
clean_all_distributed_runs(repo)

reg = opt.Registry()

reg.add("VALVE_CL_INT", lower=85, upper=130, default=114)
reg.add("int_lift_max", lower=85, upper=130, default=13.31)
reg.add("int_duration", lower=85, upper=130, default=118)
reg.add("int_theta1", lower=0, upper=20, default=1.5)
reg.add("int_theta2", lower=0, upper=15, default=9.3)
reg.add("int_theta3", lower=0, upper=20, default=11.9)
reg.add("int_theta4", lower=0, upper=20, default=1)
reg.add("int_theta5", lower=0, upper=20, default=6.6)
reg.add("int_f1", lower=0, upper=4, default=1.130)
reg.add("int_f2", lower=0, upper=4, default=1.503)
reg.add("int_f3", lower=0, upper=4, default=1.140)
reg.add("int_f4", lower=0, upper=4, default=1.916)
reg.add("VALVE_CL_EXH", lower=85, upper=130, default=120)
reg.add("exh_lift_max", lower=85, upper=130, default=13.49)
reg.add("exh_duration", lower=85, upper=130, default=142)
reg.add("exh_theta1", lower=0, upper=20, default=1.5)
reg.add("exh_theta2", lower=0, upper=15, default=9.3)
reg.add("exh_theta3", lower=0, upper=20, default=11.9)
reg.add("exh_theta4", lower=0, upper=20, default=1)
reg.add("exh_theta5", lower=0, upper=20, default=6.6)
reg.add("exh_f1", lower=0, upper=4, default=1.130)
reg.add("exh_f2", lower=0, upper=4, default=1.503)
reg.add("exh_f3", lower=0, upper=4, default=1.140)
reg.add("exh_f4", lower=0, upper=4, default=1.916)
reg.add_operating_point_param(opt.OperatingPointParameter("erpm"))

op_points = opt.OperatingPoints(erpm=[1000, 2000, 3000])

x0 = reg.pack()
x1 = np.vstack([x0, 1.01 * x0, 1.02 * x0, 1.03 * x0])
x1[1, 0] = 160  # Move int_valve_cl to cause wrapping
ctx = reg.context_physical(x1, operating_points=op_points)
guard = build_guard_design(reg, op_points)

popsize = op_points.size * x1.shape[0]
licenses = 3
packet_size = int(popsize / licenses) + 1

caselist = CaseList(ctx)
expanded = ExpandedCaseList(
    caselist=caselist, packet_size=packet_size, guard_design=guard
)

DesignWriter().write_file(caselist=expanded, workspace=workspace)
sim = SerialExecution()
sim = DistributedExecution(licenses=licenses, packet_size=packet_size)
res = sim.run(workspace)
DataExporter().export(workspace=workspace, export_spec=export_spec)


data = ExportedOutputParser().read(
    spec=export_spec,
    workspace=workspace,
    caselist=expanded,
)
data_sampled = SampledOutputParser().read(
    spec=sampled_spec,
    workspace=workspace,
    caselist=expanded,
)


int_qda_cols = [
    Columns.INT_CONTACT_PRESSURE_IDLE,
    Columns.INT_CONTACT_PRESSURE_MAX_SPEED,
    Columns.INT_SPRING_MARGIN,
]

exh_qda_cols = [
    Columns.EXH_CONTACT_PRESSURE_IDLE,
    Columns.EXH_CONTACT_PRESSURE_MAX_SPEED,
    Columns.EXH_SPRING_MARGIN,
]

qda_int = QDAResultsAlign(
    timing_param="VALVE_CL_INT",
    intake=True,
    start_of_cycle_angle=-60,
    signals=int_qda_cols,
)
qda_exh = QDAResultsAlign(
    timing_param="VALVE_CL_EXH",
    intake=False,
    start_of_cycle_angle=-60,
    signals=exh_qda_cols,
)


cases = [qda_int(qda_exh(case)) for case in data.case_data]


def plot_signal(int_col, exh_col):
    fig_exh, ax_exh = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig_int, ax_int = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for case0, case1 in zip(data.case_data, cases):
        x = case0.get("angle")
        try:
            y0 = case0.get(int_col)
            y1 = case1.get(int_col)
            ax_int[0].plot(x, y0)
            ax_int[1].plot(x, y1)
        except MissingSignalError:
            pass

        try:
            y0 = case0.get(exh_col)
            y1 = case1.get(exh_col)
            ax_exh[0].plot(x, y0)
            ax_exh[1].plot(x, y1)
        except MissingSignalError:
            pass

    ax_exh[0].set_title(exh_col)
    ax_int[0].set_title(int_col)

    fig_int.tight_layout()
    fig_exh.tight_layout()


for int_, exh in zip(int_qda_cols, exh_qda_cols):
    plot_signal(int_, exh)
