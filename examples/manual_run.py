import logging
from enum import StrEnum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optwrap as opt

import pygt as gt
from pygt import metrics
from pygt.resolvers import Ref, Signal, Window

plt.set_loglevel("WARNING")
logging.basicConfig(level=logging.INFO, filename="./logs/manual_run.log")


class Columns(StrEnum):
    LIFT = "lift"
    VELOCITY = "vel"
    ACCEL = "accel"
    JERK = "jerk"
    ROC = "roc_knife"
    ROC_EFF = "roc_effective"
    SPRING_MARGIN = "spring_margin"
    CONTACT_PRESSURE_IDLE = "contact_pressure_idle"
    CONTACT_PRESSURE_MAX_SPEED = "contact_pressure_max_speed"


plt.style.use("bmh")

export_spec = gt.ExportSpec(
    name="export",
    config_source="./gt_files/export_configuration_non_integrated.exp",
    output_filename="exported.txt",
    index_name="angle",
    name_map={
        "Lift": Columns.LIFT,
        "Velocity": Columns.VELOCITY,
        "Acceleration": Columns.ACCEL,
        "Jerk": Columns.JERK,
        "RoC (Knife Edge)": Columns.ROC,
        "RoC Effective": Columns.ROC_EFF,
        "Spring Margin @6000rpm": Columns.SPRING_MARGIN,
        "Contact Pressure @650rpm": Columns.CONTACT_PRESSURE_IDLE,
        "Contact Pressure @6000rpm": Columns.CONTACT_PRESSURE_MAX_SPEED,
    },
)
sampled_spec = gt.SampledOutputSpec(
    name="sampled",
    fname="Output_v1.trn",
    index_name="count",
    name_map={"Cyc.Angle": "angle", "Kin vel": "vel"},
)

root = Path("./runs/manual_run")
root.mkdir(exist_ok=True, parents=True)
factory = gt.WorkspaceFactory(model_path="./gt_files/non_integrated.gtm", root=root)
# workspace = factory.build(export_spec=export_spec)

repo = gt.WorkspaceRepository(
    root=root, retention_policy=gt.workspace.KeepRecentPolicy(keep=5)
)
repo.refresh()
repo.apply_retention_policy()
workspace = repo.latest()

reg = opt.Registry()

reg.add("duration", lower=85, upper=130, default=115)
reg.add("theta1", lower=1, upper=20, default=10)
reg.add("theta2", lower=1, upper=15, default=10)
reg.add("theta3", lower=1, upper=20, default=10)
reg.add("theta5", lower=1, upper=20, default=5)
reg.add_operating_point_parameter(opt.OperatingPointParameter("erpm"))

op_points = opt.OperatingPoints(erpm=[1000, 2000, 3000])

x0 = reg.pack()
x1 = np.vstack([x0, 1.1 * x0, 1.05 * x0, 1.2 * x0, 1.6 * x0])
x1[1, 1] = 100  # overwrite theta1 with bad value to make design fail
ctx = reg.context_physical(x1, operating_points=op_points)

guard = gt.build_guard_design(reg, op_points)

caselist = gt.CaseList.from_context(ctx)
expanded = gt.ExpandedCaseList.from_packet_size(
    caselist=caselist, packet_size=4, guard_design=guard
)

gt.DesignWriter().write_file(caselist=expanded, workspace=workspace)
sim = gt.SerialExecution()
sim.run(workspace)
gt.DataExporter().export(workspace=workspace, export_spec=export_spec)

data = gt.ExportedOutputParser().read(
    spec=export_spec,
    workspace=workspace,
    caselist=expanded,
)
data_sampled = gt.SampledOutputParser().read(
    spec=sampled_spec,
    workspace=workspace,
    caselist=expanded,
)

between_ramps = Window(Signal("angle"), duration=Ref("duration"))
func = metrics.MultiCaseEvaluator(
    func=np.max,
    case_dependencies=Signal(
        Columns.CONTACT_PRESSURE_IDLE,
        where=between_ramps,
    ),
    op_points_count=op_points.size,
)

print(func(data.case_data, context=ctx))

fig, ax = plt.subplots()
ax.plot(
    data.case_data[0].get("angle"), data.case_data[0].get(Columns.CONTACT_PRESSURE_IDLE)
)
