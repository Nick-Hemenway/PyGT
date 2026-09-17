import logging
from enum import StrEnum

import matplotlib.pyplot as plt
import numpy as np
import optwrap as opt
import optwrap.resolvers as rs

from pygt import metrics
from pygt.cases import CaseList, ExpandedCaseList, build_guard_design
from pygt.data_export import DataExporter
from pygt.output_spec import ExportSpec, SampledOutputSpec
from pygt.param_file import DesignWriter  # noqa
from pygt.results_parse import (
    ExportedOutputParser,
    SampledOutputParser,
)
from pygt.results_sanitize import QDAResultsAlign
from pygt.simdata import Signal
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
logging.basicConfig(level=logging.INFO, filename="./logs/metrics.log")


class Signals(StrEnum):
    INDEX = "angle"
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


class SampledSignals(StrEnum):
    INDEX = "count"
    TORQUE = "torque"
    POWER = "power"
    OVERLAP = "overlap"


plt.style.use("bmh")

export_spec = ExportSpec(
    name="export",
    config_source="./gt_files/export_configuration_integrated.exp",
    output_filename="exported.txt",
    index_name=Signals.INDEX,
    name_map={
        "Int_Lift": Signals.INT_LIFT,
        "Int_Velocity": Signals.INT_VELOCITY,
        "Int_Acceleration": Signals.INT_ACCEL,
        "Int_Jerk": Signals.INT_JERK,
        "Int_RoC (Knife Edge)": Signals.INT_ROC,
        "Int_RoC Effective": Signals.INT_ROC_EFF,
        "Int_Spring Margin @6000rpm": Signals.INT_SPRING_MARGIN,
        "Int_Contact Pressure @650rpm": Signals.INT_CONTACT_PRESSURE_IDLE,
        "Int_Contact Pressure @6000rpm": Signals.INT_CONTACT_PRESSURE_MAX_SPEED,
        "Exh_Lift": Signals.EXH_LIFT,
        "Exh_Velocity": Signals.EXH_VELOCITY,
        "Exh_Acceleration": Signals.EXH_ACCEL,
        "Exh_Jerk": Signals.EXH_JERK,
        "Exh_RoC (Knife Edge)": Signals.EXH_ROC,
        "Exh_RoC Effective": Signals.EXH_ROC_EFF,
        "Exh_Spring Margin @6000rpm": Signals.EXH_SPRING_MARGIN,
        "Exh_Contact Pressure @650rpm": Signals.EXH_CONTACT_PRESSURE_IDLE,
        "Exh_Contact Pressure @6000rpm": Signals.EXH_CONTACT_PRESSURE_MAX_SPEED,
    },
)
sampled_spec = SampledOutputSpec(
    name="sampled",
    fname="RLT.trn",
    index_name=SampledSignals.INDEX,
    name_map={
        "Brake_Torque": SampledSignals.TORQUE,
        "Brake_Power": SampledSignals.POWER,
        "Overlap_Factor": SampledSignals.OVERLAP,
    },
)

root = "./runs/runs_integrated"
factory = WorkspaceFactory(model_path="./gt_files/integrated.gtm", root=root)
# workspace = factory.build(export_spec=export_spec)

repo = WorkspaceRepository(root=root, retention_policy=KeepRecentPolicy(keep=5))
repo.refresh()
repo.apply_retention_policy()
workspace = repo.latest()
clean_all_distributed_runs(repo)


class Params(StrEnum):
    INT_VALVE_CL = "VALVE_CL_INT"
    INT_LIFT_MAX = "int_lift_max"
    INT_DURATION = "int_duration"
    INT_THETA1 = "int_theta1"
    INT_THETA2 = "int_theta2"
    INT_THETA3 = "int_theta3"
    INT_THETA4 = "int_theta4"
    INT_THETA5 = "int_theta5"
    INT_F1 = "int_f1"
    INT_F2 = "int_f2"
    INT_F3 = "int_f3"
    INT_F4 = "int_f4"
    EXH_VALVE_CL = "VALVE_CL_EXH"
    EXH_LIFT_MAX = "exh_lift_max"
    EXH_DURATION = "exh_duration"
    EXH_THETA1 = "exh_theta1"
    EXH_THETA2 = "exh_theta2"
    EXH_THETA3 = "exh_theta3"
    EXH_THETA4 = "exh_theta4"
    EXH_THETA5 = "exh_theta5"
    EXH_F1 = "exh_f1"
    EXH_F2 = "exh_f2"
    EXH_F3 = "exh_f3"
    EXH_F4 = "exh_f4"


class Op(StrEnum):
    RPM = "erpm"


reg = opt.Registry()

reg.add(Params.INT_VALVE_CL, lower=85, upper=130, default=114)
reg.add(Params.INT_LIFT_MAX, lower=85, upper=130, default=13.31)
reg.add(Params.INT_DURATION, lower=85, upper=130, default=118)
reg.add(Params.INT_THETA1, lower=0, upper=20, default=1.5)
reg.add(Params.INT_THETA2, lower=0, upper=15, default=9.3)
reg.add(Params.INT_THETA3, lower=0, upper=20, default=11.9)
reg.add(Params.INT_THETA4, lower=0, upper=20, default=1)
reg.add(Params.INT_THETA5, lower=0, upper=20, default=6.6)
reg.add(Params.INT_F1, lower=0, upper=4, default=1.130)
reg.add(Params.INT_F2, lower=0, upper=4, default=1.503)
reg.add(Params.INT_F3, lower=0, upper=4, default=1.140)
reg.add(Params.INT_F4, lower=0, upper=4, default=1.916)
reg.add(Params.EXH_VALVE_CL, lower=85, upper=130, default=120)
reg.add(Params.EXH_LIFT_MAX, lower=85, upper=130, default=13.49)
reg.add(Params.EXH_DURATION, lower=85, upper=130, default=142)
reg.add(Params.EXH_THETA1, lower=0, upper=20, default=1.5)
reg.add(Params.EXH_THETA2, lower=0, upper=15, default=9.3)
reg.add(Params.EXH_THETA3, lower=0, upper=20, default=11.9)
reg.add(Params.EXH_THETA4, lower=0, upper=20, default=1)
reg.add(Params.EXH_THETA5, lower=0, upper=20, default=6.6)
reg.add(Params.EXH_F1, lower=0, upper=4, default=1.130)
reg.add(Params.EXH_F2, lower=0, upper=4, default=1.503)
reg.add(Params.EXH_F3, lower=0, upper=4, default=1.140)
reg.add(Params.EXH_F4, lower=0, upper=4, default=1.916)
reg.add_operating_point_param("erpm")

op_points = opt.OperatingPoints(erpm=[1000, 2000, 3000])

x0 = reg.pack()
x1 = np.vstack([x0, 1.01 * x0, 1.02 * x0, 1.03 * x0])
x1[1, 0] = 160  # Move int_valve_cl to cause wrapping
guard = build_guard_design(reg, op_points)

popsize = op_points.size * x1.shape[0]
licenses = 3
packet_size = int(popsize / licenses) + 1


def simulate(expanded: ExpandedCaseList) -> None:
    DesignWriter().write_file(caselist=expanded, workspace=workspace)
    sim = SerialExecution()
    sim = DistributedExecution(licenses=licenses, packet_size=packet_size)
    _ = sim.run(workspace)
    DataExporter().export(workspace=workspace, export_spec=export_spec)


reg.add_provider(opt.Provider(name="caselist", func=CaseList))
reg.add_provider(
    opt.Provider(
        name="expanded",
        func=lambda caselist: ExpandedCaseList(
            caselist, packet_size=packet_size, guard_design=guard
        ),
    )
)
reg.add_provider(opt.Provider(name="run_sim", func=simulate))
reg.add_provider(
    opt.Provider(
        name="sim_data",
        func=lambda expanded, run_sim: ExportedOutputParser().read(
            spec=export_spec, workspace=workspace, caselist=expanded
        ),
    )
)
reg.add_provider(
    opt.Provider(
        name="sim_data_sampled",
        func=lambda expanded, run_sim: SampledOutputParser().read(
            spec=sampled_spec, workspace=workspace, caselist=expanded
        ),
    )
)
int_qda_cols = [
    Signals.INT_CONTACT_PRESSURE_IDLE,
    Signals.INT_CONTACT_PRESSURE_MAX_SPEED,
    Signals.INT_SPRING_MARGIN,
]

exh_qda_cols = [
    Signals.EXH_CONTACT_PRESSURE_IDLE,
    Signals.EXH_CONTACT_PRESSURE_MAX_SPEED,
    Signals.EXH_SPRING_MARGIN,
]

qda_int = QDAResultsAlign(
    timing_param=Params.INT_VALVE_CL,
    intake=True,
    start_of_cycle_angle=-60,
    signals=int_qda_cols,
)
qda_exh = QDAResultsAlign(
    timing_param=Params.EXH_VALVE_CL,
    intake=False,
    start_of_cycle_angle=-60,
    signals=exh_qda_cols,
)
reg.add_provider(
    opt.Provider(
        name="case_data",
        func=lambda sim_data: [qda_int(qda_exh(case)) for case in sim_data.case_data],
    )
)
reg.add_provider(
    opt.Provider(
        name="case_data_sampled",
        func=lambda sim_data_sampled: sim_data_sampled.case_data,
    )
)

m = opt.AggregatedMetric(
    name="overlap",
    func=metrics.MultiCaseEvaluator(
        func=lambda signal: np.squeeze(signal),
        arg_map={
            "signal": Signal(SampledSignals.OVERLAP),
        },
        op_points_count=op_points.size,
    ),
    arg_map={"cases": "case_data_sampled"},
    where=rs.Smallest(Op.RPM),
    agg=opt.aggregate.Squeeze(),
)
m1 = opt.AggregatedMetric(
    name="int_spring_margin",
    func=metrics.MultiCaseEvaluator(
        func=metrics.SignalAggregate(kind="min"),
        arg_map={
            "index": Signal(Signals.INDEX),
            "signal": Signal(Signals.INT_SPRING_MARGIN),
            "duration": Params.INT_DURATION,
        },
        op_points_count=op_points.size,
    ),
    arg_map={"cases": "case_data"},
    # where=rs.IndexedOperatingPoint(index=1),
    where=rs.EqualTo(Op.RPM, 2000),
    agg=opt.aggregate.Squeeze(),
)

reg.add_metric(m)
reg.add_metric(m1)

ctx = reg.context_physical(x1, operating_points=op_points)


# # TODO: Create function that gets operating point rpms and qda rpm list and returns
# # TODO: index to extract qda results from such that they are correct


# # f(cases_sampled)
# m.evaluate(ctx)
# m1.evaluate(ctx)
