import logging

import matplotlib.pyplot as plt
import numpy as np
import optwrap as opt
import optwrap.resolvers as rs
from gt_vars import (
    Op,
    SampledSignals,
    Signals,
    Vars,
    export_signal_map,
    sampled_signal_map,
)

import pygt.metrics.cam as cm
from pygt import metrics
from pygt.cases import CaseList, ExpandedCaseList, build_guard_design
from pygt.data_export import DataExporter
from pygt.output_spec import ExportSpec, SampledOutputSpec
from pygt.param_file import DesignWriter  # noqa
from pygt.resolvers import Literal, Ref, Signal, Window
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

plt.set_loglevel("INFO")
logging.basicConfig(
    level=logging.INFO,
    filename="./logs/metrics.log",
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

plt.style.use("bmh")

export_spec = ExportSpec(
    name="export",
    config_source="./gt_files/export_configuration_integrated.exp",
    output_filename="exported.txt",
    index_name=Signals.INDEX,
    name_map=export_signal_map,
)
sampled_spec = SampledOutputSpec(
    name="sampled",
    fname="RLT.trn",
    index_name=SampledSignals.INDEX,
    name_map=sampled_signal_map,
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

reg.add(Vars.INT_VALVE_CL, lower=85, upper=130, default=113)
reg.add(Vars.INT_LIFT_MAX, lower=85, upper=130, default=10)
reg.add(Vars.INT_DURATION, lower=85, upper=130, default=130)
reg.add(Vars.INT_THETA1, lower=0, upper=20, default=8)
reg.add(Vars.INT_THETA2, lower=0, upper=15, default=8)
reg.add(Vars.INT_THETA3, lower=0, upper=20, default=8)
reg.add(Vars.INT_THETA4, lower=0, upper=20, default=8)
reg.add(Vars.INT_THETA5, lower=0, upper=20, default=8)
reg.add(Vars.INT_F1, lower=0, upper=4, default=1.130)
reg.add(Vars.INT_F3, lower=0, upper=4, default=1.140)
reg.add(Vars.INT_F4, lower=0, upper=4, default=1.916)
reg.add(Vars.EXH_VALVE_CL, lower=85, upper=130, default=110)
reg.add(Vars.EXH_LIFT_MAX, lower=85, upper=130, default=10)
reg.add(Vars.EXH_DURATION, lower=85, upper=130, default=130)
reg.add(Vars.EXH_THETA1, lower=0, upper=20, default=8)
reg.add(Vars.EXH_THETA2, lower=0, upper=15, default=8)
reg.add(Vars.EXH_THETA3, lower=0, upper=20, default=8)
reg.add(Vars.EXH_THETA4, lower=0, upper=20, default=8)
reg.add(Vars.EXH_THETA5, lower=0, upper=20, default=8)
reg.add(Vars.EXH_F1, lower=0, upper=4, default=1.130)
reg.add(Vars.EXH_F3, lower=0, upper=4, default=1.140)
reg.add(Vars.EXH_F4, lower=0, upper=4, default=1.916)
reg.add_operating_point_param(opt.OperatingPointParameter(Op.RPM))

op_points = opt.OperatingPoints(**{Op.RPM: [1000, 2000, 3000]})

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
    sim = DistributedExecution(
        licenses=expanded.packet_count,
        packet_size=expanded.packet_size,
    )
    _ = sim.run(workspace)
    DataExporter().export(workspace=workspace, export_spec=export_spec)


reg.add_provider(opt.Provider(name="caselist", func=CaseList.from_context))
reg.add_provider(
    opt.Provider(
        name="expanded",
        func=lambda caselist: ExpandedCaseList.from_license_count(
            caselist, guard_design=guard, licenses=licenses
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
int_qda_cols = [sig for sig in Signals if "qda_int" in sig]
exh_qda_cols = [sig for sig in Signals if "qda_exh" in sig]

qda_int = QDAResultsAlign(
    timing_param=Vars.INT_VALVE_CL,
    intake=True,
    start_of_cycle_angle=-60,
    signals=int_qda_cols,
)
qda_exh = QDAResultsAlign(
    timing_param=Vars.EXH_VALVE_CL,
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
        case_arg_map={
            "signal": Signal(SampledSignals.OVERLAP),
        },
        op_points_count=op_points.size,
    ),
    arg_map={"case_data": "case_data_sampled"},
    where=rs.Smallest(Op.RPM),
    agg=opt.aggregate.Squeeze(),
)

between_ramps_int = Window(Signal(Signals.INDEX), duration=Ref(Vars.INT_DURATION))
reg.add_metric(m)

m1 = opt.AggregatedMetric(
    name="int_spring_margin",
    func=metrics.MultiCaseEvaluator(
        func=metrics.signal_min,
        case_arg_map={
            "signal": Signal(Signals.QDA_INT_SPRING_MARGIN, where=between_ramps_int),
        },
        op_points_count=op_points.size,
    ),
    arg_map={"case_data": "case_data"},
    where=rs.EqualTo(Op.RPM, 2000),
    agg=opt.aggregate.Squeeze(),
)

m2 = opt.AggregatedMetric(
    name="int_contact_pressure",
    func=metrics.MultiCaseEvaluator(
        func=metrics.signal_max,
        case_arg_map={
            "signal": Signal(
                Signals.QDA_INT_CONTACT_PRESSURE_IDLE, where=between_ramps_int
            ),
        },
        op_points_count=op_points.size,
    ),
    arg_map={"case_data": "case_data"},
    where=rs.EqualTo(Op.RPM, 2000),
    agg=opt.aggregate.Squeeze(),
)

reg.add_provider(
    opt.Provider(
        name="impulse_width",
        func=metrics.MultiCaseEvaluator(
            func=cm.impulse_width,
            case_arg_map={
                "theta1": Vars.INT_THETA1,
                "theta2": Vars.INT_THETA2,
                "theta3": Vars.INT_THETA3,
                "theta4": Vars.INT_THETA4,
            },
            op_points_count=op_points.size,
        ),
        arg_map={"case_data": "case_data"},
    )
)

m3 = opt.AggregatedMetric(
    name="stability",
    func=lambda impulse_width: cm.stability(
        impulse_width=impulse_width, natural_freq_hz=1000, max_rpm=6600
    ),
    where=rs.FirstOperatingPoint(),
    agg=opt.aggregate.Squeeze(),
)

reg.add_provider(
    opt.Provider(
        name="theta6",
        func=lambda t1, t2, t3, t4, t5, duration: np.asarray(
            duration / 2 - (t1 + t2 + t3 + t4 + t5)
        ).flatten(),
        arg_map={
            "t1": Vars.INT_THETA1,
            "t2": Vars.INT_THETA2,
            "t3": Vars.INT_THETA3,
            "t4": Vars.INT_THETA4,
            "t5": Vars.INT_THETA5,
            "duration": Vars.INT_DURATION,
        },
    )
)

reg.add_metric(
    opt.AggregatedMetric(
        name="theta_6",
        func=lambda t1, t2, t3, t4, t5, duration: (
            duration / 2 - (t1 + t2 + t3 + t4 + t5)
        ),
        arg_map={
            "t1": Vars.INT_THETA1,
            "t2": Vars.INT_THETA2,
            "t3": Vars.INT_THETA3,
            "t4": Vars.INT_THETA4,
            "t5": Vars.INT_THETA5,
            "duration": Vars.INT_DURATION,
        },
        agg=opt.aggregate.Squeeze(),
        where=opt.resolvers.FirstOperatingPoint(),
    )
)

DV = Ref
theta6_int_case_resolver = DV(Vars.INT_DURATION) / 2 - (
    DV(Vars.INT_THETA1)
    + DV(Vars.INT_THETA2)
    + DV(Vars.INT_THETA3)
    + DV(Vars.INT_THETA4)
    + DV(Vars.INT_THETA5)
)

reg.add_metric(m1)
reg.add_metric(m2)
reg.add_metric(m3)


m4 = opt.Metric(
    name="lubrication_number",
    func=metrics.MultiCaseEvaluator(
        func=cm.lubrication_number,
        case_arg_map={
            "angle": Signal(Signals.INDEX),
            "entrainment_velocity": Signal(Signals.INT_ENTRAINMENT_VELOCITY),
        },
        context_arg_map={"theta6": "theta6"},
        op_points_count=op_points.size,
    ),
    arg_map={"case_data": "case_data"},
)

m5 = opt.Metric(
    name="lift_area",
    func=metrics.MultiCaseEvaluator(
        func=cm.lift_area,
        case_arg_map={
            "angle": Signal(Signals.INDEX),
            "lift": Signal(Signals.INT_LIFT),
            "ramp_height": Literal(0.1),
        },
        op_points_count=op_points.size,
    ),
)

m6 = opt.Metric(
    name="position_margin",
    func=metrics.MultiCaseEvaluator(
        func=cm.bucket_lateral_position_margin,
        case_arg_map={
            "position": Signal(Signals.INT_LATERAL_POSITION),
            "bucket_diameter": Literal(32),
        },
        op_points_count=op_points.size,
    ),
)

m7 = opt.Metric(
    name="min_neg_roc",
    func=metrics.MultiCaseEvaluator(
        func=cm.min_negative_radius_of_curvature,
        case_arg_map={
            "roc": Signal(Signals.INT_ROC),
        },
        op_points_count=op_points.size,
    ),
)


ctx = reg.context_physical(x1, operating_points=op_points)

print(ctx.int_contact_pressure)

print(m.evaluate(ctx))
print(m1.evaluate(ctx))
print(m2.evaluate(ctx))
print(m3.evaluate(ctx))
print(m4.evaluate(ctx))
print(m5.evaluate(ctx))
print(m6.evaluate(ctx))

print("\ntheta6 resolved:", theta6_int_case_resolver.resolve(ctx.case_data[0]))


# # # TODO: Create function that gets operating point rpms and qda rpm list and returns
# # # TODO: index to extract qda results from such that they are correct
# TODO: Should we use RoC effective of knife edge?
# TODO: Add ScalarExpand() resolver to expand scalar to number of cases. This would allow scalar providers
# to be used as input to multicaseevaluator
#
# TODO: Make sure we can run sims with guard design and no operating points
