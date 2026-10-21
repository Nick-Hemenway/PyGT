import logging
from pathlib import Path

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

import pygt as gt
from pygt import metrics
from pygt.resolvers import Literal, Ref, ScalarExpand, Signal, Window

plt.set_loglevel("INFO")
logging.basicConfig(
    level=logging.INFO,
    filename="./logs/metrics.log",
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

plt.style.use("bmh")

export_spec = gt.ExportSpec(
    name="export",
    config_source="./gt_files/export_configuration_integrated.exp",
    output_filename="exported.txt",
    index_name=Signals.INDEX,
    name_map=export_signal_map,
)
sampled_spec = gt.SampledOutputSpec(
    name="sampled",
    fname="RLT.trn",
    index_name=SampledSignals.INDEX,
    name_map=sampled_signal_map,
)

root = Path("./runs/metrics")
root.mkdir(exist_ok=True, parents=True)
factory = gt.WorkspaceFactory(model_path="./gt_files/integrated.gtm", root=root)
# workspace = factory.build(export_spec=export_spec)

repo = gt.WorkspaceRepository(
    root=root, retention_policy=gt.workspace.KeepRecentPolicy(keep=5)
)
repo.refresh()
repo.apply_retention_policy()
workspace = repo.latest()
gt.clean_all_distributed_runs(repo)


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
reg.add_operating_point_parameter(opt.OperatingPointParameter(Op.RPM))

op_points = opt.OperatingPoints(**{Op.RPM: [1000, 2000, 3000]})

x0 = reg.pack()
x1 = np.vstack([x0, 1.01 * x0, 1.02 * x0, 1.03 * x0])
x1[1, 0] = 160  # Move int_valve_cl to cause wrapping
guard = gt.build_guard_design(reg, op_points)

popsize = op_points.size * x1.shape[0]
licenses = 3
packet_size = int(popsize / licenses) + 1


def simulate(expanded: gt.ExpandedCaseList) -> None:
    gt.DesignWriter().write_file(caselist=expanded, workspace=workspace)
    sim = gt.SerialExecution()
    sim = gt.DistributedExecution(
        licenses=expanded.packet_count,
        packet_size=expanded.packet_size,
    )
    _ = sim.run(workspace)
    gt.DataExporter().export(workspace=workspace, export_spec=export_spec)


reg.add_node(opt.ComputedObject(name="caselist", func=gt.CaseList.from_context))
reg.add_node(
    opt.ComputedObject(
        name="expanded",
        func=lambda caselist: gt.ExpandedCaseList.from_license_count(
            caselist, guard_design=guard, licenses=licenses
        ),
    )
)
reg.add_node(opt.ComputedObject(name="run_sim", func=simulate))
reg.add_node(
    opt.ComputedObject(
        name="sim_data",
        func=lambda expanded, run_sim: gt.ExportedOutputParser().read(
            spec=export_spec, workspace=workspace, caselist=expanded
        ),
    )
)
reg.add_node(
    opt.ComputedObject(
        name="sim_data_sampled",
        func=lambda expanded, run_sim: gt.SampledOutputParser().read(
            spec=sampled_spec, workspace=workspace, caselist=expanded
        ),
    )
)
int_qda_cols = [sig for sig in Signals if "qda_int" in sig]
exh_qda_cols = [sig for sig in Signals if "qda_exh" in sig]

qda_int = gt.results_sanitize.QDAResultsAlign(
    timing_param=Vars.INT_VALVE_CL,
    intake=True,
    start_of_cycle_angle=-60,
    signals=int_qda_cols,
)
qda_exh = gt.results_sanitize.QDAResultsAlign(
    timing_param=Vars.EXH_VALVE_CL,
    intake=False,
    start_of_cycle_angle=-60,
    signals=exh_qda_cols,
)
reg.add_node(
    opt.ComputedObject(
        name="case_data",
        func=lambda sim_data: [qda_int(qda_exh(case)) for case in sim_data.case_data],
    )
)
reg.add_node(
    opt.ComputedObject(
        name="case_data_sampled",
        func=lambda sim_data_sampled: sim_data_sampled.case_data,
    )
)

m = opt.AggregatedMetric(
    name="overlap",
    func=metrics.MultiCaseEvaluator(
        func=lambda signal: np.squeeze(signal),
        case_dependencies={
            "signal": Signal(SampledSignals.OVERLAP),
        },
        op_points_count=op_points.size,
    ),
    dependencies={"case_data": "case_data_sampled"},
    where=rs.Smallest(Op.RPM),
    agg=opt.aggregate.Squeeze(),
)

between_ramps_int = Window(Signal(Signals.INDEX), duration=Ref(Vars.INT_DURATION))
reg.add_node(m)

m1 = opt.AggregatedMetric(
    name="int_spring_margin",
    func=metrics.MultiCaseEvaluator(
        func=metrics.signal_min,
        case_dependencies={
            "signal": Signal(Signals.QDA_INT_SPRING_MARGIN, where=between_ramps_int),
        },
        op_points_count=op_points.size,
    ),
    dependencies={"case_data": "case_data"},
    where=rs.EqualTo(Op.RPM, 2000),
    agg=opt.aggregate.Squeeze(),
)

m2 = opt.AggregatedMetric(
    name="int_contact_pressure",
    func=metrics.MultiCaseEvaluator(
        func=metrics.signal_max,
        case_dependencies={
            "signal": Signal(
                Signals.QDA_INT_CONTACT_PRESSURE_IDLE, where=between_ramps_int
            ),
        },
        op_points_count=op_points.size,
    ),
    dependencies={"case_data": "case_data"},
    where=rs.EqualTo(Op.RPM, 2000),
    agg=opt.aggregate.Squeeze(),
)

reg.add_node(
    opt.ComputedQuantity(
        name="impulse_width",
        func=metrics.MultiCaseEvaluator(
            func=metrics.cam.impulse_width,
            case_dependencies={
                "theta1": Vars.INT_THETA1,
                "theta2": Vars.INT_THETA2,
                "theta3": Vars.INT_THETA3,
                "theta4": Vars.INT_THETA4,
            },
            op_points_count=op_points.size,
        ),
        dependencies={"case_data": "case_data"},
    )
)

m3 = opt.AggregatedMetric(
    name="stability",
    func=lambda impulse_width: metrics.cam.stability(
        impulse_width=impulse_width, natural_freq_hz=1000, max_rpm=6600
    ),
    where=rs.FirstOperatingPoint(),
    agg=opt.aggregate.Squeeze(),
)

reg.add_node(
    opt.ComputedQuantity(
        name="theta6",
        func=lambda t1, t2, t3, t4, t5, duration: np.asarray(
            duration / 2 - (t1 + t2 + t3 + t4 + t5)
        ).flatten(),
        dependencies={
            "t1": Vars.INT_THETA1,
            "t2": Vars.INT_THETA2,
            "t3": Vars.INT_THETA3,
            "t4": Vars.INT_THETA4,
            "t5": Vars.INT_THETA5,
            "duration": Vars.INT_DURATION,
        },
    )
)

reg.add_node(
    opt.AggregatedMetric(
        name="theta_6",
        func=lambda t1, t2, t3, t4, t5, duration: (
            duration / 2 - (t1 + t2 + t3 + t4 + t5)
        ),
        dependencies={
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

reg.add_node(m1)
reg.add_node(m2)
reg.add_node(m3)


m4 = opt.Metric(
    name="lubrication_number",
    func=metrics.MultiCaseEvaluator(
        func=metrics.cam.lubrication_number,
        case_dependencies={
            "angle": Signal(Signals.INDEX),
            "entrainment_velocity": Signal(Signals.INT_ENTRAINMENT_VELOCITY),
        },
        context_dependencies={"theta6": "theta6"},
        op_points_count=op_points.size,
    ),
    dependencies={"case_data": "case_data", "context": opt.resolvers.CONTEXT},
)

m5 = opt.Metric(
    name="lift_area",
    func=metrics.MultiCaseEvaluator(
        func=metrics.cam.lift_area,
        case_dependencies={
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
        func=metrics.cam.bucket_lateral_position_margin,
        case_dependencies={
            "position": Signal(Signals.INT_LATERAL_POSITION),
            "bucket_diameter": Literal(32),
        },
        op_points_count=op_points.size,
    ),
)

m7 = opt.Metric(
    name="min_neg_roc",
    func=metrics.MultiCaseEvaluator(
        func=metrics.cam.min_negative_radius_of_curvature,
        case_dependencies={
            "roc": Signal(Signals.INT_ROC),
        },
        op_points_count=op_points.size,
    ),
)

reg.add_node(
    opt.ComputedQuantity(
        name="scalar", func=lambda x: x, dependencies={"x": Literal(5)}
    )
)


ctx = reg.context_physical(x1, operating_points=op_points)

print(ScalarExpand("scalar", count=10).resolve(ctx))

print(ctx.int_contact_pressure)

print(m.evaluate(ctx))
print(m1.evaluate(ctx))
print(m2.evaluate(ctx))
print(m3.evaluate(ctx))
print(m4.evaluate(ctx))
print(m5.evaluate(ctx))
print(m6.evaluate(ctx))

print("\ntheta6 resolved:", theta6_int_case_resolver.resolve(ctx.case_data()[0]))


def plot_qda_alignment(
    int_col: str,
    exh_col: str,
    cases_before: list[gt.CaseData],
    cases_after: list[gt.CaseData],
):
    fig_exh, ax_exh = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig_int, ax_int = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for case0, case1 in zip(cases_before, cases_after):
        x = case0.get("angle")
        try:
            y0 = case0.get(int_col)
            y1 = case1.get(int_col)
            ax_int[0].plot(x, y0)
            ax_int[1].plot(x, y1)
        except gt.exceptions.MissingSignalError:
            pass

        try:
            y0 = case0.get(exh_col)
            y1 = case1.get(exh_col)
            ax_exh[0].plot(x, y0)
            ax_exh[1].plot(x, y1)
        except gt.exceptions.MissingSignalError:
            pass

    ax_exh[0].set_title(exh_col)
    ax_int[0].set_title(int_col)

    fig_int.tight_layout()
    fig_exh.tight_layout()


plot = True
# plot = False

if plot:
    for int_, exh in zip(int_qda_cols, exh_qda_cols):
        plot_qda_alignment(
            int_,
            exh,
            cases_before=ctx.sim_data().case_data,
            cases_after=ctx.case_data(),
        )


logging.shutdown()
