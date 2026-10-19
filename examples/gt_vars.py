from enum import StrEnum


class Signals(StrEnum):
    INDEX = "angle"
    INT_LIFT = "int_lift"
    INT_VELOCITY = "int_velocity"
    INT_ACCEL = "int_acceleration"
    INT_JERK = "int_jerk"
    INT_ROC = "int_roc_knife_edge"
    INT_ROC_EFFECTIVE = "int_roc_effective"
    INT_LATERAL_POSITION = "int_cam_follower_contact_lateral_position"
    INT_ENTRAINMENT_VELOCITY = "int_entrainment_velocity"
    QDA_INT_SPRING_MARGIN = "qda_int_spring_margin_max_speed"
    QDA_INT_CONTACT_PRESSURE_IDLE = "qda_int_contact_pressure_idle"
    QDA_INT_CONTACT_PRESSURE_MAX_SPEED = "qda_int_contact_pressure_max_speed"
    QDA_INT_CONTACT_FORCE_IDLE = "qda_int_contact_force_idle"
    QDA_INT_CONTACT_FORCE_MAX_SPEED = "qda_int_contact_force_max_speed"
    EXH_LIFT = "exh_lift"
    EXH_VELOCITY = "exh_velocity"
    EXH_ACCELERATION = "exh_acceleration"
    EXH_JERK = "exh_jerk"
    EXH_ROC_KNIFE_EDGE = "exh_roc_knife_edge"
    EXH_ROC_EFFECTIVE = "exh_roc_effective"
    EXH_CAM_FOLLOWER_CONTACT_LATERAL_POSITION = (
        "exh_cam_follower_contact_lateral_position"
    )
    EXH_ENTRAINMENT_VELOCITY = "exh_entrainment_velocity"
    QDA_EXH_SPRING_MARGIN = "qda_exh_spring_margin_max_speed"
    QDA_EXH_CONTACT_PRESSURE_IDLE = "qda_exh_contact_pressure_idle"
    QDA_EXH_CONTACT_PRESSURE_MAX_SPEED = "qda_exh_contact_pressure_max_speed"
    QDA_EXH_CONTACT_FORCE_IDLE = "qda_exh_contact_force_idle"
    QDA_EXH_CONTACT_FORCE_MAX_SPEED = "qda_exh_contact_force_max_speed"
    INT_BUCKET_CONTACT_LATERAL_POSITION = "int_bucket_contact_lateral_position"
    EXH_BUCKET_CONTACT_LATERAL_POSITION = "exh_bucket_contact_lateral_position"
    QDA_INT_CAM_TORQUE_HIGH_QDA_SPEED = "qda_int_cam_torque_high_qda_speed"
    QDA_EXH_CAM_TORQUE_HIGH_QDA_SPEED = "qda_exh_cam_torque_high_qda_speed"


class SampledSignals(StrEnum):
    INDEX = "count"
    TORQUE = "torque"
    POWER = "power"
    OVERLAP = "overlap"


class Vars(StrEnum):
    INT_VALVE_CL = "int_valve_cl"
    INT_LIFT_MAX = "int_lift_max"
    INT_DURATION = "int_duration"
    INT_THETA1 = "int_theta1"
    INT_THETA2 = "int_theta2"
    INT_THETA3 = "int_theta3"
    INT_THETA4 = "int_theta4"
    INT_THETA5 = "int_theta5"
    INT_F1 = "int_f1"
    INT_F3 = "int_f3"
    INT_F4 = "int_f4"
    EXH_VALVE_CL = "exh_valve_cl"
    EXH_LIFT_MAX = "exh_lift_max"
    EXH_DURATION = "exh_duration"
    EXH_THETA1 = "exh_theta1"
    EXH_THETA2 = "exh_theta2"
    EXH_THETA3 = "exh_theta3"
    EXH_THETA4 = "exh_theta4"
    EXH_THETA5 = "exh_theta5"
    EXH_F1 = "exh_f1"
    EXH_F3 = "exh_f3"
    EXH_F4 = "exh_f4"


class Op(StrEnum):
    RPM = "erpm"


export_signal_map = {
    "int_lift": Signals.INT_LIFT,
    "int_velocity": Signals.INT_VELOCITY,
    "int_acceleration": Signals.INT_ACCEL,
    "int_jerk": Signals.INT_JERK,
    "int_roc_knife_edge": Signals.INT_ROC,
    "int_roc_effective": Signals.INT_ROC_EFFECTIVE,
    "int_cam_follower_contact_lateral_position": Signals.INT_LATERAL_POSITION,
    "int_entrainment_velocity": Signals.INT_ENTRAINMENT_VELOCITY,
    "qda_int_spring_margin_max_speed": Signals.QDA_INT_SPRING_MARGIN,
    "qda_int_contact_pressure_idle": Signals.QDA_INT_CONTACT_PRESSURE_IDLE,
    "qda_int_contact_pressure_max_speed": Signals.QDA_INT_CONTACT_PRESSURE_MAX_SPEED,
    "qda_int_contact_force_idle": Signals.QDA_INT_CONTACT_FORCE_IDLE,
    "qda_int_contact_force_max_speed": Signals.QDA_INT_CONTACT_FORCE_MAX_SPEED,
    "exh_lift": Signals.EXH_LIFT,
    "exh_velocity": Signals.EXH_VELOCITY,
    "exh_acceleration": Signals.EXH_ACCELERATION,
    "exh_jerk": Signals.EXH_JERK,
    "exh_roc_knife_edge": Signals.EXH_ROC_KNIFE_EDGE,
    "exh_roc_effective": Signals.EXH_ROC_EFFECTIVE,
    "exh_cam_follower_contact_lateral_position": Signals.EXH_CAM_FOLLOWER_CONTACT_LATERAL_POSITION,
    "exh_entrainment_velocity": Signals.EXH_ENTRAINMENT_VELOCITY,
    "qda_exh_spring_margin_max_speed": Signals.QDA_EXH_SPRING_MARGIN,
    "qda_exh_contact_pressure_idle": Signals.QDA_EXH_CONTACT_PRESSURE_IDLE,
    "qda_exh_contact_pressure_max_speed": Signals.QDA_EXH_CONTACT_PRESSURE_MAX_SPEED,
    "qda_exh_contact_force_idle": Signals.QDA_EXH_CONTACT_FORCE_IDLE,
    "qda_exh_contact_force_max_speed": Signals.QDA_EXH_CONTACT_FORCE_MAX_SPEED,
    "int_bucket_contact_lateral_position": Signals.INT_BUCKET_CONTACT_LATERAL_POSITION,
    "exh_bucket_contact_lateral_position": Signals.EXH_BUCKET_CONTACT_LATERAL_POSITION,
    "qda_int_cam_torque_high_qda_speed": Signals.QDA_INT_CAM_TORQUE_HIGH_QDA_SPEED,
    "qda_exh_cam_torque_high_qda_speed": Signals.QDA_EXH_CAM_TORQUE_HIGH_QDA_SPEED,
}

sampled_signal_map = {
    "Brake_Torque": SampledSignals.TORQUE,
    "Brake_Power": SampledSignals.POWER,
    "Overlap_Factor": SampledSignals.OVERLAP,
}
