from src.reward.common.new_reward import NewReward
from src.reward.common.get_hinge_soft_limit import get_hinge_soft_lower_limit, get_hinge_soft_upper_limit
from src.reward.common.get_foot_state import are_foot_touching_ground, get_foot_state
from src.reward.alive import is_alive, is_fatal
from src.reward.contact import illegal_contact_l1
from src.reward.track import robot_xy_velocity_l2_exp, z_angular_velocity_l2_exp
from src.reward.track import robot_xy_velocity_rbf_logcosh, z_angular_velocity_rbf_logcosh
# from src.reward.track import xy_velocity_error_abs_diff_clip, z_angular_velocity_error_abs_diff_clip
from src.reward.track import xy_velocity_error_integral_l2, z_angular_velocity_error_integral_l2
from src.reward.constraint import z_velocity_l2_xy_vel_weighted, z_position_l2_xy_vel_weighted, xy_angular_velocity_l2, xy_angular_gravity_projection
from src.reward.constraint import z_velocity_l2, z_position_l2
from src.reward.action import action_change_l2
from src.reward.hinge import hinge_angular_velocity_l2, hinge_position_l2, hinge_exceed_limit_l1, hinge_energy_l1
from src.reward.gait import gait_loop_duration_tanh, trot_loop_duration_tanh, gait_loop_duration_tanh_mode_weighted, gait_transfer, trot_sync
from src.reward.gait import command_to_gait_name, gait_loop_dict
from src.reward.foot import foot_state_duration_exp, foot_state_duration_exp3, foot_velocity_variance
from src.reward.foot import foot_sliding_velocity_l2, foot_lift_height_l2_exp_xy_vel_weighted, foot_lift_height_l2_xy_vel_weighted_exp, foot_contact_without_cmd


__all__ = [
    "NewReward",

    "get_hinge_soft_lower_limit",
    "get_hinge_soft_upper_limit",
    "are_foot_touching_ground",
    "get_foot_state",
    # "speed_to_gait_index",
    "command_to_gait_name",
    "gait_loop_dict",

    "is_alive",
    "is_fatal",

    "illegal_contact_l1",

    "robot_xy_velocity_l2_exp",
    "z_angular_velocity_l2_exp",
    "robot_xy_velocity_rbf_logcosh",
    "z_angular_velocity_rbf_logcosh",
    # "xy_velocity_error_abs_diff_clip",
    # "z_angular_velocity_error_abs_diff_clip",
    "xy_velocity_error_integral_l2",
    "z_angular_velocity_error_integral_l2",

    "z_velocity_l2",
    "z_position_l2",
    "z_velocity_l2_xy_vel_weighted",
    "z_position_l2_xy_vel_weighted",
    "xy_angular_velocity_l2",
    "xy_angular_gravity_projection",

    "action_change_l2",

    "hinge_angular_velocity_l2",
    "hinge_position_l2",
    "hinge_exceed_limit_l1",
    "hinge_energy_l1",

    "gait_loop_duration_tanh",
    "gait_loop_duration_tanh_mode_weighted",
    "trot_loop_duration_tanh",
    "gait_transfer",
    "trot_sync",

    "foot_state_duration_exp", # need to call gait_loop_duration before
    "foot_state_duration_exp3", # need to call gait_loop_duration before
    "foot_velocity_variance",
    "foot_sliding_velocity_l2",
    "foot_lift_height_l2_exp_xy_vel_weighted",
    "foot_lift_height_l2_xy_vel_weighted_exp",
    "foot_contact_without_cmd",
    
    ]