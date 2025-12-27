from src.reward.common.new_reward import NewReward
from src.reward.common.get_hinge_soft_limit import get_hinge_soft_lower_limit, get_hinge_soft_upper_limit
from src.reward.common.get_foot_state import are_foot_touching_ground, get_foot_state
from src.reward.alive import is_alive
from src.reward.contact import illegal_contact_l1
from src.reward.track import robot_xy_velocity_l2_exp, z_angular_velocity_l2_exp
from src.reward.constraint import z_velocity_l2, z_position_l2, xy_angular_velocity_l2, xy_angular_gravity_projection
from src.reward.action import action_change_l2
from src.reward.hinge import hinge_angular_velocity_l2, hinge_position_l2, hinge_exceed_limit_l1, hinge_energy_l1
from src.reward.gait import gait_loop_duration_tanh
from src.reward.foot import foot_state_duration_exp, foot_sliding_velocity_l2, foot_lift_height_l2_exp_xy_vel_weighted


__all__ = [
    "NewReward",

    "get_hinge_soft_lower_limit",
    "get_hinge_soft_upper_limit",
    "are_foot_touching_ground",
    "get_foot_state",

    "is_alive",

    "illegal_contact_l1",

    "robot_xy_velocity_l2_exp",
    "z_angular_velocity_l2_exp",

    "z_velocity_l2",
    "z_position_l2",
    "xy_angular_velocity_l2",
    "xy_angular_gravity_projection",

    "action_change_l2",

    "hinge_angular_velocity_l2",
    "hinge_position_l2",
    "hinge_exceed_limit_l1",
    "hinge_energy_l1",

    "gait_loop_duration_tanh", # call it before foot_related rewards

    "foot_state_duration_exp",
    "foot_sliding_velocity_l2",
    "foot_lift_height_l2_exp_xy_vel_weighted",
    ]