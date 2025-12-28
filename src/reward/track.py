import numpy as np

from src.reward.common.get_rotation_matrix import get_rotation_matrix


def robot_xy_velocity_l2_exp(rwd):
    R_rotate = get_rotation_matrix(rwd.env.state_vector()[3:7])
    y_velocity = rwd.env.state_vector()[20]
    x_velocity = rwd.env.state_vector()[19]
    robot_velocity = R_rotate.T @ np.array([x_velocity, y_velocity, 0.])

    robot_x_velocity = robot_velocity[0]
    robot_x_velocity_target = rwd.env.control_vector[0]
    robot_x_velocity_l2 = np.square(robot_x_velocity - robot_x_velocity_target)
    robot_x_velocity_std = max(1, rwd.env.controller.controller.schedule[0]["amp"][int(rwd.env.stage)])
    robot_x_velocity_l2_exp = np.exp(-robot_x_velocity_l2 / (robot_x_velocity_std**2))

    robot_y_velocity = robot_velocity[1]
    robot_y_velocity_target = rwd.env.control_vector[1]
    robot_y_velocity_l2 = np.square(robot_y_velocity - robot_y_velocity_target)
    robot_y_velocity_std = max(1, rwd.env.controller.controller.schedule[1]["amp"][int(rwd.env.stage)])
    robot_y_velocity_l2_exp = np.exp(-robot_y_velocity_l2 / (robot_y_velocity_std**2))

    info = {
        "robot_x_velocity": robot_x_velocity,
        "robot_x_velocity_target": robot_x_velocity_target,
        "robot_x_velocity_l2": robot_x_velocity_l2,
        "robot_x_velocity_std": robot_x_velocity_std,
        "robot_x_velocity_l2_exp": robot_x_velocity_l2_exp,
        "robot_y_velocity": robot_y_velocity,
        "robot_y_velocity_target": robot_y_velocity_target,
        "robot_y_velocity_l2": robot_y_velocity_l2,
        "robot_y_velocity_std": robot_y_velocity_std,
        "robot_y_velocity_l2_exp": robot_y_velocity_l2_exp
    }
    return np.mean([robot_x_velocity_l2_exp + robot_y_velocity_l2_exp]), info


def z_angular_velocity_l2_exp(rwd):
    z_angular_velocity = rwd.env.state_vector()[24]
    z_angular_velocity_target = rwd.env.control_vector[2]
    z_angular_velocity_l2 = np.square(z_angular_velocity - z_angular_velocity_target)
    z_angular_velocity_std = max(1, rwd.env.controller.controller.schedule[2]["amp"][int(rwd.env.stage)])
    z_angular_velocity_l2_exp = np.exp(-z_angular_velocity_l2 / (z_angular_velocity_std**2))

    info = {
        "z_angular_velocity": z_angular_velocity,
        "z_angular_velocity_target": z_angular_velocity_target,
        "z_angular_velocity_l2": z_angular_velocity_l2,
        "z_angular_velocity_std": z_angular_velocity_std,
        "z_angular_velocity_l2_exp": z_angular_velocity_l2_exp
    }
    return z_angular_velocity_l2_exp, info