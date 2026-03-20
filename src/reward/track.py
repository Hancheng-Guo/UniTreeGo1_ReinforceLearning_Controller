import numpy as np


def robot_xy_velocity_l2_exp(rwd):
    robot_x_velocity = rwd.env.envdata.vel_vec[0]
    robot_x_velocity_target = rwd.env.envdata.previer_cmd_vec[0]
    robot_x_velocity_l2 = np.square(robot_x_velocity - robot_x_velocity_target)
    robot_x_velocity_std = rwd.robot_x_velocity_std if hasattr(rwd, "robot_x_velocity_std") else 1.
    robot_x_velocity_l2_exp = np.exp(-robot_x_velocity_l2 / (robot_x_velocity_std**2))

    robot_y_velocity = rwd.env.envdata.vel_vec[1]
    robot_y_velocity_target = rwd.env.envdata.previer_cmd_vec[1]
    robot_y_velocity_l2 = np.square(robot_y_velocity - robot_y_velocity_target)
    robot_y_velocity_std = rwd.robot_y_velocity_std if hasattr(rwd, "robot_y_velocity_std") else 1.
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
    return np.sum([robot_x_velocity_l2_exp + robot_y_velocity_l2_exp]), info


def robot_xy_velocity_rbf_logcosh(rwd):
    robot_x_velocity = rwd.env.envdata.vel_vec[0]
    robot_x_velocity_target = rwd.env.envdata.previer_cmd_vec[0]
    robot_x_velocity_std = rwd.robot_x_velocity_std if hasattr(rwd, "robot_x_velocity_std") else 1.
    robot_x_velocity_rbf = np.exp(-np.square(robot_x_velocity - robot_x_velocity_target) / (robot_x_velocity_std**2))
    robot_x_velocity_logcosh = 1 - np.log(np.cosh(2 * (robot_x_velocity - robot_x_velocity_target) / robot_x_velocity_std))
    robot_x_velocity_rbf_logcosh = rwd.rbf_k * robot_x_velocity_rbf + (1 - rwd.rbf_k) * robot_x_velocity_logcosh

    robot_y_velocity = rwd.env.envdata.vel_vec[1]
    robot_y_velocity_target = rwd.env.envdata.previer_cmd_vec[1]
    robot_y_velocity_std = rwd.robot_y_velocity_std if hasattr(rwd, "robot_y_velocity_std") else 1.
    robot_y_velocity_rbf = np.exp(-np.square(robot_y_velocity - robot_y_velocity_target) / (robot_y_velocity_std**2))
    robot_y_velocity_logcosh = 1 - np.log(np.cosh(2 * (robot_y_velocity - robot_y_velocity_target) / robot_y_velocity_std))
    robot_y_velocity_rbf_logcosh = rwd.rbf_k * robot_y_velocity_rbf + (1 - rwd.rbf_k) * robot_y_velocity_logcosh

    info = {
        "robot_x_velocity": robot_x_velocity,
        "robot_x_velocity_target": robot_x_velocity_target,
        "robot_x_velocity_std": robot_x_velocity_std,
        "robot_x_velocity_rbf": robot_x_velocity_rbf,
        "robot_x_velocity_logcosh": robot_x_velocity_logcosh,
        "robot_x_velocity_rbf_logcosh": robot_x_velocity_rbf_logcosh,

        "robot_y_velocity": robot_y_velocity,
        "robot_y_velocity_target": robot_y_velocity_target,
        "robot_y_velocity_std": robot_y_velocity_std,
        "robot_y_velocity_rbf": robot_y_velocity_rbf,
        "robot_y_velocity_logcosh": robot_y_velocity_logcosh,
        "robot_y_velocity_rbf_logcosh": robot_y_velocity_rbf_logcosh
    }
    return np.sum([robot_x_velocity_rbf_logcosh, robot_y_velocity_rbf_logcosh]), info


def xy_velocity_error_integral_l2(rwd):
    xy_velocity = np.linalg.norm(rwd.env.envdata.vel_vec[0:2])
    xy_velocity_target = np.linalg.norm(rwd.env.envdata.previer_cmd_vec[0:2])
    vel_error = xy_velocity_target - xy_velocity
    rwd.vel_error.append(vel_error)
    vel_error_integral = np.sum(rwd.vel_error)
    vel_error_integral_l2 = np.square(vel_error_integral)

    info = {
        "xy_velocity_error": vel_error,
        "xy_velocity_error_integral": vel_error_integral,
        "xy_velocity_error_integral_l2": vel_error_integral_l2,
    }
    return vel_error_integral_l2, info


def z_angular_velocity_l2_exp(rwd):
    z_angular_velocity = rwd.env.envdata.vel_vec[2]
    z_angular_velocity_target = rwd.env.envdata.previer_cmd_vec[2]
    z_angular_velocity_l2 = np.square(z_angular_velocity - z_angular_velocity_target)
    z_angular_velocity_std = rwd.z_angular_velocity_std if hasattr(rwd, "z_angular_velocity_std") else 1.
    z_angular_velocity_l2_exp = np.exp(-z_angular_velocity_l2 / (2 * np.pi) / (z_angular_velocity_std**2))

    info = {
        "z_angular_velocity": z_angular_velocity,
        "z_angular_velocity_target": z_angular_velocity_target,
        "z_angular_velocity_std": z_angular_velocity_std,
        "z_angular_velocity_l2": z_angular_velocity_l2,
        "z_angular_velocity_l2_exp": z_angular_velocity_l2_exp
    }
    return z_angular_velocity_l2_exp, info


def z_angular_velocity_rbf_logcosh(rwd):
    z_angular_velocity = rwd.env.envdata.vel_vec[2]
    z_angular_velocity_target = rwd.env.envdata.previer_cmd_vec[2]
    z_angular_velocity_std = rwd.z_angular_velocity_std if hasattr(rwd, "z_angular_velocity_std") else 1.
    z_angular_velocity_rbf = np.exp(-np.square(z_angular_velocity - z_angular_velocity_target)
                                    / (2 * np.pi) / (z_angular_velocity_std**2))
    z_angular_velocity_logcosh = 1 - np.log(np.cosh(2 * (z_angular_velocity - z_angular_velocity_target)
                                                    / np.sqrt(2 * np.pi) / z_angular_velocity_std))
    z_angular_velocity_rbf_logcosh = rwd.rbf_k * z_angular_velocity_rbf + (1 - rwd.rbf_k) * z_angular_velocity_logcosh

    info = {
        "z_angular_velocity": z_angular_velocity,
        "z_angular_velocity_target": z_angular_velocity_target,
        "z_angular_velocity_std": z_angular_velocity_std,
        "z_angular_velocity_rbf": z_angular_velocity_rbf,
        "z_angular_velocity_logcosh": z_angular_velocity_logcosh,
        "z_angular_velocity_rbf_logcosh": z_angular_velocity_rbf_logcosh,
    }
    return z_angular_velocity_logcosh, info



def z_angular_velocity_error_integral_l2(rwd):
    z_angular_velocity = rwd.env.envdata.vel_vec[2]
    z_angular_velocity_target = rwd.env.envdata.previer_cmd_vec[2]
    angular_vel_error = z_angular_velocity_target - z_angular_velocity
    rwd.angular_vel_error.append(angular_vel_error)
    angular_vel_error_integral = np.sum(rwd.angular_vel_error)
    angular_vel_error_integral_l2 = np.square(angular_vel_error_integral)

    info = {
        "z_angular_velocity_error": angular_vel_error,
        "z_angular_velocity_error_integral": angular_vel_error_integral,
        "z_angular_velocity_error_integral_l2": angular_vel_error_integral_l2,
    }
    return angular_vel_error_integral_l2, info

