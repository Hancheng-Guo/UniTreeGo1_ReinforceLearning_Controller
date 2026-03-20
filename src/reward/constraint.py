import numpy as np
from src.reward.common.get_rotation_matrix import get_rotation_matrix


def z_velocity_l2(rwd):
    z_velocity = rwd.env.data.qvel[2]
    z_velocity_l2 = np.square(z_velocity)

    info = {
        "z_velocity": z_velocity,
        "z_velocity_l2": z_velocity_l2
    }
    return z_velocity_l2, info


def z_velocity_l2_xy_vel_weighted(rwd):
    z_velocity = rwd.env.data.qvel[2]
    z_velocity_l2 = np.square(z_velocity)
    xy_velocity = np.linalg.norm(rwd.env.data.qvel[:2])
    z_velocity_l2_xy_vel_weighted = z_velocity_l2 / max(0.5, xy_velocity)

    info = {
        "z_velocity": z_velocity,
        "z_velocity_l2": z_velocity_l2,
        "z_velocity_l2_xy_vel_weighted": z_velocity_l2_xy_vel_weighted
    }
    return z_velocity_l2_xy_vel_weighted, info


def z_position_l2(rwd):
    z_position = rwd.env.data.qpos[2]
    z_position_l2 = np.square(z_position - rwd.z_position_target)

    info = {
        "z_position": z_position,
        "z_position_l2": z_position_l2
    }
    return z_position_l2, info


def z_position_l2_xy_vel_weighted(rwd):
    z_position = rwd.env.data.qpos[2]
    z_position_l2 = np.square(z_position - rwd.z_position_target)
    xy_velocity = np.linalg.norm(rwd.env.data.qvel[:2])
    z_position_l2_xy_vel_weighted = z_position_l2 / max(0.5, xy_velocity)

    info = {
        "z_position": z_position,
        "z_position_l2": z_position_l2,
        "z_position_l2_xy_vel_weighted": z_position_l2_xy_vel_weighted
    }
    return z_position_l2_xy_vel_weighted, info


def xy_angular_velocity_l2(rwd):
    x_angular_velocity = rwd.env.data.qvel[3]
    y_angular_velocity = rwd.env.data.qvel[4]
    xy_angular_velocity_l2 = np.mean(np.square([x_angular_velocity, y_angular_velocity]))

    info = {
        "x_angular_velocity": x_angular_velocity,
        "y_angular_velocity": y_angular_velocity,
        "xy_angular_velocity_l2": xy_angular_velocity_l2
    }
    return xy_angular_velocity_l2, info


def xy_angular_gravity_projection(rwd):
    gravity_projection = rwd.env.envdata.gravity_projection
    
    info = {
        "gravity_projection": gravity_projection,
    }
    return gravity_projection, info
