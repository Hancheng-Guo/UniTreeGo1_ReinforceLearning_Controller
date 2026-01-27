import numpy as np
from src.reward.common.get_rotation_matrix import get_rotation_matrix


def z_velocity_l2(rwd):
    z_velocity = rwd.env.state_vector()[21]
    z_velocity_l2 = np.square(z_velocity)

    info = {
        "z_velocity": z_velocity,
        "z_velocity_l2": z_velocity_l2
    }
    return z_velocity_l2, info


def z_velocity_l2_xy_vel_weighted(rwd):
    z_velocity = rwd.env.state_vector()[21]
    z_velocity_l2 = np.square(z_velocity)
    xy_velocity = np.linalg.norm(rwd.env.state_vector()[19:21])
    z_velocity_l2_xy_vel_weighted = z_velocity_l2 / max(0.5, xy_velocity)

    info = {
        "z_velocity": z_velocity,
        "z_velocity_l2": z_velocity_l2,
        "z_velocity_l2_xy_vel_weighted": z_velocity_l2_xy_vel_weighted
    }
    return z_velocity_l2_xy_vel_weighted, info


def z_position_l2(rwd):
    z_position = rwd.env.data.body(rwd.env._main_body).xpos[2]
    z_position_l2 = np.square(z_position - rwd.z_position_target)

    info = {
        "z_position": z_position,
        "z_position_l2": z_position_l2
    }
    return z_position_l2, info


def z_position_l2_xy_vel_weighted(rwd):
    z_position = rwd.env.data.body(rwd.env._main_body).xpos[2]
    z_position_l2 = np.square(z_position - rwd.z_position_target)
    xy_velocity = np.linalg.norm(rwd.env.state_vector()[19:21])
    z_position_l2_xy_vel_weighted = z_position_l2 / max(0.5, xy_velocity)

    info = {
        "z_position": z_position,
        "z_position_l2": z_position_l2,
        "z_position_l2_xy_vel_weighted": z_position_l2_xy_vel_weighted
    }
    return z_position_l2_xy_vel_weighted, info


def xy_angular_velocity_l2(rwd):
    x_angular_velocity = rwd.env.state_vector()[22]
    y_angular_velocity = rwd.env.state_vector()[23]
    xy_angular_velocity_l2 = np.mean(np.square([x_angular_velocity, y_angular_velocity]))

    info = {
        "x_angular_velocity": x_angular_velocity,
        "y_angular_velocity": y_angular_velocity,
        "xy_angular_velocity_l2": xy_angular_velocity_l2
    }
    return xy_angular_velocity_l2, info


def xy_angular_gravity_projection(rwd):
    R_rotate = get_rotation_matrix(rwd.env.state_vector()[3:7])
    z_unit_vector = R_rotate[:,2]
    g_vector = rwd.env.model.opt.gravity
    g_z = np.dot(z_unit_vector, g_vector)
    g_xoy = np.sqrt(max(np.sum(np.square(g_vector)) - np.square(g_z), 0.))
    g_xoy_norm = g_xoy / np.linalg.norm(g_vector)
    
    info = {
        "g_z": g_z,
        "g_xoy": g_xoy,
        "g_xoy_norm": g_xoy_norm
    }
    return g_xoy_norm, info
