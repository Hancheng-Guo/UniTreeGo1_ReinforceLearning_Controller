import mujoco
import numpy as np
from collections import deque


# region | Constants

# contact rule
fatal_contact = ["trunk", "FR_hip", "FL_hip", "RR_hip", "RL_hip"]
# x_velocity_control = 0
idle_loop = [0b1111]
# # x_velocity_control ∈ (0, 2]
# walk_loop = [0b1110, 0b1010, 0b1011, 0b1101, 0b0101, 0b0111]
# x_velocity_control ∈ (0, 6]
trot_loop = [0b1001, 0b0110]
# x_velocity_control ∈ (6, 8]
canter_loop_A = [0b1110, 0b1000, 0b0000, 0b0001, 0b0111, 0b0110]
canter_loop_B = [0b1101, 0b0100, 0b0000, 0b0010, 0b1011, 0b1001]
# x_velocity_control ∈ (8, inf)
gallop_loop_A = [0b1000, 0b1100, 0b0100, 0b0000, 0b0010, 0b0011, 0b0001, 0b0000]
gallop_loop_B = [0b0100, 0b1100, 0b1000, 0b0000, 0b0001, 0b0011, 0b0010, 0b0000]

gait_loop_dict = {
    "idle": [idle_loop],
    "trot": [trot_loop],
    "canter": [canter_loop_A, canter_loop_B],
    "gallop": [gallop_loop_A, gallop_loop_B],
}

# endregion

# region | Functions

def get_rotation_matrix(quaternion):
    mat = np.zeros((9, 1)) # R00 R01 R02 R10 R11 R12 R20 R21 R22
    mujoco.mju_quat2Mat(mat, quaternion) # Convert quaternion to 3D rotation matrix
    return mat.reshape(3, 3)


def get_hinge_soft_upper_limit(rwd, limit_ratio):
    hinge_upper_bounder = np.array([jnt_range[1] for jnt_range in rwd.env.model.jnt_range[1:]])
    hinge_position0=rwd.env.model.key_qpos[0][7:19]
    hinge_upper_limit = (hinge_upper_bounder - hinge_position0) * limit_ratio + hinge_position0
    return hinge_upper_limit


def get_hinge_soft_lower_limit(rwd, limit_ratio):
    hinge_lower_bounder = np.array([jnt_range[0] for jnt_range in rwd.env.model.jnt_range[1:]])
    hinge_position0=rwd.env.model.key_qpos[0][7:19]
    hinge_lower_limit = hinge_position0 - (hinge_position0 - hinge_lower_bounder) * limit_ratio
    return hinge_lower_limit

# endrregion

# region | Rewards

def is_alive(rwd):
    if not np.isfinite(rwd.env.state_vector()).all():
        return False, {"is_alive": False}
    
    for c in rwd.env.data.contact:
        body1 = rwd.env.model.body(rwd.env.model.geom_bodyid[c.geom1]).name
        body2 = rwd.env.model.body(rwd.env.model.geom_bodyid[c.geom2]).name
        fatal_touching = ((body1 == "world" and body2 in fatal_contact) or
                          (body1 in fatal_contact and body2 == "world"))
        if fatal_touching:
            return False, {"is_alive": False}
    
    return True, {"is_alive": True}


def illegal_contact_l1(rwd):
    illegal_contact_l1 = 0.
    for c in rwd.env.data.contact:
        illegal_touching = (
            (c.geom1 not in rwd.env._foot_ids and c.geom2 == rwd.env._floor_id) or
            (c.geom1 == rwd.env._floor_id and c.geom2 not in rwd.env._foot_ids))
        illegal_contact_l1 += 1 if illegal_touching else 0
    
    info = {
        "illegal_contact_l1": illegal_contact_l1
    }
    return illegal_contact_l1, info


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


def z_velocity_l2(rwd):
    z_velocity = rwd.env.state_vector()[21]
    z_velocity_l2 = np.square(z_velocity)

    info = {
        "z_velocity": z_velocity,
        "z_velocity_l2": z_velocity_l2
    }
    return z_velocity_l2, info


def z_position_l2(rwd):
    z_position = rwd.env.data.body(rwd.env._main_body).xpos[2]
    z_position_l2 = np.square(z_position - rwd.z_position_target)

    info = {
        "z_position": z_position,
        "z_position_l2": z_position_l2
    }
    return z_position_l2, info


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


def action_change_l2(rwd):
    if rwd.env.action_old is None:
        return 0., {"action_change_l2": 0.}
    
    action_change = rwd.env.action - rwd.env.action_old
    action_change_l2 = np.mean(np.square(action_change))

    info = {
        "action_change_l2": action_change_l2
    }
    return action_change_l2, info


def hinge_angular_velocity_l2(rwd):
    hinge_angular_velocity = rwd.env.state_vector()[25:37]
    hinge_angular_velocity_l2 = np.mean(np.square(hinge_angular_velocity))

    info = {
        "hinge_angular_velocity_l2": hinge_angular_velocity_l2
    }
    return hinge_angular_velocity_l2, info


def hinge_position_l2(rwd):
    hinge_position = rwd.env.state_vector()[7:19]
    hinge_position_l2 = np.mean(np.square(
        (hinge_position - rwd.hinge_position0) / rwd.hinge_position_std))

    info = {
        "hinge_position_l2": hinge_position_l2
    }
    return hinge_position_l2, info


def hinge_exceed_limit_l1(rwd):
    hinge_position = rwd.env.state_vector()[7:19]
    hinge_exceed_upper_limit = np.clip(hinge_position - rwd.hinge_upper_limit, 0., None) 
    hinge_exceed_lower_limit = np.clip(rwd.hinge_lower_limit - hinge_position, 0., None)
    hinge_exceed_limit_l1 = np.sum([hinge_exceed_upper_limit, hinge_exceed_lower_limit])

    info = {
        "hinge_exceed_upper_limit": hinge_exceed_upper_limit,
        "hinge_exceed_lower_limit": hinge_exceed_lower_limit,
        "hinge_exceed_limit_l1": hinge_exceed_limit_l1
    }
    return hinge_exceed_limit_l1, info


def hinge_energy_l1(rwd):
    hinge_force = rwd.env.data.qfrc_actuator[6:19]
    hinge_angular_velocity = rwd.env.state_vector()[25:37]
    hinge_energy_l1 = np.sum(np.abs(hinge_force * hinge_angular_velocity))
    
    info = {
        "hinge_energy_l1": hinge_energy_l1
    }
    return hinge_energy_l1, info


def gait_loop_duration_tanh(rwd):
    info = {}

    # get legal gait type
    velocity_control = np.linalg.norm(rwd.env.control_vector[0:2])
    if velocity_control > 8:
        gait_target = "gallop"
    elif velocity_control > 6:
        gait_target = "canter"
    elif velocity_control > 0:
        gait_target = "trot"
    else:
        gait_target = "idle"
    info["gait_target"] = gait_target

    # get current feet_state 
    feet_state = rwd.env._feet_state
    info["feet_state"] = bin(feet_state)

    # get/delete gait_loop_options
    if gait_target == rwd.gait_type and len(rwd.gait_loop_options) > 0: # loop continue and has legal loop
        for i in range(len(rwd.gait_loop_options) - 1, -1, -1):
            gait_loop_option = rwd.gait_loop_options[i]
            if gait_loop_option[0] == feet_state: # feet_state continue
                continue
            elif gait_loop_option[1 % len(gait_loop_option)] == feet_state: # feet_state go on
                gait_loop_option.append(gait_loop_option.popleft())
            else:
                rwd.gait_loop_options.pop(i) # delete illegal loop 
    else: # loop change or loop continue but hasn't legal loop
        rwd.gait_type = gait_target
        # get new gait_loop_options
        rwd.gait_loop_options = []
        for gait_loop in gait_loop_dict[gait_target]: # filt legal loop and add to gait_loop_options
            for i, gait_loop_state in enumerate(gait_loop):
                if feet_state == gait_loop_state:
                    rwd.gait_loop_options.append(deque(gait_loop[i:] + gait_loop[:i],
                                                       maxlen=len(gait_loop)))

    # get next gait_loop_option and update gait_loop_duration
    if len(rwd.gait_loop_options) > 0: # have legal loop
        next_gait_option = [gait_loop_option[1 % len(gait_loop_option)]
                            for gait_loop_option in rwd.gait_loop_options]
        rwd.gait_loop_duration += 1
        info["in_gait_loop"] = True
    else: # it isn't in a legal loop
        next_gait_option = []
        rwd.gait_loop_duration = 0
        info["in_gait_loop"] = False
    info["next_gait_option"] = next_gait_option
    info["gait_loop_duration"] = rwd.gait_loop_duration

    # calculate reward
    gait_loop_duration_tanh = np.tanh(rwd.gait_loop_k * rwd.gait_loop_duration)
    info["gait_loop_duration_tanh"] = gait_loop_duration_tanh

    return gait_loop_duration_tanh, info


def feet_state_duration_exp(rwd):
    if rwd.env._feet_state == rwd.feet_state_old:
        rwd.feet_state_duration += 1 
    else:
        rwd.feet_state_duration = 0
    rwd.feet_state_old = rwd.env._feet_state

    if rwd.env.reward.reward_info["in_gait_loop"]:
        velocity_control = np.linalg.norm(rwd.env.control_vector[0:2])
        feet_state_duration_exp = np.exp(-velocity_control * rwd.feet_state_k * rwd.feet_state_duration)
    else:
        feet_state_duration_exp = 0.
    
    info = {
        "feet_state_duration": rwd.feet_state_duration,
        "feet_state_duration_exp": feet_state_duration_exp
    }
    return feet_state_duration_exp, info


def foot_sliding_velocity_l2(rwd):
    are_feet_landed = np.array(rwd.env._are_feet_touching_ground)
    foot_sliding_velocity = np.zeros(len(are_feet_landed))
    for i, foot_id in enumerate(rwd.env._foot_ids):
        if are_feet_landed[i]:
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(rwd.env.model, rwd.env.data, mujoco.mjtObj.mjOBJ_GEOM,
                                     foot_id, vel, 0)
            foot_sliding_velocity[i] = np.sqrt(np.sum(np.square(vel[3:5])))
    foot_sliding_velocity_l2 = np.mean(np.square(foot_sliding_velocity))

    info = {
        "foot_sliding_velocity": foot_sliding_velocity,
        "foot_sliding_velocity_l2": foot_sliding_velocity_l2
    }
    return foot_sliding_velocity_l2, info


def foot_lift_height_l2_exp_xy_vel_weighted(rwd):
    if rwd.env.reward.reward_info["in_gait_loop"]:
        are_feet_lifted = np.logical_not(np.array(rwd.env._are_feet_touching_ground))
        lifted_foot_ids = [foot_id for foot_id in np.array(rwd.env._foot_ids)[are_feet_lifted]]
        if len(lifted_foot_ids) == 0:
            return 0., {"foot_lift_xy_velocity": [],
                        "foot_lift_height_l2_exp_xy_vel_weighted": 0.}
        
        foot_lift_height = rwd.env.data.geom_xpos[lifted_foot_ids][:,2]
        foot_lift_height_l2 = np.square(foot_lift_height - rwd.foot_lift_height_target)
        foot_lift_height_l2_exp = np.exp(-foot_lift_height_l2)

        foot_xy_velocity = np.zeros(len(lifted_foot_ids))
        for i, foot_id in enumerate(lifted_foot_ids):
            if are_feet_lifted[i]:
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(rwd.env.model, rwd.env.data, mujoco.mjtObj.mjOBJ_GEOM,
                                        foot_id, vel, 0)
                foot_xy_velocity[i] = np.sqrt(np.sum(np.square(vel[3:5])))

        foot_lift_height_l2_exp_xy_vel_weighted = np.mean(foot_lift_height_l2_exp * foot_xy_velocity)

        info = {
            "foot_lift_xy_velocity": foot_xy_velocity,
            "foot_lift_height_l2_exp_xy_vel_weighted": foot_lift_height_l2_exp_xy_vel_weighted
        }
        return foot_lift_height_l2_exp_xy_vel_weighted, info

    return 0., {"foot_lift_xy_velocity": [],
                "foot_lift_height_l2_exp_xy_vel_weighted": 0.}

# endregion