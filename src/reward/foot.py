import mujoco
import numpy as np
from src.reward.gait import command_to_gait_name

def foot_state_duration_exp(rwd):
    if rwd.env.envdata.foot_state == rwd.foot_state_old:
        rwd.foot_state_duration += rwd.env.dt
    else:
        rwd.foot_state_duration = 0.
    rwd.foot_state_old = rwd.env.envdata.foot_state

    foot_state_duration_exp = (
        (rwd.env.reward.rewards["gait_loop"].gait_loop_duration > 0) * 
        np.exp(-rwd.env.envdata.previer_cmd_vec_norm *
               (rwd.foot_state_duration / rwd.foot_state_sigma)))
    
    info = {
        "foot_state_duration": rwd.foot_state_duration,
        "foot_state_duration_exp": foot_state_duration_exp
    }
    return foot_state_duration_exp, info


def foot_state_duration_exp3(rwd):
    if rwd.env.envdata.foot_state == rwd.foot_state_old:
        rwd.foot_state_duration += rwd.env.dt
    else:
        rwd.foot_state_duration = 0.
    rwd.foot_state_old = rwd.env.envdata.foot_state

    foot_state_duration_exp3 = (
        (rwd.env.reward.rewards["gait_loop"].gait_loop_duration > 0) * 
        np.exp(-rwd.env.envdata.previer_cmd_vec_norm *
               (rwd.foot_state_duration / rwd.foot_state_sigma)**3))
    
    info = {
        "foot_state_duration": rwd.foot_state_duration,
        "foot_state_duration_exp3": foot_state_duration_exp3
    }
    return foot_state_duration_exp3, info


def foot_sliding_velocity_l2(rwd):
    foot_sliding_velocity = rwd.env.envdata.foot_lin_vel
    foot_sliding_velocity_l2 = np.sum(np.square(foot_sliding_velocity * rwd.env.envdata.foot_landed))

    info = {
        "foot_sliding_velocity": foot_sliding_velocity,
        "foot_sliding_velocity_l2": foot_sliding_velocity_l2
    }
    return foot_sliding_velocity_l2, info


def foot_lift_height_l2_exp_xy_vel_weighted(rwd):

    foot_lift_height = rwd.env.data.geom_xpos[rwd.env.envdata._foot_ids][:,2]
    foot_lift_height_l2 = np.square(foot_lift_height - rwd.foot_lift_height_target)
    foot_lift_height_l2_exp = np.exp(-foot_lift_height_l2)
    foot_lift_height_l2_exp_xy_vel_weighted = np.sum(foot_lift_height_l2_exp * rwd.env.envdata.foot_lin_vel)

    info = {
        "foot_lift_height": foot_lift_height,
        "foot_lift_height_l2": foot_lift_height_l2,
        "foot_lift_height_l2_exp": foot_lift_height_l2_exp,
        "foot_lift_height_l2_exp_xy_vel_weighted": foot_lift_height_l2_exp_xy_vel_weighted
    }
    return foot_lift_height_l2_exp_xy_vel_weighted, info


def foot_lift_height_l2_xy_vel_weighted_exp(rwd):

    foot_lift_height = rwd.env.data.geom_xpos[rwd.env.envdata._foot_ids][:,2]
    foot_lift_height_l2 = np.square(foot_lift_height - rwd.foot_lift_height_target)
    foot_lift_height_l2_xy_vel_weighted = foot_lift_height_l2 * rwd.env.envdata.foot_lin_vel
    foot_lift_height_l2_xy_vel_weighted_exp = np.sum(np.exp(-foot_lift_height_l2_xy_vel_weighted))

    info = {
        "foot_lift_height": foot_lift_height,
        "foot_lift_height_l2": foot_lift_height_l2,
        "foot_lift_height_l2_xy_vel_weighted": foot_lift_height_l2_xy_vel_weighted,
        "foot_lift_height_l2_xy_vel_weighted_exp": foot_lift_height_l2_xy_vel_weighted_exp
    }
    return foot_lift_height_l2_xy_vel_weighted_exp, info


def foot_velocity_variance(rwd):
    lifted_foot_ids = [foot_id for foot_id in np.array(rwd.env.envdata._foot_ids)[rwd.env.envdata.foot_lifted]]

    if len(lifted_foot_ids) >= 2:
        foot_x_velocity = np.zeros(len(lifted_foot_ids))
        foot_y_velocity = np.zeros(len(lifted_foot_ids))
        for i, foot_id in enumerate(lifted_foot_ids):
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(rwd.env.model, rwd.env.data, mujoco.mjtObj.mjOBJ_GEOM,
                                    foot_id, vel, 0)
            foot_x_velocity[i] = vel[3]
            foot_y_velocity[i] = vel[4]
        
        foot_x_velocity_variance = np.var(np.clip((foot_x_velocity - np.mean(foot_x_velocity)), -1, 1))
        foot_y_velocity_variance = np.var(np.clip((foot_y_velocity - np.mean(foot_y_velocity)), -1, 1))
        foot_velocity_variance = foot_x_velocity_variance + foot_y_velocity_variance

        info = {
            "foot_x_velocity_variance": foot_x_velocity_variance,
            "foot_y_velocity_variance": foot_y_velocity_variance,
            "foot_velocity_variance": foot_velocity_variance
        }
        return foot_velocity_variance, info
    
    else:
        info = {
            "foot_x_velocity_variance": 0.,
            "foot_y_velocity_variance": 0.,
            "foot_velocity_variance": 0.
        }
        return 0., info
    
def foot_contact_without_cmd(rwd):
    gait_target = command_to_gait_name(rwd.env.envdata.previer_cmd_vec)
    foot_contact_num_without_cmd = np.sum(rwd.env.envdata.foot_landed * (gait_target == "idle"))
    return foot_contact_num_without_cmd, {"foot_contact_num_without_cmd": foot_contact_num_without_cmd}