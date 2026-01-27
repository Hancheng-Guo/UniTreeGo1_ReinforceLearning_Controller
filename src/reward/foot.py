import mujoco
import numpy as np
from src.reward.common.get_foot_state import are_foot_touching_ground, get_foot_state


def foot_state_duration_exp(rwd):
    if get_foot_state(rwd.env) == rwd.foot_state_old:
        rwd.foot_state_duration += 1 
    else:
        rwd.foot_state_duration = 0
    rwd.foot_state_old = get_foot_state(rwd.env)

    if rwd.env.reward.rewards["gait_loop"].gait_loop_duration > 0:
        velocity_control = np.linalg.norm(rwd.env.control_vector[0:2])
        foot_state_duration_exp = np.exp(-velocity_control * rwd.foot_state_k * rwd.foot_state_duration)
    else:
        foot_state_duration_exp = 0.
    
    info = {
        "foot_state_duration": rwd.foot_state_duration,
        "foot_state_duration_exp": foot_state_duration_exp
    }
    return foot_state_duration_exp, info


def foot_sliding_velocity_l2(rwd):
    are_foot_landed = np.array(are_foot_touching_ground(rwd.env))
    foot_sliding_velocity = np.zeros(len(are_foot_landed))
    for i, foot_id in enumerate(rwd.env._foot_ids):
        if are_foot_landed[i]:
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
    if rwd.env.reward.rewards["gait_loop"].gait_loop_duration > 0:
        are_foot_lifted = np.logical_not(np.array(are_foot_touching_ground(rwd.env)))
        lifted_foot_ids = [foot_id for foot_id in np.array(rwd.env._foot_ids)[are_foot_lifted]]
        if len(lifted_foot_ids) == 0:
            return 0., {"foot_lift_xy_velocity": [],
                        "foot_lift_height_l2_exp_xy_vel_weighted": 0.}
        
        foot_lift_height = rwd.env.data.geom_xpos[lifted_foot_ids][:,2]
        foot_lift_height_l2 = np.square(foot_lift_height - rwd.foot_lift_height_target)
        foot_lift_height_l2_exp = np.exp(-foot_lift_height_l2)

        foot_xy_velocity = np.zeros(len(lifted_foot_ids))
        for i, foot_id in enumerate(lifted_foot_ids):
            if are_foot_lifted[i]:
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