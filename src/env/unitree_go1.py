import mujoco
import copy
import numpy as np
from collections import deque
from enum import IntEnum
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.ant_v5 import AntEnv

import src.env.rewards as rwd
from src.renders.matplotlib import PltRenderer
from src.renders.mujoco import set_tracking_camera
from src.utils.decays import radial_decay, clip_exp_decay, clip_exp_gain, clip_step_decay, StepGain
from src.callbacks.stage_schedule import Stage
from src.env.control import UniTreeGo1ControlGenerator, UniTreeGo1ControlROS


feet = ["FR", "FL", "RR", "RL"]
hip_joints = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]

class UniTreeGo1Env(AntEnv):
    def __init__(
            self,
            xml_file: str = "ant.xml",
            frame_skip: int = 5,
            main_body: int | str = 1,
            reset_noise_scale: float = 0.1,
            exclude_current_positions_from_observation: bool = False,
            include_cfrc_ext_in_observation: bool = True,

            render_mode: str = None,
            plt_n_lines: int = 1,
            plt_x_range: int = 200,
            width: int = 480,
            height: int = 480,

            **kwargs):

        super().__init__(xml_file=xml_file,
                         frame_skip=frame_skip,
                         main_body=main_body,
                         reset_noise_scale=reset_noise_scale,
                         include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
                         exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                         width=width,
                         height=height)

        self.stage = 0 # update in callback
        self.action = None
        self.action_old = None
        self.reward = UniTreeGo1Reward(self, reward_config=kwargs["reward_config"])
        self.controller = UniTreeGo1Control(self, control_config=kwargs["control_config"])
        self.control_vector = self.controller.get()
        # for demo
        self.render_mode = render_mode
        self._init_render(plt_n_lines, plt_x_range)
        self.mjc_img = None
        self.plt_img = None
        # for obs
        self._init_customize_obs()
        self._hip_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, hip_joint)
                               for hip_joint in hip_joints]
        self._hip_joint_addrs = [self.model.jnt_qposadr[hip_joint_id]
                                 for hip_joint_id in self._hip_joint_ids]
        self._foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot)
                          for foot in feet]
        self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def reset(self, *, seed=None, options=None):
        ob, info = super().reset(seed=seed, options=options)
        self.controller.reset()
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            set_tracking_camera(self)
            self.plt_render.reset()
        return ob, info
    
    def step(self, action):
        self.control_vector = self.controller.get()
        self.action_old = copy.deepcopy(self.action)
        self.action = action
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew()
        terminated = (not reward_info["is_alive"]) and self._terminate_when_unhealthy
        info = {"stage": self.stage, "reward": reward, **reward_info}

        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            self.mjc_img = self.render()
            self.plt_img = self.plt_render(self.state_vector(), info)
            if terminated:
                self.plt_render.reset()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info
    
    def _get_rew(self, *akwargs, **kwargs):
        return self.reward()
    
    # region | Render

    def _init_render(self, plt_n_lines, plt_x_range):
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            self.plt_render = PltRenderer(self.render_mode,
                                          plt_n_lines=plt_n_lines,
                                          plt_x_range=plt_x_range)
            self.plt_render.reset()

    def render(self, render_mode=None):
        if render_mode:
            return self.mujoco_renderer.render(render_mode)
        return self.mujoco_renderer.render(self.render_mode)

    # endregion

    # region | Obs

    def _init_customize_obs(self):
        self._feet_landed_time = np.zeros(len(feet))
        self._feet_airborne_time = np.zeros(len(feet))
        obs_size = self.observation_space.shape[0] + 2 * len(feet) + len(self.controller)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.observation_structure["foot_landed_time"] = len(feet)
        self.observation_structure["foot_airborne_time"] = len(feet)
        self.observation_structure["control_vector"] = len(self.controller)

    def _get_obs(self):
        obs = super()._get_obs()
        feet_obs = self._get_feet_obs()
        control_obs = self.control_vector
        return np.concatenate((obs, feet_obs.flatten(), control_obs.flatten()))

    def _get_feet_obs(self):
        for i, is_touching in enumerate(self._are_feet_touching_ground):
            self._feet_airborne_time[i] = 0 if is_touching else self._feet_airborne_time[i] + 1
            self._feet_landed_time[i] = self._feet_landed_time[i] + 1 if is_touching else 0
        return np.concatenate((self._feet_landed_time.flatten(), self._feet_airborne_time.flatten()))
    
    @property
    def _are_feet_touching_ground(self):
        are_touching = []
        for foot_id in self._foot_ids:
            is_touching = False
            for c in self.data.contact:
                is_touching = ((c.geom1 == foot_id and c.geom2 == self._floor_id) or
                               (c.geom1 == self._floor_id and c.geom2 == foot_id))
                if is_touching:
                    break
            are_touching.append(is_touching)
        return are_touching
    
    @property
    def _feet_state(self):
        _are_feet_touching_ground = self._are_feet_touching_ground
        n = len(_are_feet_touching_ground)
        return sum(int(b) << (n - 1 - i) for i, b in enumerate(_are_feet_touching_ground))
    
    # endregion

# region | Control

class UniTreeGo1Control:
    def __init__(self, env, control_config):
        self.env = env
        if control_config["control_type"] == "ros":
            self.controller = UniTreeGo1ControlROS(self.env, **control_config)
        else:
            self.controller = UniTreeGo1ControlGenerator(self.env, **control_config)
    
    def __len__(self):
        return len(self.controller)

    def get(self):
        return self.controller.get()
    
    def reset(self):
        self.controller.reset()

# endregion

# region | Reward

class NewReward:
    def __init__(self, env, fun, weight, stage_range=[-np.inf, np.inf], **kwargs):
        self.env = env
        self.fun = fun
        self.weight = weight
        self.stage_range = stage_range
        self.__dict__.update(kwargs)
        
    def __call__(self) -> float:
        if (self.env.stage < self.stage_range[0]): return 0, {}
        if (self.env.stage >= self.stage_range[-1]): return 0, {}
        reward, reward_info = self.fun(self)
        return self.weight * reward, reward_info


class UniTreeGo1Reward:
    def __init__(self, env, reward_config):
        self.env = env
        self.rewards = None
        self.reward_info = {}
        self._init_rewards(**reward_config)

    def __call__(self):
        reward = 0
        self.reward_info = {}
        for reward_name, reward_fun in self.rewards.items():
            r, i = reward_fun()
            reward += r
            self.reward_info.update(i)
            type_str = "reward" if reward_fun.weight >= 0 else "penalty"
            self.reward_info.update({f"{reward_name}_{type_str}": r})
        return reward, self.reward_info

    def _init_rewards(self,
            alive_weight,
            illegal_contact_weight,
            robot_xy_velocity_weight,
            z_angular_velocity_weight,
            z_velocity_weight,
            z_position_weight,
            z_position_target,
            xy_angular_velocity_weight,
            xy_angular_weight,
            action_change_weight,
            hinge_angular_velocity_weight,
            hinge_position_weight,
            hinge_exceed_limit_weight,
            hinge_exceed_limit_ratio,
            hinge_energy_weight,
            gait_loop_weight,
            gait_loop_k,
            foot_sliding_velocity_weight,
            foot_lift_height_weight,
            foot_lift_height_target,
            **kwargs):
        
        self.rewards = {
            "alive":                    NewReward(self.env, rwd.is_alive, alive_weight),
            "illegal_contact":          NewReward(self.env, rwd.illegal_contact, illegal_contact_weight),
            "robot_xy_velocity":        NewReward(self.env, rwd.robot_xy_velocity_mse_exp, robot_xy_velocity_weight),
            "z_angular_velocity":       NewReward(self.env, rwd.z_angular_velocity_mse_exp, z_angular_velocity_weight),
            "z_velocity":               NewReward(self.env, rwd.z_velocity_ms, z_velocity_weight,),
            "z_position":               NewReward(self.env, rwd.z_position_mse, z_position_weight,
                                                  z_position_target=z_position_target),
            "xy_angular_velocity":      NewReward(self.env, rwd.xy_angular_velocity_ms, xy_angular_velocity_weight),
            "xy_angular":               NewReward(self.env, rwd.xy_angular_ms, xy_angular_weight),
            "action_change":            NewReward(self.env, rwd.action_change_ms, action_change_weight),
            "hinge_angular_velocity":   NewReward(self.env, rwd.hinge_angular_velocity_ms, hinge_angular_velocity_weight),
            "hinge_position":           NewReward(self.env, rwd.hinge_position_mse, hinge_position_weight,
                                                  hinge_position0=self.env.model.key_qpos[0][7:19]),
            "hinge_exceed_limit":       NewReward(self.env, rwd.hinge_exceed_limit_sum, hinge_exceed_limit_weight,
                                                  hinge_upper_limit=rwd.get_hinge_soft_upper_limit(self, hinge_exceed_limit_ratio),
                                                  hinge_lower_limit=rwd.get_hinge_soft_lower_limit(self, hinge_exceed_limit_ratio)),
            "hinge_energy":             NewReward(self.env, rwd.hinge_energy_sum, hinge_energy_weight),
            "gait_loop":                NewReward(self.env, rwd.gait_loop_tanh, gait_loop_weight,
                                                  gait_loop_k=gait_loop_k, gait_type=None, gait_loop_options=[], gait_loop_duration=0),
            "foot_sliding_velocity":    NewReward(self.env, rwd.foot_sliding_velocity_ms, foot_sliding_velocity_weight),
            "foot_lift_height":         NewReward(self.env, rwd.foot_lift_height_mse_exp_xy_vel_weighted, foot_lift_height_weight,
                                                  foot_lift_height_target=foot_lift_height_target),
            
        }
