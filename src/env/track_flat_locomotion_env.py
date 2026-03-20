import mujoco
import math
import copy
import numpy as np
from gymnasium.spaces import Box, Dict

import numbers
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from collections import deque
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils

import src.reward.base as rwd
from src.reward.base import NewReward
from src.env.common.control import UniTreeGo1Control
from src.env.common.data import UniTreeGo1Data
from src.callback.base import CustomMatPlotLibCallback, CustomMujocoCallback


class FlatLocomotionEnv(MujocoEnv, utils.EzPickle):

    def __init__(
            self,
            xml_file: str,
            frame_skip: int = 5,
            main_body: int | str = 1,
            reset_noise_scale: float = 0.1,
            exclude_obs_base_pos: bool = True,
            exclude_obs_joint_pos: bool = True,
            exclude_obs_base_vel: bool = True,
            include_obs_cfrc_ext: bool = False,

            render_mode: str = None,
            plt_n_cols:  int = 4,
            plt_n_lines: int = 1,
            plt_x_range: int = 200,
            width: int = 480,
            height: int = 480,

            reward_config: dict = {},
            control_config: dict = {},

            **kwargs):
        
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            main_body,
            reset_noise_scale,
            exclude_obs_base_pos,
            exclude_obs_joint_pos,
            exclude_obs_base_vel,
            include_obs_cfrc_ext,

            reward_config,
            control_config,
            **kwargs,
        )

        self._main_body = main_body
        self._reset_noise_scale = reset_noise_scale
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=None, width=width, height=height)

        # self._exclude_obs_base_pos = exclude_obs_base_pos
        # self._exclude_obs_joint_pos = exclude_obs_joint_pos
        # self._exclude_obs_base_vel = exclude_obs_base_vel
        # self._include_obs_cfrc_ext = include_obs_cfrc_ext
        self.qpos_idx = np.arange(2 * exclude_obs_base_pos, len(self.data.qpos) - exclude_obs_joint_pos *
                                  (self.model.jnt_qposadr[-1] - self.model.jnt_qposadr[1] + 1))
        self.qvel_idx = np.arange(2 * exclude_obs_base_vel, self.model.nv)
        self.cfrc_idx = np.arange(1, self.model.nbody * include_obs_cfrc_ext)
        

        # self.metadata = {
        #     "render_modes": [
        #         "human",
        #         "rgb_array",
        #         "depth_array",
        #         "rgbd_tuple",
        #     ],
        #     "render_fps": int(np.round(1.0 / self.dt)),
        # }

        self.stage = 0 # update in callback
        self.reward = UniTreeGo1Reward(self, reward_config=reward_config)
        self.controller = UniTreeGo1Control(self, control_config=control_config)
        self.envdata = UniTreeGo1Data(self, reward_config=reward_config)
        self._init_customize_obs()

        self.render_mode = render_mode
        self.callbacks = [CustomMatPlotLibCallback(render_mode,
                                                   plt_n_cols=plt_n_cols,
                                                   plt_n_lines=plt_n_lines,
                                                   plt_x_range=plt_x_range),
                          CustomMujocoCallback(render_mode),]
        self._dispatch("_on_training_start", env=self)
        

    def reset(self, *, seed=None, options=None):
        super(MujocoEnv, self).reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.envdata.reset()

        qpos = (np.squeeze(self.model.key_qpos) + self.np_random.uniform(size=self.model.nq,
                                                                         low=-self._reset_noise_scale,
                                                                         high=self._reset_noise_scale))
        qvel = (np.squeeze(self.model.key_qvel) +
                self._reset_noise_scale * self.np_random.standard_normal(self.model.nv))
        self.set_state(qpos, qvel)
        observation = self._get_obs()

        self._dispatch("_on_episode_start")
        if self.render_mode == "human":
            self.render()
        return observation, {}
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.envdata.update(action)

        observation = self._get_obs()
        reward, reward_info, is_alive = self._get_rew()
        terminated = not is_alive
        info = {"stage": self.stage,
                "decimal_stage": self.stage - math.floor(self.stage),
                "reward": reward, **reward_info}

        self._dispatch("_on_step", state=self.state_vector(), info=info)
        if terminated:
            self._dispatch("_on_episode_end")

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info
    
    def render(self, render_mode=None):
        return self.mujoco_renderer.render(render_mode)

    def set_stage(self, stage):
        self.stage = stage

    def _dispatch(self, event_name, *args, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, event_name, None)
            if fn is not None:
                fn(*args, **kwargs)
    
    # region | Obs
    def _init_customize_obs(self):
        obs_size = 0
        self.observation_structure = {}

        def _add_obs_item(obs_name, obs_sample):
            nonlocal self, obs_size
            if obs_sample is not None and len(obs_sample) > 0:
                self.observation_structure[obs_name] = len(obs_sample)  
                obs_size += len(obs_sample)

        _add_obs_item("qpos", self.data.qpos[self.qpos_idx])
        _add_obs_item("qvel", self.data.qvel[self.qvel_idx])
        _add_obs_item("cfrc_ext", self.data.cfrc_ext[self.cfrc_idx].flatten())
        _add_obs_item("foot_landed_time", self.envdata.foot_landed_time)
        _add_obs_item("foot_lifted_time", self.envdata.foot_lifted_time)
        _add_obs_item("command_vector", self.envdata.cmd_vec)
        _add_obs_item("track_vel_vector", self.envdata.vel_vec)
        _add_obs_item("vel_diff", self.envdata.vel_diff)
        _add_obs_item("joint_diff", self.envdata.joint_diff)
        _add_obs_item("previous_action", self.envdata.previous_action)
        _add_obs_item("gravity_projection", [self.envdata.gravity_projection])
        # _add_obs_item("gait_type", self._get_gait_obs())

        self.observation_scale = {  
            "qpos": np.ones(len(self.data.qpos)),
            "qvel": np.concatenate((5*np.ones(2), [np.pi], 10*np.ones(3), 20*np.ones(self.model.nv - 6))),
            "time": 0.5*np.ones(4),
            "command": np.array([5.0, 1.0, np.pi]),
            "gravity": np.array([9.8]),
        }

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

    def _get_obs(self):
        observation = np.concatenate((
            (self.data.qpos / self.observation_scale["qpos"])[self.qpos_idx],
            (self.data.qvel / self.observation_scale["qvel"])[self.qvel_idx],
            np.tanh(self.data.cfrc_ext)[self.cfrc_idx].flatten(),
            (1 - np.exp(-self.envdata.foot_landed_time / self.observation_scale["time"])),
            (1 - np.exp(-self.envdata.foot_lifted_time / self.observation_scale["time"])),
            self.envdata.cmd_vec / self.observation_scale["command"],
            self.envdata.vel_vec / self.observation_scale["command"],
            self.envdata.vel_diff / self.observation_scale["command"],
            self.envdata.joint_diff,
            self.envdata.previous_action,
            self.envdata.gravity_projection / self.observation_scale["gravity"]))
        return observation

# region | Reward

    def _get_rew(self, *akwargs, **kwargs):
        reward, reward_info = self.reward()
        is_alive = reward_info["is_alive"]
        return reward, reward_info, is_alive
    
    
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
            if not np.isfinite(r).all():
                raise ValueError("reward is not finite")
            reward += r
            self.reward_info.update(i)
            type_str = "reward" if reward_fun.weight >= 0 else "penalty"
            self.reward_info.update({f"{reward_name}_{type_str}": r})
        return reward, self.reward_info

    def _init_rewards(self,
            alive_weight,
            illegal_contact_weight,
            track_rbf_k,
            robot_xy_velocity_weight,
            robot_x_velocity_std,
            robot_y_velocity_std,
            z_angular_velocity_weight,
            z_angular_velocity_std,
            xy_velocity_error_integral_weight,
            z_angular_velocity_error_integral_weight,
            intergral_len,
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
            # foot_contract_fz_sigma,
            # foot_lift_height_sigma,
            foot_state_duration_weight,
            foot_state_sigma,
            foot_sliding_velocity_weight,
            foot_lift_height_weight,
            foot_lift_height_target,
            foot_velocity_variance_weight,
            foot_contact_without_cmd_weight,
            **kwargs):
        
        self.rewards = {

            ### Basic Survival and Mission Rewards
            "alive":                    NewReward(self.env, rwd.is_alive, alive_weight),
            "illegal_contact":          NewReward(self.env, rwd.illegal_contact_l1, illegal_contact_weight),
            "robot_xy_velocity":        NewReward(self.env, rwd.robot_xy_velocity_rbf_logcosh, robot_xy_velocity_weight,
                                                  rbf_k=track_rbf_k,
                                                  robot_x_velocity_std=robot_x_velocity_std,
                                                  robot_y_velocity_std=robot_y_velocity_std),
            "z_angular_velocity":       NewReward(self.env, rwd.z_angular_velocity_rbf_logcosh, z_angular_velocity_weight,
                                                  rbf_k=track_rbf_k, z_angular_velocity_std=z_angular_velocity_std),
            "gait_loop":                NewReward(self.env, rwd.trot_loop_duration_tanh, gait_loop_weight,
                                                  gait_type=None, gait_loop_options=[], gait_loop_duration=0, gait_loop_k=gait_loop_k),
            "foot_state_duration":      NewReward(self.env, rwd.foot_state_duration_exp3, foot_state_duration_weight,
                                                  foot_state_sigma=foot_state_sigma, foot_state_old=None, foot_state_duration=0),
            "z_position":               NewReward(self.env, rwd.z_position_l2_xy_vel_weighted, z_position_weight,
                                                  z_position_target=z_position_target),

            ### Additional Constraints and Intensive Rewards
            "z_velocity":               NewReward(self.env, rwd.z_velocity_l2_xy_vel_weighted, z_velocity_weight),
            "xy_angular":               NewReward(self.env, rwd.xy_angular_gravity_projection, xy_angular_weight),
            "xy_angular_velocity":      NewReward(self.env, rwd.xy_angular_velocity_l2, xy_angular_velocity_weight),
            "hinge_position":           NewReward(self.env, rwd.hinge_position_l2, hinge_position_weight),
            "hinge_exceed_limit":       NewReward(self.env, rwd.hinge_exceed_limit_l1, hinge_exceed_limit_weight,
                                                  hinge_upper_limit=rwd.get_hinge_soft_upper_limit(self, hinge_exceed_limit_ratio),
                                                  hinge_lower_limit=rwd.get_hinge_soft_lower_limit(self, hinge_exceed_limit_ratio)),
            "foot_lift_height":         NewReward(self.env, rwd.foot_lift_height_l2_xy_vel_weighted_exp, foot_lift_height_weight,
                                                  foot_lift_height_target=foot_lift_height_target),
            
            ### Advanced Smoothing Rewards
            "action_change":            NewReward(self.env, rwd.action_change_l2, action_change_weight),
            "hinge_angular_velocity":   NewReward(self.env, rwd.hinge_angular_velocity_l2, hinge_angular_velocity_weight),
            "hinge_energy":             NewReward(self.env, rwd.hinge_energy_l1, hinge_energy_weight),

            ### Addictional Tracking Rewards
            "xy_velocity_error_integral":
                                        NewReward(self.env, rwd.xy_velocity_error_integral_l2, xy_velocity_error_integral_weight,
                                                  vel_error = deque(maxlen=intergral_len)),
            "z_angular_velocity_error_integral":
                                        NewReward(self.env, rwd.z_angular_velocity_error_integral_l2, z_angular_velocity_error_integral_weight,
                                                  angular_vel_error = deque(maxlen=intergral_len)),

            # ### Advanced Intensive Rewards
            # "foot_sliding_velocity":    NewReward(self.env, rwd.foot_sliding_velocity_l2, foot_sliding_velocity_weight),
            # "foot_velocity_variance":   NewReward(self.env, rwd.foot_velocity_variance, foot_velocity_variance_weight),
            # "foot_contact_without_cmd": NewReward(self.env, rwd.foot_contact_without_cmd, foot_contact_without_cmd_weight),

            # ### Alternative Gait Rewards
            # "gait_loop":                NewReward(self.env, rwd.gait_transfer, gait_loop_weight,
            #                                       gait_type=None, gait_loop_options=[], gait_loop_duration=0, gait_loop_k=gait_loop_k,
            #                                       foot_contract_fz_sigma=foot_contract_fz_sigma,
            #                                       foot_lift_height_sigma=foot_lift_height_sigma),
            # "gait_loop":                NewReward(self.env, rwd.trot_sync, gait_loop_weight*10),

        }
