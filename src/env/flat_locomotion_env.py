import mujoco
import copy
import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.ant_v5 import AntEnv

import src.reward.base as rwd
from src.reward.base import NewReward, speed_to_gait_index, gait_loop_dict
from src.control.base import UniTreeGo1ControlGenerator, UniTreeGo1ControlUDP
from src.callback.base import CustomMatPlotLibCallback, CustomMujocoCallback


feet = ["FR", "FL", "RR", "RL"]
hip_joints = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]


class FlatLocomotionEnv(AntEnv):
    def __init__(
            self,
            xml_file: str = "ant.xml",
            frame_skip: int = 5,
            main_body: int | str = 1,
            reset_noise_scale: float = 0.1,
            exclude_current_positions_from_observation: bool = False,
            include_cfrc_ext_in_observation: bool = True,

            render_mode: str = None,
            plt_n_cols:  int = 4,
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

        # for stage
        self.stage = 0 # update in callback
        # for reward
        self.action = None
        self.action_old = None
        self._foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, foot)
                          for foot in feet]
        self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.reward = UniTreeGo1Reward(self, reward_config=kwargs["reward_config"])
        # for control
        self.controller = UniTreeGo1Control(self, control_config=kwargs["control_config"])
        self.control_vector = self.controller.get()
        # for obs
        self._feet_landed_time = np.zeros(len(feet))
        self._feet_airborne_time = np.zeros(len(feet))
        self._init_customize_obs()
        # for demo
        self.render_mode = render_mode
        self.callbacks = [CustomMatPlotLibCallback(render_mode,
                                                   plt_n_cols=plt_n_cols,
                                                   plt_n_lines=plt_n_lines,
                                                   plt_x_range=plt_x_range),
                          CustomMujocoCallback(render_mode),]
        self._dispatch("_on_training_start", env=self)
        

    def reset(self, *, seed=None, options=None):
        ob, info = super().reset(seed=seed, options=options)
        self.controller.reset()
        self._dispatch("_on_episode_start")
        return ob, info
    
    def step(self, action):
        self.control_vector = self.controller.get()
        self.action_old = copy.deepcopy(self.action)
        self.action = action
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info, is_alive = self._get_rew()
        terminated = (not is_alive) and self._terminate_when_unhealthy
        info = {"stage": self.stage, "reward": reward, **reward_info}

        self._dispatch("_on_step", state=self.state_vector(), info=info)
        if terminated:
            self._dispatch("_on_episode_end")

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info
    

    def render(self, render_mode=None):
        return self.mujoco_renderer.render(render_mode)


    def _get_rew(self, *akwargs, **kwargs):
        reward, reward_info = self.reward()
        is_alive = reward_info["is_alive"]
        return reward, reward_info, is_alive
    

    def _dispatch(self, event_name, *args, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, event_name, None)
            if fn is not None:
                fn(*args, **kwargs)
    
    # region | Obs

    def _init_customize_obs(self):
        obs_size = self.observation_space.shape[0]

        def _add_obs_item(obs_name, obs_sample):
            nonlocal self, obs_size
            self.observation_structure[obs_name] = len(obs_sample)
            obs_size += len(obs_sample)

        _add_obs_item("foot_landed_time", self._feet_landed_time)
        _add_obs_item("foot_airborne_time", self._feet_airborne_time)
        _add_obs_item("control_vector", self.controller)
        # _add_obs_item("gait_type", self._get_gait_obs())
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        

    def _get_obs(self):
        obs = super()._get_obs()
        feet_obs = self._get_feet_obs()
        control_obs = self.control_vector
        # gait_obs = self._get_gait_obs()
        # return np.concatenate((obs, feet_obs.flatten(), control_obs.flatten(), gait_obs.flatten()))
        return np.concatenate((obs, feet_obs.flatten(), control_obs.flatten()))

    def _get_feet_obs(self):
        for i, is_touching in enumerate(rwd.are_foot_touching_ground(self)):
            self._feet_airborne_time[i] = 0 if is_touching else self._feet_airborne_time[i] + 1
            self._feet_landed_time[i] = self._feet_landed_time[i] + 1 if is_touching else 0
        return np.concatenate((self._feet_landed_time.flatten(), self._feet_airborne_time.flatten()))
    

    # def _get_gait_obs(self):
    #     gait_index = speed_to_gait_index(np.linalg.norm(self.control_vector[0:2]))
    #     return np.eye(len(gait_loop_dict))[gait_index].flatten()


# region | Control

class UniTreeGo1Control:
    def __init__(self, env, control_config):
        self.env = env
        if control_config["control_type"] == "udp":
            self.controller = UniTreeGo1ControlUDP(self.env, **control_config)
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
            hinge_position_std,
            hinge_exceed_limit_weight,
            hinge_exceed_limit_ratio,
            hinge_energy_weight,
            gait_loop_weight,
            gait_loop_k,
            foot_state_duration_weight,
            foot_state_k,
            foot_sliding_velocity_weight,
            foot_lift_height_weight,
            foot_lift_height_target,
            **kwargs):
        
        self.rewards = {
            "alive":                    NewReward(self.env, rwd.is_alive, alive_weight),
            "illegal_contact":          NewReward(self.env, rwd.illegal_contact_l1, illegal_contact_weight),
            "robot_xy_velocity":        NewReward(self.env, rwd.robot_xy_velocity_l2_exp, robot_xy_velocity_weight),
            "z_angular_velocity":       NewReward(self.env, rwd.z_angular_velocity_l2_exp, z_angular_velocity_weight),
            "z_velocity":               NewReward(self.env, rwd.z_velocity_l2_xy_vel_weighted, z_velocity_weight,),
            "z_position":               NewReward(self.env, rwd.z_position_l2_xy_vel_weighted, z_position_weight,
                                                  z_position_target=z_position_target),
            "xy_angular_velocity":      NewReward(self.env, rwd.xy_angular_velocity_l2, xy_angular_velocity_weight),
            "xy_angular":               NewReward(self.env, rwd.xy_angular_gravity_projection, xy_angular_weight),
            "action_change":            NewReward(self.env, rwd.action_change_l2, action_change_weight),
            "hinge_angular_velocity":   NewReward(self.env, rwd.hinge_angular_velocity_l2, hinge_angular_velocity_weight),
            "hinge_position":           NewReward(self.env, rwd.hinge_position_l2, hinge_position_weight,
                                                  hinge_position0=self.env.model.key_qpos[0][7:19],
                                                  hinge_position_std=hinge_position_std),
            "hinge_exceed_limit":       NewReward(self.env, rwd.hinge_exceed_limit_l1, hinge_exceed_limit_weight,
                                                  hinge_upper_limit=rwd.get_hinge_soft_upper_limit(self, hinge_exceed_limit_ratio),
                                                  hinge_lower_limit=rwd.get_hinge_soft_lower_limit(self, hinge_exceed_limit_ratio)),
            "hinge_energy":             NewReward(self.env, rwd.hinge_energy_l1, hinge_energy_weight),
            "gait_loop":                NewReward(self.env, rwd.gait_loop_duration_tanh, gait_loop_weight,
                                                  gait_loop_k=gait_loop_k, gait_type=None, gait_loop_options=[], gait_loop_duration=0),
            "foot_state_duration":      NewReward(self.env, rwd.foot_state_duration_exp, foot_state_duration_weight,
                                                  foot_state_k=foot_state_k, foot_state_old=None, foot_state_duration=0),
            "foot_sliding_velocity":    NewReward(self.env, rwd.foot_sliding_velocity_l2, foot_sliding_velocity_weight),
            "foot_lift_height":         NewReward(self.env, rwd.foot_lift_height_l2_exp_xy_vel_weighted, foot_lift_height_weight,
                                                  foot_lift_height_target=foot_lift_height_target),
        }
