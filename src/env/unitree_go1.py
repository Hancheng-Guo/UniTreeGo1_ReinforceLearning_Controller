import mujoco
import copy
import numpy as np
from collections import deque
from enum import IntEnum
from gymnasium.spaces import Box
from gymnasium.envs.mujoco.ant_v5 import AntEnv

from src.renders.matplotlib import PltRenderer
from src.renders.mujoco import set_tracking_camera
from src.utils.decays import radial_decay, clip_exp_decay, clip_exp_gain, clip_step_decay, StepGain
from src.callbacks.stage_schedule import Stage


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

            reset_state: str = "home",
            render_mode: str = None,
            plt_n_lines: int = 1,
            plt_x_range: int = 200,

            **kwargs):

        super().__init__(xml_file=xml_file,
                         frame_skip=frame_skip,
                         main_body=main_body,
                         reset_noise_scale=reset_noise_scale,
                         include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
                         exclude_current_positions_from_observation=exclude_current_positions_from_observation,)
        self._reset_state = reset_state
        self.stage = None # update in callback
        self.action = None
        self.action_old = None
        self.data_old = None
        self.reward = UniTreeGo1Reward(self, reward_config=None, **kwargs["reward_config"])
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

    def _init_render(self, plt_n_lines, plt_x_range):
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            self.plt_render = PltRenderer(self.render_mode,
                                          plt_n_lines=plt_n_lines,
                                          plt_x_range=plt_x_range)
            self.plt_render.reset()
    
    def _init_customize_obs(self):
        self._foot_landed_time = np.zeros(len(feet))
        self._foot_airborne_time = np.zeros(len(feet))
        obs_size = self.observation_space.shape[0] + 2 * len(feet)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        self.observation_structure["foot_landed_time"] = len(feet)
        self.observation_structure["foot_airborne_time"] = len(feet)

    def render(self, render_mode=None):
        if render_mode:
            return self.mujoco_renderer.render(render_mode)
        return self.mujoco_renderer.render(self.render_mode)
    
    def reset(self, *, seed=None, options=None):
        self.reward.reset()
        options = options or {}
        options["init_key"] = self._reset_state
        ob, info = super().reset(seed=seed, options=options)
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            set_tracking_camera(self)
            self.plt_render.reset()
        return ob, info
    
    def step(self, action):
        self.action_old = copy.deepcopy(self.action)
        self.action = action
        self.data_old = copy.deepcopy(self.data)
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew()
        terminated = (not self.reward.is_alive) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "stage": self.stage,
            **reward_info,
        }

        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            self.mjc_img = self.render()
            self.plt_img = self.plt_render(self.state_vector(), info)
            if terminated:
                self.plt_render.reset()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info
    
    def _get_obs(self):
        obs = super()._get_obs()
        landed_obs, airborne_obs = self._get_foot_obs
        return np.concatenate((obs, landed_obs.flatten(), airborne_obs.flatten()))
    
    def _get_rew(self):
        reward, reward_info = self.reward()
        return reward, reward_info
    
    @property
    def _get_foot_obs(self):
        for i, is_touching in enumerate(self._are_feet_touching_ground):
            self._foot_airborne_time[i] = 0 if is_touching else self._foot_airborne_time[i] + 1
            self._foot_landed_time[i] = self._foot_landed_time[i] + 1 if is_touching else 0
        return self._foot_landed_time, self._foot_airborne_time
    
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

# region | Reward

class NewReward():
    def __init__(self, env, fun, weight, stage_range=[-np.inf, np.inf]):
        self.env = env
        self.fun = fun
        self.weight = weight
        self.stage_range = stage_range
        
    def __call__(self) -> float:
        if (self.env.stage < self.stage_range[0]): return 0
        if (self.env.stage >= self.stage_range[-1]): return 0
        return self.weight * self.fun(self.env)


class UniTreeGo1Reward():
    def __init__(self, env,
            reward_config,
            healthy_pitch_range: list[float, float] = [-0.5, 0.5],
            healthy_z_range: list[float, float] = [0.18, 0.7],
            healthy_z_target: float = 0.27,
            healthy_reward_weight: float = 1.,
            posture_reward_weight: float = 1.,
            forward_reward_weight: float = 1.,
            gait_reward_alpha: float = 100.,
            state_reward_weight: float = 1.,
            state_reward_alpha: float = 0.2,
            state_reward_hold: float = 5,
            ctrl_cost_weight: float = 0.05,
            contact_cost_weight: float = 5.e-4,
            contact_force_range: list[float, float] = [-1.0, 1.0],
            ):
        self.env = env
        self.rewards = None
        # self._init_rewards(**reward_config)









        # for healthy_reward
        self._healthy_reward_weight = healthy_reward_weight
        self._healthy_pitch_range = healthy_pitch_range
        self._healthy_z_range = healthy_z_range
        self._healthy_z_target = healthy_z_target
        
        # for posture_reward
        self._posture_reward_weight = posture_reward_weight
        # for forward_reward
        self._forward_reward_weight = forward_reward_weight
        self._gait_reward_alpha = gait_reward_alpha
        # for state_reward
        self._state_reward_weight = state_reward_weight
        self._state_reward_alpha = state_reward_alpha
        self._state_reward_hold = state_reward_hold
        self.fsm = UniTreeGo1FSM()
        # for ctrl_cost
        self._ctrl_cost_weight = ctrl_cost_weight
        # for contact_cost
        self._contact_cost_weight = contact_cost_weight
        self._total_mass = self.env.model.body_mass.sum()
        self._contact_force_scale = self._total_mass * np.linalg.norm(self.env.model.opt.gravity)
        self._contact_force_range = contact_force_range

        
    def _init_rewards(self,
            ):
        
        self.rewards = [

        ]


    def __call__(self):
        # reward = 0
        # info = {}
        # for reward_item in self.rewards:
        #     r, i = reward_item()
        #     reward += r
        #     info.update(i)

        # return reward, info


        forward_reward = self.forward_reward
        healthy_reward = self.healthy_reward
        state_reward = self.state_reward
        posture_reward = self.posture_reward
        rewards = forward_reward + healthy_reward + state_reward + posture_reward

        ctrl_cost = self.control_cost(self.env.action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        reward_info = {
            "reward_healthy": healthy_reward,
            "reward_forward": forward_reward,
            "reward_state": state_reward,
            "reward_posture": posture_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_total": reward,
            **self.forward_info,
            **self.state_info,
            **self.posture_info,
            **self.contact_info,
        }

        return reward, reward_info
    
    def reset(self):
        self.fsm.reset()
        return True













    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

# region | Posture Reward

    @property
    def _get_rotation_matrix(self):
        mat = np.zeros((9, 1)) # R00 R01 R02 R10 R11 R12 R20 R21 R22
        mujoco.mju_quat2Mat(mat, self.env.state_vector()[3:7]) # Convert quaternion to 3D rotation matrix
        return mat

    @property
    def posture_info(self):
        mat = self._get_rotation_matrix

        posture_info = {
            "yaw": np.arctan2(mat[3], mat[0])[0],
            "pitch": np.arcsin(-mat[6])[0],
            "roll": np.arctan2(mat[7], mat[8])[0],
        }
        return posture_info
    
    @property
    def posture_reward(self):
        posture_info = self.posture_info

        yaw_decay = radial_decay(posture_info["yaw"],
                                 boundary=np.pi)

        pitch_decay = radial_decay(posture_info["pitch"],
                                   boundary=self._healthy_pitch_range,
                                   boundary_value=0.5)
        
        hip_joints_decay = 0
        for i, hip_joint_addr in enumerate(self.env._hip_joint_addrs):
            hip_joints_pos = float(self.env.data.qpos[hip_joint_addr])
            hip_joints_decay += radial_decay(
                hip_joints_pos,
                boundary=self.env.model.jnt_range[self.env._hip_joint_ids[i]])
        hip_joints_decay /= len(self.env._hip_joint_addrs)
        
        z_decay = radial_decay(self.env.state_vector()[2],
                               center=self._healthy_z_target,
                               boundary=self._healthy_z_target-self._healthy_z_range[0],
                               boundary_value=0.1)
        
        feet_z = [self.env.data.geom_xpos[foot_id][2] for foot_id in self.env._foot_ids]
        feet_z_decaies = [clip_exp_decay(foot_z) for foot_z in feet_z]
        feet_z_decay = np.prod(feet_z_decaies)
        
        return self._posture_reward_weight * (
            yaw_decay * pitch_decay * z_decay * hip_joints_decay + 
            (self.env.stage < Stage.mid) * feet_z_decay)

# endregion

# region | Healthy Reward
    
    @property
    def is_alive(self):
        for c in self.env.data.contact:
            illegal_touching = (
                (c.geom1 not in self.env._foot_ids and c.geom2 == self.env._floor_id) or
                (c.geom1 == self.env._floor_id and c.geom2 not in self.env._foot_ids))
            if illegal_touching:
                return False
        
        state = self.env.state_vector()
        z_min, z_max = self._healthy_z_range
        pitch_min, pitch_max = self._healthy_pitch_range
        is_alive = (
            np.isfinite(state).all()
            and z_min <= state[2] <= z_max
            and pitch_min <= self.posture_info["pitch"] <= pitch_max
            )
        return is_alive
    
    @property
    def healthy_reward(self):
        return (self._healthy_reward_weight * self.is_alive)
    
# endregion

# region | Forward Reward
    
    @property
    def forward_info(self):
        xy_position_before = self.env.data_old.body(self.env._main_body).xpos[:2].copy()
        xy_position_after = self.env.data.body(self.env._main_body).xpos[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.env.dt
        x_velocity, y_velocity = xy_velocity

        mat = self._get_rotation_matrix
        body_x = np.array([mat[0], mat[3], mat[6]]).reshape(-1)
        body_x = body_x / np.linalg.norm(body_x)

        feet_dpos = self.env.data.geom_xpos[self.env._foot_ids] - self.env.data_old.geom_xpos[self.env._foot_ids]
        feet_dx = np.dot(body_x, feet_dpos.T)
        feet_vx = feet_dx / self.env.dt
        foot_filted_vx = feet_vx[self.env._are_feet_touching_ground]
        if foot_filted_vx.size >= 2:
            feet_filted_vx_mse = np.mean((foot_filted_vx - np.mean(foot_filted_vx))**2)
        else:
            feet_filted_vx_mse = 0

        forward_info = {
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "feet_vx": feet_vx,
            "feet_filted_vx_mse": feet_filted_vx_mse,
        }
        return forward_info
    
    @property
    def forward_reward(self):

        forward_info = self.forward_info
        x_velocity = forward_info["x_velocity"]
        y_velocity = forward_info["y_velocity"]
        _forward_reward = x_velocity / (1 + np.abs(y_velocity))

        state_loop_time = self.state_info["state_loop_time"]
        state_loop_gain = clip_exp_gain(state_loop_time,
                                        alpha=1-self._state_reward_alpha,
                                        x_shift=-0.5
                                        )
        
        feet_filted_vx_mse = forward_info["feet_filted_vx_mse"]
        feet_vx_decay = clip_exp_decay(feet_filted_vx_mse, alpha=self._gait_reward_alpha)

        return self._forward_reward_weight * (
            (self.env.stage >= Stage.mid) * (_forward_reward + feet_vx_decay))
    
# endregion

# region | Contact Cost

    @property
    def contact_info(self):
        norm_contact_forces = self.contact_forces / self._contact_force_scale
        min_value, max_value = self._contact_force_range
        clip_contact_forces = np.clip(norm_contact_forces, min_value, max_value)   
        clip_contact_forces_squared_sum = np.sum(np.square(clip_contact_forces))

        contact_info = {
            "clip_contact_forces_squared_sum": clip_contact_forces_squared_sum,
        }
        return contact_info
    
    @property
    def contact_forces(self):
        raw_contact_forces = self.env.data.cfrc_ext     
        return raw_contact_forces
    
    @property
    def contact_cost(self):
        contact_info = self.contact_info
        return self._contact_cost_weight * contact_info["clip_contact_forces_squared_sum"]
    
# endregion

# region | State Reward

    

    
    
    @property
    def state_info(self):
        bonus, state_reward_time, state_loop_time = self.fsm.update(self.env._are_feet_touching_ground)
        state_info = {
            "state": self.fsm.state,
            "bonus": bonus,
            "state_reward_time": state_reward_time,
            "state_loop_time": state_loop_time,
        }
        return state_info

    @property
    def state_reward(self):
        state_info = self.state_info
        state_loop_time = state_info["state_loop_time"]
        state_loop_gain = clip_exp_gain(state_loop_time,
                                        alpha=1-self._state_reward_alpha,
                                        x_shift=-0.5
                                        )
        # bonus = state_info["bonus"]
        state_decay = clip_step_decay(state_info["state_reward_time"],
                                        alpha=self._state_reward_alpha,
                                        x_shift=self._state_reward_hold)
        # return (bonus + state_decay) * self._state_reward_weight
        return (self._state_reward_weight * 
                ((self.env.stage < Stage.mid) * (state_info["state"] in {State.s0, State.s1, State.s5, State.s9}) + 
                 (self.env.stage >= Stage.mid) * (state_decay * max(state_loop_gain, (-self.env.stage + 2 * Stage.mid)))))

# endregion

# region | Finite State Machine

class State(IntEnum):
    s0 = 0  # init state
    s1 = 1  # 
    s2 = 2  # 
    s3 = 3  # 
    s4 = 4  # 
    s5 = 5  # 
    s6 = 6  # 
    s7 = 7  # 
    s8 = 8  # 
    s9 = 9  # communal 4-legged stance
    x0 = -1 # 0-legged stance
    x1 = -2 # 1-legged stance
    x2 = -3 # 2-legged stance (without diagonal support legs)


class UniTreeGo1FSM:
    def __init__(self):
        self.state = State.s0
        self._state_duration = -1
        self.loop_duration = -1
        self.trans = {
            State.s0: self.s0,
            State.s1: self.s1,
            State.s2: self.s2,
            State.s3: self.s3,
            State.s4: self.s4,
            State.s5: self.s5,
            State.s6: self.s6,
            State.s7: self.s7,
            State.s8: self.s8,
            State.s9: self.s9,
            State.x0: self.x0,
            State.x1: self.x1,
            State.x2: self.x2,
        }

    def check(self, are_feet_touching_ground):
        # only return s2-s4, s6-s9, x0-x2
        # s9 include s1 and s5
        # x0 include s0
        landed_count = sum(are_feet_touching_ground)
        if landed_count == 0:
            return State.x0
        if landed_count == 1:
            return State.x1
        if landed_count == 4:
            return State.s9
        if landed_count == 3:
            if not are_feet_touching_ground[0]:
                return State.s2
            if not are_feet_touching_ground[1]:
                return State.s6
            if not are_feet_touching_ground[2]:
                return State.s8
            return State.s4 # are_feet_touching_ground[3] == 0
        # landed_count == 2
        if ((are_feet_touching_ground[0] + are_feet_touching_ground[1] == 0) or
            (are_feet_touching_ground[2] + are_feet_touching_ground[3] == 0) or
            (are_feet_touching_ground[0] + are_feet_touching_ground[2] == 0) or
            (are_feet_touching_ground[1] + are_feet_touching_ground[3] == 0)):
                return State.x2
        if (are_feet_touching_ground[0] + are_feet_touching_ground[3] == 0):
            return State.s3
        return State.s7 # are_feet_touching_ground[1] + are_feet_touching_ground[2] == 0

    def update(self, obs):
        target_state = self.check(obs)
        bonus, has_reward = self.trans[self.state](target_state)
        reward_time = self._state_duration if has_reward else -1
        self.loop_duration = self.loop_duration + 1 if has_reward else -1
        return bonus, reward_time, self.loop_duration
    
    def reset(self):
        self.state = State.s0
        self._state_duration = -1
        self.loop_duration = -1

    def s0(self, target_state):
        self.state = State.s0 if target_state == State.x0 else target_state
        has_reward = False if self.state in {State.s0, State.x1, State.x2} else True
        if self.state == State.s9:
            bonus = 1
        else:
            bonus = 0
        self._state_duration = self._state_duration + 1 if self.state == State.s0 else 0
        return bonus, has_reward
    
    def s1(self, target_state):
        self.state = State.s1 if target_state == State.s9 else target_state
        has_reward = True if self.state in {State.s1, State.s2, State.s3} else False
        if self.state == State.s1:
            bonus = 0
        elif self.state in {State.s2, State.s3}:
            bonus = 1
        else:
            bonus = -1
        self._state_duration = self._state_duration + 1 if self.state == State.s1 else 0
        return bonus, has_reward
    
    def s2(self, target_state):
        self.state = target_state
        has_reward = True if self.state in {State.s2, State.s3, State.s4} else False
        if self.state == State.s2:
            bonus = 0
        elif self.state in {State.s3, State.s4}:
            bonus = 1
        else:
            bonus = -1
        self._state_duration = self._state_duration + 1 if self.state == State.s2 else 0
        return bonus, has_reward
    
    def s3(self, target_state):
        self.state = State.s5 if target_state == State.s9 else target_state
        has_reward = True if self.state in {State.s3, State.s4, State.s5} else False
        if self.state == State.s3:
            bonus = 0
        elif self.state in {State.s4, State.s5}:
            bonus = 1
        else:
            bonus = -1
        self._state_duration = self._state_duration + 1 if self.state == State.s3 else 0
        return bonus, has_reward
    
    def s4(self, target_state):
        self.state = State.s5 if target_state == State.s9 else target_state
        has_reward = True if self.state in {State.s4, State.s5, State.s6} else False
        if self.state == State.s4:
            bonus = 0
        elif self.state in {State.s5, State.s6}:
            bonus = 1
        else:
            bonus = -1
        self._state_duration = self._state_duration + 1 if self.state == State.s4 else 0
        return bonus, has_reward
    
    def s5(self, target_state):
        self.state = State.s5 if target_state == State.s9 else target_state
        has_reward = True if self.state in {State.s5, State.s6, State.s7} else False
        if self.state == State.s5:
            bonus = 0
        elif self.state in {State.s6, State.s7}:
            bonus = 1
        else:
            bonus = -1
        self._state_duration = self._state_duration + 1 if self.state == State.s5 else 0
        return bonus, has_reward
    
    def s6(self, target_state):
        self.state = target_state
        has_reward = True if self.state in {State.s6, State.s7, State.s8} else False
        if self.state == State.s6:
            bonus = 0
        elif self.state in {State.s7, State.s8}:
            bonus = 1
        else:
            bonus = -1
        self._state_duration = self._state_duration + 1 if self.state == State.s6 else 0
        return bonus, has_reward
    
    def s7(self, target_state):
        self.state = State.s1 if target_state == State.s9 else target_state
        has_reward = True if self.state in {State.s7, State.s8, State.s1} else False
        if self.state == State.s7:
            bonus = 0
        elif self.state in {State.s8, State.s1}:
            bonus = 1
        else:
            bonus = -1
        self._state_duration = self._state_duration + 1 if self.state == State.s7 else 0
        return bonus, has_reward
    
    def s8(self, target_state):
        self.state = State.s1 if target_state == State.s9 else target_state
        has_reward = True if self.state in {State.s8, State.s1, State.s2} else False
        if self.state == State.s8:
            bonus = 0
        elif self.state in {State.s1, State.s2}:
            bonus = 1
        else:
            bonus = -1
        self._state_duration = self._state_duration + 1 if self.state == State.s8 else 0
        return bonus, has_reward
    
    def s9(self, target_state):
        self.state = target_state
        has_reward = True if self.state in {State.s2, State.s3, State.s6, State.s7, State.s9} else False
        if self.state == State.s9:
            bonus = 0
        elif self.state in {State.s2, State.s3, State.s6, State.s7}:
            bonus = 1
        else:
            bonus = -1
        self._state_duration = self._state_duration + 1 if self.state == State.s9 else 0
        return bonus, has_reward
    
    def x0(self, target_state):
        self.state = target_state
        has_reward = False if self.state in {State.x0, State.x1, State.x2} else True
        if self.state == State.s9:
            bonus = 1
        else:
            bonus = 0
        self._state_duration = self._state_duration + 1 if self.state == State.x0 else 0
        return bonus, has_reward
    
    def x1(self, target_state):
        self.state = target_state
        has_reward = False if self.state in {State.x0, State.x1, State.x2} else True
        if self.state == State.s9:
            bonus = 1
        else:
            bonus = 0
        self._state_duration = self._state_duration + 1 if self.state == State.x1 else 0
        return bonus, has_reward
    
    def x2(self, target_state):
        self.state = target_state
        has_reward = False if self.state in {State.x0, State.x1, State.x2} else True
        if self.state == State.s9:
            bonus = 1
        else:
            bonus = 0
        self._state_duration = self._state_duration + 1 if self.state == State.x2 else 0
        return bonus, has_reward

# endregion

# endregion