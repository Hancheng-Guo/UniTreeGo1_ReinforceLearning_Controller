import mujoco
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
from src.config.config import CONFIG
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from src.render.render_matplotlib import init_plt_render
from src.config.config import CONFIG
from src.utils.noop import noop
from src.utils.decays import radial_decay


class UniTreeGo1Env(AntEnv):
    def __init__(
            self,
            healthy_pitch_range: tuple[float, float] = (0.2, 1.0),
            healthy_reward_weight: float = 1,
            demo_type: str = "multiple",
            **kwargs):
        super().__init__(
            **kwargs)
        self.healthy_reward_weight = healthy_reward_weight
        self._healthy_pitch_range = healthy_pitch_range
        self.demo_type = demo_type
        plt_clr = not (demo_type == "multiple")
        self.plt_render, self.plt_endline = init_plt_render(plt_clr) if self.render_mode == "human" else (noop, noop)
        self.plt_timer = time.time()

    @property
    def healthy_info(self):
        state = self.state_vector()
        mat = np.zeros((9, 1))
        mujoco.mju_quat2Mat(mat, state[3:7]) # Convert quaternion to 3D rotation matrix
        yaw = np.arctan2(mat[3], mat[0])[0]
        pitch = np.arcsin(-mat[6])[0]
        roll = np.arctan2(mat[7], mat[8])[0]
        healthy_info = {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }
        return healthy_info
    
    @property
    def healthy_reward(self):
        pitch_radius = (CONFIG["train"]["healthy_pitch_max"] - CONFIG["train"]["healthy_pitch_min"]) / 2
        pitch_reward = radial_decay(self.healthy_info["pitch"], radius=pitch_radius)
        yaw_reward = radial_decay(self.healthy_info["yaw"], radius=np.pi)

        hip_joints = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        hip_joints_decays = []
        for hip_joint in hip_joints:
            hip_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, hip_joint)
            hip_joint_addr = self.model.jnt_qposadr[hip_joint_id]
            hip_joints_pos = float(self.data.qpos[hip_joint_addr])
            hip_joints_min, hip_joints_max = self.model.jnt_range[hip_joint_id]
            hip_joints_decays.append(
                radial_decay(hip_joints_pos,
                             radius=(hip_joints_max - hip_joints_min) / 2, 
                             radius_value=0.5))
        hip_joints_decay = sum(hip_joints_decays) / len(hip_joints_decays)

        return yaw_reward * pitch_reward * self.healthy_reward_weight * hip_joints_decay
    
    @property
    def is_alive(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        min_pitch, max_pitch = self._healthy_pitch_range
        is_alive = np.isfinite(state).all() and min_z <= state[2] <= max_z and min_pitch <= self.healthy_info["pitch"] <= max_pitch
        return is_alive

    @property
    def alive_reward(self):
        return self.is_alive * self.healthy_reward
    
    @property
    def forward_info(self):
        xy_position_before = self.data_old.body(self._main_body).xpos[:2].copy()
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_info = {
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }
        return forward_info
    
    @property
    def forward_reward(self):
        forward_info = self.forward_info
        x_velocity = forward_info["x_velocity"]
        airborne_decay = self.contact_info["airborne_decay"]
        return x_velocity * self._forward_reward_weight * airborne_decay
        # return x_velocity * self._forward_reward_weight
    
    @property
    def contact_info(self):
        contact_forces = self.contact_forces
        total_mass = self.model.body_mass.sum()
        force_scale = total_mass * np.linalg.norm(self.model.opt.gravity)
        norm_contact_forces = contact_forces / force_scale
        min_value, max_value = self._contact_force_range
        clip_contact_forces = np.clip(norm_contact_forces, min_value, max_value)   
        clip_contact_forces_squared_sum = np.sum(np.square(clip_contact_forces))

        foot_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        foot_fz = []
        for foot_name in foot_names:
            foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, foot_name)
            foot_fz.append(contact_forces[foot_id][2])
        if "airborne_time" not in self.__dict__:
            self.airborne_time = 0
        # foot_fz_sorted = sorted(foot_fz, reverse=True)
        # self.airborne_time = 0 if foot_fz_sorted[1] > 1 else self.airborne_time + 1
        landed_case1 = (foot_fz[0] > 1) and (foot_fz[3] > 1)
        landed_case2 = (foot_fz[1] > 1) and (foot_fz[2] > 1)
        self.airborne_time = 0 if (landed_case1 or landed_case2) else (self.airborne_time + 1)
        airborne_decay = np.exp(-CONFIG["train"]["airborne_lambda"] * self.airborne_time)

        contact_info = {
            "foot_fz": foot_fz,
            "airborne_decay": airborne_decay,
            "clip_contact_forces_squared_sum": clip_contact_forces_squared_sum,
        }
        return contact_info
    
    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext     
        return raw_contact_forces
    
    @property
    def contact_cost(self):
        contact_info = self.contact_info
        return self._contact_cost_weight * contact_info["clip_contact_forces_squared_sum"]
    
    def reset(self, *, seed=None, options=None):
        options = options or {}
        options["init_key"] = CONFIG["algorithm"]["reset_state"]
        ob, info = super().reset(seed=seed, options=options)
        if self.render_mode == "human":
            # self.render() # Has been called in the parent class
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking")
            self.mujoco_renderer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.mujoco_renderer.viewer.cam.fixedcamid = camera_id
            self.camera_id = camera_id
            self.mujoco_renderer.camera_id = camera_id
        return ob, info
    
    def render(self, render_mode=None):
        if render_mode:
            return self.mujoco_renderer.render(render_mode)
        return self.mujoco_renderer.render(self.render_mode)
    
    def step(self, action):
        self.data_old =  copy.deepcopy(self.data)
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated = (not self.is_alive) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }
        
        if time.time() - self.plt_timer > 1:
            plt.pause(0.00001)
            self.plt_timer = time.time()

        if self.render_mode == "human":
            self.render()
            self.plt_render(self.state_vector(), info)
            if terminated:
                self.plt_endline()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info
    
    def _get_rew(self, action):
        forward_reward = self.forward_reward
        alive_reward = self.alive_reward
        rewards = forward_reward + alive_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        reward = reward / 10

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_alive": alive_reward,
            "reward_total": reward,
            **self.forward_info,
            **self.healthy_info,
            **self.contact_info,
        }

        return reward, reward_info
