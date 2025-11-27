import mujoco
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from src.render.render_matplotlib import init_plt_render


def noop(*args, **kwargs):
    pass


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
        if self.render_mode == "human":
            self.render()
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking")
            self.mujoco_renderer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.mujoco_renderer.viewer.cam.fixedcamid = camera_id
            self.camera_id = camera_id
            self.mujoco_renderer.camera_id = camera_id

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
        yaw = self.healthy_info["yaw"]
        yaw_reward_min = 0.05
        yaw_reward_alpha = -np.log(yaw_reward_min) / np.square(np.pi)
        yaw_reward = np.exp(-yaw_reward_alpha * np.square(yaw))
        return yaw_reward * self.healthy_reward_weight
    
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
        return x_velocity * self._forward_reward_weight
    
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

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_alive": alive_reward,
            "reward_total": reward,
            **self.forward_info,
            **self.healthy_info,
        }

        return reward, reward_info
