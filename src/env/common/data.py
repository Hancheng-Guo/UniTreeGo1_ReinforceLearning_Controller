from collections import deque
import math
import copy
import numpy as np
import mujoco
from src.env.common.control import UniTreeGo1Control

class DataDeque:
    def __init__(self, **kwargs):
        self.data = deque(**kwargs)

    @property
    def mean(self):
        if len(self.data) != 0:
            return np.mean(self.data)
        return np.float32(0)

    def __len__(self):
        return len(self.data)

    def __getattr__(self, name):
        return getattr(self.data, name)


class UniTreeGo1Data:

    feet = ["FR", "FL", "RR", "RL"]

    def __init__(self, env, reward_config):
        self.env = env
        self.smooth_len = math.ceil(reward_config["track_smooth_time"] / self.env.dt)
        self.action_memory_len = 2
        self._foot_ids = [mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, foot) for foot in self.feet]
        self._floor_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.init()

    def init(self):
        self.foot_lifted_time = np.zeros(len(self.feet))
        self.foot_landed_time = np.zeros(len(self.feet))
        self.rotation_matrix = None
        self.gravity_projection = np.float64(0)
        self.robot_x_lin_vel = DataDeque(maxlen=self.smooth_len)
        self.robot_y_lin_vel = DataDeque(maxlen=self.smooth_len)
        self.robot_z_ang_vel = DataDeque(maxlen=self.smooth_len)
        self.previer_cmd_vec = np.zeros(len(self.env.controller))
        self.current_cmd_vec = self.env.controller.get()
        self.previous_action_deque = deque([np.zeros(self.env.action_space.shape)
                                            for _ in range(self.action_memory_len)],
                                            maxlen=self.action_memory_len)
    
    def reset(self):
        self.foot_lifted_time = np.zeros(len(self.feet))
        self.foot_landed_time = np.zeros(len(self.feet))
        self.rotation_matrix = self.get_rotation_matrix()
        self.gravity_projection = self.get_gravity_projection()
        self.robot_x_lin_vel.clear()
        self.robot_y_lin_vel.clear()
        self.robot_z_ang_vel.clear()
        self.previer_cmd_vec = np.zeros(len(self.env.controller))
        self.env.controller.reset()
        self.current_cmd_vec = self.env.controller.get()
        [self.previous_action_deque.append(np.zeros(self.env.action_space.shape))
         for _ in range(self.action_memory_len)]

    def update(self, action):
        for i, is_touching in enumerate(self.are_foot_touching_ground):
            self.foot_lifted_time[i] = 0. if is_touching else self.foot_lifted_time[i] + self.env.dt
            self.foot_landed_time[i] = self.foot_landed_time[i] + self.env.dt if is_touching else 0.
        self.foot_lin_vel = self.get_foot_lin_vel()
        self.foot_fz = self.get_foot_fz()

        self.rotation_matrix = self.get_rotation_matrix()
        self.gravity_projection = self.get_gravity_projection()
        
        robot_lin_vel = self.rotation_matrix.T @ np.array(self.env.data.qvel[0:3])
        self.robot_x_lin_vel.append(robot_lin_vel[0])
        self.robot_y_lin_vel.append(robot_lin_vel[1])
        self.robot_z_ang_vel.append(self.env.data.qvel[5])

        self.previer_cmd_vec = self.current_cmd_vec
        self.current_cmd_vec = self.env.controller.get()
        self.previer_cmd_vec_norm = np.linalg.norm(self.previer_cmd_vec)

        self.previous_action_deque.append(action)

    @property
    def previous_action(self):
        return np.array(self.previous_action_deque).flatten()

    @property
    def joint_diff(self):
        return np.array(self.env.data.qpos[-12:]) - np.array(np.squeeze(self.env.model.key_qpos)[-12:])
    
    @property
    def vel_vec(self):
        return np.array([self.robot_x_lin_vel.mean, self.robot_y_lin_vel.mean, self.robot_z_ang_vel.mean])
    
    @property
    def cmd_vec(self):
        return np.array(self.current_cmd_vec)
    
    @property
    def vel_diff(self):
        return np.array(self.cmd_vec) - np.array(self.vel_vec)
    
    @property
    def foot_landed(self):
        return np.array(self.foot_landed_time > 0)
    
    @property
    def foot_lifted(self):
        return np.array(self.foot_lifted_time > 0)
    
    @property
    def foot_state(self):
        return sum(int(b) << (len(self.feet) - 1 - i) for i, b in enumerate(self.foot_landed))

    @property
    def are_foot_touching_ground(self):
        are_touching = []
        for foot_id in self._foot_ids:
            is_touching = False
            for i in range(self.env.data.ncon):
                c = self.env.data.contact[i]
                is_match = ((c.geom1 == foot_id and c.geom2 == self._floor_id) or
                            (c.geom1 == self._floor_id and c.geom2 == foot_id))
                if is_match:
                    # out = np.zeros(6, dtype=np.float64)
                    # mujoco.mj_contactForce(env.model, env.data, i, out)
                    # foot_fz = out[2]
                    # if foot_fz > 5.0:
                    #     is_touching = True
                    # break

                    is_touching = True
                    break
            
            are_touching.append(is_touching)
        return are_touching
    
    def get_foot_lin_vel(self):
        foot_lin_vel = np.zeros(len(self.feet))
        for i, foot_id in enumerate(self._foot_ids):
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(self.env.model, self.env.data, mujoco.mjtObj.mjOBJ_GEOM,
                                     foot_id, vel, 0)
            foot_lin_vel[i] = np.linalg.norm(vel[0:3])
        return foot_lin_vel
    
    def get_foot_fz(self):
        foot_contract_fz = np.zeros(len(self._foot_ids))
        for i in range(self.env.data.ncon):
            c = self.env.data.contact[i]
            foot_index = None

            if c.geom1 in self._foot_ids and c.geom2 == self._floor_id:
                foot_index = self._foot_ids.index(c.geom1)
            elif c.geom2 in self._foot_ids and c.geom1 == self._floor_id:
                foot_index = self._foot_ids.index(c.geom2)

            if foot_index is not None:
                out = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.env.model, self.env.data, i, out)
                foot_fz = out[2]
                foot_contract_fz[foot_index] = foot_fz
        return foot_contract_fz
    
    def get_rotation_matrix(self):
        quaternion = self.env.data.qpos.flat[3:7]
        mat = np.zeros((9, 1)) # R00 R01 R02 R10 R11 R12 R20 R21 R22
        mujoco.mju_quat2Mat(mat, quaternion) # Convert quaternion to 3D rotation matrix
        return mat.reshape(3, 3)
    
    def get_gravity_projection(self):
        robot_z_unit_vector = self.rotation_matrix[:,2]
        g_vector = self.env.model.opt.gravity
        g_robot_z = np.dot(robot_z_unit_vector, g_vector)
        g_robot_xoy_projection = np.sqrt(max(np.sum(np.square(g_vector)) - np.square(g_robot_z), 0.))
        return g_robot_xoy_projection
