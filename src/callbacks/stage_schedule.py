from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from collections import deque
import numpy as np
import os
from enum import IntEnum
from src.utils.decays import StepGain
from torch.utils.tensorboard import SummaryWriter


class Stage(IntEnum):
    early = 0
    mid = 1
    late = 2
    done = 3
    
class StageScheduleCallback(BaseCallback):
    def __init__(self, 
                 base_stage = None,
                 verbose = 0,
                 **kwargs):
        super().__init__(verbose)
        self.base_stage = base_stage
        self.stage = None
        self.winlen = None

        self.state_loop_min = None
        self.state_loop_tmp = None
        self.x_velocities = None
        self.alive_fun = StepGain({0.:Stage.early,
                                   150.:Stage.mid,
                                   500.:Stage.late,
                                   800.:Stage.done})
        self.in_state_loop_fun = StepGain({-1.:Stage.mid,
                                           0.6:Stage.late,
                                           0.7:Stage.done})
        self.x_velocities_fun = StepGain({-5.0:Stage.mid,
                                          3.0:Stage.late,
                                          5.0:Stage.done})

    def _on_training_start(self):
        self.winlen = self.model.n_steps * self.model.n_envs
        self.x_velocities = [0 for _ in range(self.model.n_envs)]
        self.state_loop_min = [np.inf for _ in range(self.model.n_envs)]
        self.state_loop_tmp = [-1 for _ in range(self.model.n_envs)]
        if self.base_stage is not None:
            self.stage = np.load(self.base_stage)
        else:
            self.stage = Stage.early
        return True
    
    def _on_rollout_start(self):
        for env in self.model.env.venv.envs:
            env.env.env.env.env.stage = self.stage
        return True
    
    def _on_step(self):
        for i, env in enumerate(self.model.env.venv.envs):
            self.x_velocities[i] += env.env.env.env.env.reward.forward_info["x_velocity"]
            if env.env.env.env.env.reward.state_info["state_loop_time"] != (self.state_loop_tmp[i] + 1):
                self.state_loop_min[i] = min(self.state_loop_min[i], self.state_loop_tmp[i])
            self.state_loop_tmp[i] = env.env.env.env.env.reward.state_info["state_loop_time"]
            
        return True
    
    def _on_rollout_end(self):

        has_ep_mean = len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0
        ep_len_mean = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]) if has_ep_mean else 0
        # ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]) if has_ep_mean else 0
        stage_alive = self.alive_fun(ep_len_mean)
        stage_pct_state_loop = self.in_state_loop_fun(min(self.state_loop_min) / self.model.n_steps)
        stage_x_velocity = self.x_velocities_fun(np.mean(self.x_velocities) / self.model.n_steps)

        self.stage = max(min(stage_x_velocity, stage_pct_state_loop, stage_alive), self.stage)
        return True
