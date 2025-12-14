from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from collections import deque
import numpy as np
import os
from enum import IntEnum
from src.utils.decays import StepGain

# region | Reward Scheduler

class Stage(IntEnum):
    early = 0
    mid = 1
    late = 2

# class UniTreeGo1StageScheduler:
#     def __init__(self, env,
#                  max_winlen: int = 50):
#         self.env = env
#         self.stage = None
#         self.max_winlen = max_winlen
#         self.x_velocities = deque(maxlen=self.max_winlen)
#         self.in_state_loop = deque(maxlen=self.max_winlen)
#         self.x_velocities_fun = StepGain({1.:Stage.early,
#                                           2.:Stage.mid,
#                                           3.:Stage.late})
#         self.in_state_loop_fun = StepGain({0.2:Stage.early,
#                                            0.6:Stage.late})

#     def __call__(self):
#         return self.stage

#     def reset(self):
#         self.stage = float(Stage.early)
#         self.x_velocities.clear()
#         self.in_state_loop.clear()
    
#     def update(self):
#         self.x_velocities.append(self.env.forward_info["x_velocity"])
#         self.in_state_loop.append(self.env.state_info["state_loop_time"] >= 0)

#         stage_x_velocities = self.x_velocities_fun(np.sum(self.x_velocities)/self.max_winlen)
#         stage_in_state_loop = self.in_state_loop_fun(np.sum(self.in_state_loop)/self.max_winlen)

#         self.stage = min(stage_x_velocities, stage_in_state_loop)
#         return self.stage
    
class StageScheduleCallback(BaseCallback):
    def __init__(self, 
                 base_stage = None,
                 verbose = 0,
                 **kwargs):
        super().__init__(verbose)
        self.base_stage = base_stage
        self.stage = None
        self.winlen = None
        self.x_velocities = None
        self.in_state_loop = None
        self.x_velocities_fun = StepGain({1.:Stage.early,
                                          2.:Stage.mid,
                                          3.:Stage.late})
        self.in_state_loop_fun = StepGain({0.2:Stage.early,
                                           0.6:Stage.late})

    def _on_training_start(self):
        self.winlen = self.model.n_steps * self.model.n_envs
        self.x_velocities = deque(maxlen=self.winlen)
        self.in_state_loop = deque(maxlen=self.winlen)
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
        for env in self.model.env.venv.envs:
            self.x_velocities.append(env.env.env.env.env.forward_info["x_velocity"])
            self.in_state_loop.append(env.env.env.env.env.state_info["state_loop_time"] >= 0)
        return True
    
    def _on_rollout_end(self):
        # if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
        #     ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        #     ep_len_mean = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        stage_x_velocities = self.x_velocities_fun(np.sum(self.x_velocities)/self.winlen)
        stage_in_state_loop = self.in_state_loop_fun(np.sum(self.in_state_loop)/self.winlen)

        self.stage = max(min(stage_x_velocities, stage_in_state_loop), self.stage)
        return True
