from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from collections import deque
import numpy as np
import os
from enum import IntEnum
from src.utils.decays import StepGain
from torch.utils.tensorboard import SummaryWriter


class Stage(IntEnum):
    idle = 0
    straight_trot = 1
    trot = 2
    canter = 3
    gallop = 4
    
class StageScheduleCallback(BaseCallback):
    def __init__(self, 
                 base_stage = None,
                 control_generator_schedule = None,
                 verbose = 0,
                 **kwargs):
        super().__init__(verbose)
        self.base_stage = base_stage
        self.stage = None
        self.winlen = None
        self.control_generator_schedule = control_generator_schedule

        self.ep_lengths = None
        self.ep_lengths_fun = StepGain(
            {0.0:  Stage.idle,
             400.0:Stage.straight_trot,
             450.0:Stage.trot,
             500.0:Stage.canter,
             550.0:Stage.gallop})
        
        self.robot_x_velocity_l2_exp = None
        self.robot_x_velocity_l2_exp_fun = StepGain({0.0: 0, 0.8: 1})

    def _on_training_start(self):
        self.winlen = self.model.n_steps * self.model.n_envs
        self.ep_lengths = deque([], maxlen=100)
        self.robot_x_velocity_l2_exp = deque([], maxlen=self.winlen)
        if self.base_stage is not None:
            self.stage = np.load(self.base_stage)
        else:
            self.stage = Stage.idle
        return True
    
    def _on_rollout_start(self):
        for env in self.model.env.venv.envs:
            env.env.env.env.env.stage = self.stage
        return True
    
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_lengths.append(info["episode"]["l"])
        for env in self.model.env.venv.envs:
            self.robot_x_velocity_l2_exp.append(
                env.env.env.env.env.reward.reward_info["robot_x_velocity_l2_exp"])
        return True
    
    def _on_rollout_end(self):
        stage_robot_x_velocity_l2_exp = (int(self.stage) +
            self.robot_x_velocity_l2_exp_fun(np.mean(self.robot_x_velocity_l2_exp)))
        stage_ep_lengths = self.ep_lengths_fun(np.mean(self.ep_lengths))
        self.stage = max(min(stage_robot_x_velocity_l2_exp,  stage_ep_lengths),
                         self.stage)
        print(np.mean(self.ep_lengths))
        return True
