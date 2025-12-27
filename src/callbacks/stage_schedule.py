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
    trot_a = 1
    trot_b = 2
    trot_c = 3
    canter_a = 4
    canter_b = 5
    gallop_a = 6
    gallop_b = 7
    done = 8
    
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
            {0.0:   Stage.idle,
             500.0: Stage.trot_a,
             800.0: Stage.done})
        
        self.robot_x_velocity = None
        self.robot_y_velocity = None
        self.z_angular_velocity = None
        self.robot_x_velocity_fun = StepGain({0.0: 0, 0.9: 1})
        self.robot_y_velocity_fun = StepGain({0.0: 0, 0.9: 1})
        self.z_angular_velocity_fun = StepGain({0.0: 0, 0.75: 1})

    def _on_training_start(self):
        self.winlen = self.model.n_steps * self.model.n_envs
        self.ep_lengths = deque([], maxlen=100)
        self.robot_x_velocity = deque([], maxlen=self.winlen)
        self.robot_y_velocity = deque([], maxlen=self.winlen)
        self.z_angular_velocity = deque([], maxlen=self.winlen)
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
            reward_info = env.env.env.env.env.reward.reward_info
            self.robot_x_velocity.append(reward_info["robot_x_velocity_l2_exp"])
            self.robot_y_velocity.append(reward_info["robot_y_velocity_l2_exp"])
            self.z_angular_velocity.append(reward_info["z_angular_velocity_l2_exp"])
        return True
    
    def _on_rollout_end(self):
        stage_robot_x_velocity = (int(self.stage) +
            self.robot_x_velocity_fun(np.mean(self.robot_x_velocity)))
        
        stage_robot_y_velocity = (int(self.stage) +
            self.robot_y_velocity_fun(np.mean(self.robot_y_velocity)))
        
        stage_z_angular_velocity = (int(self.stage) +
            self.z_angular_velocity_fun(np.mean(self.z_angular_velocity)))
        
        stage_ep_lengths = self.ep_lengths_fun(np.mean(self.ep_lengths))

        self.stage = max(min(stage_ep_lengths,
                             stage_robot_x_velocity,
                             stage_robot_y_velocity,
                             stage_z_angular_velocity),
                         self.stage)
        
        info = {
            "stage": self.stage,
            "stage_ep_lengths": stage_ep_lengths,
            "stage_robot_x_velocity": stage_robot_x_velocity,
            "stage_robot_y_velocity": stage_robot_y_velocity,
            "stage_z_angular_velocity": stage_z_angular_velocity,
        }
        if info:
            # Find max widths
            key_width = max(map(len, info.keys()))
            val_width = max(map(len, map("{:.3f}".format, info.values())))
            # Write out the data
            dashes = "-" * (key_width + val_width + 7)
            lines = [dashes]
            for key, value in info.items():
                key_space = " " * (key_width - len(key))
                val_space = " " * (val_width - len("{:.3f}".format(value)))
                lines.append(f"| {key}{key_space} | {"{:.3f}".format(int(value*1000)/1000)}{val_space} |")
            lines.append(dashes)
            for line in lines:
                print(line)

        return True
