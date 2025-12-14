import os
import shutil
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

from src.config.config import CONFIG


target_items = {
    "reward_forward": 0,
    "reward_state": 0,
    "reward_posture": 0,
    "state_loop_time": 0,
    "stage": 0,
    }


class CustomTensorboardCallback(BaseCallback):
    def __init__(self,
                 log_freq: int = 2048,
                 verbose=0,
                 **kwargs):
        super().__init__(verbose)
        self.writer = None
        self.log_freq = log_freq
        self.rollout_index = None
        self.data = None

    def _on_training_start(self) -> bool:
        self.writer = [SummaryWriter(os.path.join(self.logger.dir,  f"env_{env_id}"))
                       for env_id in range(self.model.n_envs)]
        self.log_freq = min((-self.log_freq % self.model.n_envs) + self.log_freq,
                            self.model.n_envs * self.model.n_steps)
        return True
    
    def _on_rollout_start(self) -> bool:
        self.rollout_index = self.num_timesteps
        self._data_reset()
        # self._data_split()
        return True

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for env_id in range(self.model.n_envs):
            for key, _ in target_items.items():
                self.data[env_id][key] += infos[env_id][key]

        timesteps_past = self.num_timesteps - self.rollout_index
        if (timesteps_past % self.log_freq == 0) and (timesteps_past != 0):
            self._data_dump()
            self._data_reset()
        return True

    def _on_training_end(self):
        for env_id in range(self.n_envs):
            self.writer[env_id].close()
            files = os.listdir(os.path.join(self.logger.dir, f"env_{env_id}"))
            for file in files:
                shutil.move(os.path.join(self.logger.dir, f"env_{env_id}", file), self.logger.dir)
            shutil.rmtree(os.path.join(self.logger.dir, f"env_{env_id}"))
        return True
    
    def _tb_log_name(self, key, suffix):
        return f"custom/{key}_{suffix}"
    
    def _data_reset(self):
        self.data = [{key: value for key, value in target_items.items()}
                     for _ in range(self.model.n_envs)]

    def _data_dump(self):
        for env_id in range(self.model.n_envs):
            for key, _ in target_items.items():
                self.writer[env_id].add_scalar(
                    self._tb_log_name(key, "mean"),
                    self.data[env_id][key] / self.log_freq * self.model.n_envs,
                    self.num_timesteps)
        self.writer[env_id].flush()
                
    def _data_split(self):
        for env_id in range(self.model.n_envs):
            for key, _ in target_items.items():
                self.writer[env_id].add_scalar(
                    self._tb_log_name(key, "mean"),
                    float("inf"),
                    self.num_timesteps)
        self.writer[env_id].flush()