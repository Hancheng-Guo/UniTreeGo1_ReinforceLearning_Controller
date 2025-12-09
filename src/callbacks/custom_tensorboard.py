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
    }


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.writer = None
        self.log_freq = CONFIG["train"]["custom_log_freq"]
        self.rollout_index = None
        self.data = None

    def _on_training_start(self) -> bool:
        self.writer = SummaryWriter(os.path.join(self.logger.dir, "tmp"))
        self.log_freq = (-self.log_freq % self.model.n_envs) + self.log_freq
        return True
    
    def _on_rollout_start(self) -> bool:
        self.rollout_index = self.num_timesteps
        self._data_reset()
        self._nan_dump()
        return True

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for i in range(self.model.n_envs):
            for key, _ in target_items.items():
                self.data[i][key] += infos[i][key]

        if (self.num_timesteps - self.rollout_index) % self.log_freq == 0:
            self._data_dump()
            self._data_reset()
        return True

    def _on_training_end(self):
        self.writer.close()
        files = os.listdir(os.path.join(self.logger.dir, "tmp"))
        for file in files:
            shutil.move(os.path.join(self.logger.dir, "tmp", file), self.logger.dir)
        shutil.rmtree(os.path.join(self.logger.dir, "tmp"))
        return True
    
    def _data_reset(self):
        self.data = [{key: value for key, value in target_items.items()} for _ in range(self.model.n_envs)]

    def _data_dump(self):
        for i in range(self.model.n_envs):
            for key, _ in target_items.items():
                self.writer.add_scalar(
                    f"custom/smooth_{key}_{i}",
                    self.data[i][key] / self.log_freq,
                    self.num_timesteps)
        self.writer.flush()
                
    def _nan_dump(self):
        for i in range(self.model.n_envs):
            for key, _ in target_items.items():
                self.writer.add_scalar(f"custom/smooth_{key}_{i}", float("nan"), self.num_timesteps)
        self.writer.flush()