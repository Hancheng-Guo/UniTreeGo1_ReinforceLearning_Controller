from stable_baselines3.common.callbacks import BaseCallback
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LogData:
    reward_sum: Optional[float] = 0
    # imu: Optional[List[float]] = None
    # encoders: Optional[List[float]] = None
    # battery: Optional[float] = None

def init_log():
    return LogData()
    
def update_log(log_data, action, obs_predict, obs, reward, terminated, truncated, info):
    log_data.reward_sum += reward
    return log_data


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_training_start(self) -> bool:
        return True

    def _on_rollout_start(self) -> bool:
        return True

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        return True
    
    def _on_rollout_end(self) -> bool:
        self.bar.clear()
        return True
    