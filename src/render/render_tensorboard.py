import subprocess
import threading
from dataclasses import dataclass
from typing import Optional, List
from src.config.config import CONFIG


@dataclass
class LogData:
    reward_sum: Optional[float] = 0
    # imu: Optional[List[float]] = None
    # encoders: Optional[List[float]] = None
    # battery: Optional[float] = None


def run_tensorboard():
    result = subprocess.run("tensorboard --logdir " + CONFIG["path"]["tensorboard"], shell=True)
    print(result.stdout)

def init_tensorboard():
    t = threading.Thread(target=run_tensorboard, daemon=True)
    t.start()

def init_log():
    return LogData()
    
def update_log(log_data, action, obs_predict, obs, reward, terminated, truncated, info):
    log_data.reward_sum += reward
    return log_data