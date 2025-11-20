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
