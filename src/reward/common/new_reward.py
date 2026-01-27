import numpy as np


class NewReward:
    def __init__(self, env, fun, weight, stage_range=[-np.inf, np.inf], **kwargs):
        self.env = env
        self.fun = fun
        self.weight = weight
        self.stage_range = stage_range
        self.__dict__.update(kwargs)
        
    def __call__(self) -> float:
        if (self.env.stage < self.stage_range[0]): return 0, {}
        if (self.env.stage >= self.stage_range[-1]): return 0, {}
        reward, reward_info = self.fun(self)
        return self.weight * reward, reward_info