import numpy as np


class UniTreeGo1ControlROS:
    def __init__(self, env, **kwargs):
        self.env = env

    def __len__(self):
        return 3

    def get(self):
        return np.concatenate((4, 0, 0))

    def reset(self):
        pass