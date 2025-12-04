import numpy as np


def radial_decay(x, radius=1, radius_value=0.01):
    reward_alpha = -np.log(radius_value) / np.square(radius)
    decayed_value = np.exp(-reward_alpha * np.square(x))
    return decayed_value

def clip_exp_decay(x, alpha=0.2):
    if x < 0:
        return 0
    return np.exp(-alpha * x)