import numpy as np
from collections.abc import Iterable


def radial_decay(x, center=0, boundary=1, boundary_value=0.01):
    if isinstance(boundary, Iterable):
        if x <= center:
            boundary = center - boundary[0]
        else:
            boundary = boundary[1] - center
    reward_alpha = -np.log(boundary_value) / np.square(boundary)
    decayed_value = np.exp(-reward_alpha * np.square(np.abs(x - center)))
    return decayed_value

def clip_exp_decay(x, alpha=0.2):
    if x < 0:
        return 0
    return np.exp(-alpha * x)