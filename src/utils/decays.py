import numpy as np
from collections.abc import Iterable
from functools import partial


def radial_decay(x, center=0, boundary=1, boundary_value=0.01):
    if isinstance(boundary, Iterable):
        if x <= center:
            boundary = center - boundary[0]
        else:
            boundary = boundary[1] - center
    reward_alpha = -np.log(boundary_value) / np.square(boundary)
    decayed_value = np.exp(-reward_alpha * np.square(np.abs(x - center)))
    return decayed_value

def clip_exp_decay(x, alpha=0.2, x_shift=0):
    if x < 0:
        return 0
    return np.exp(-alpha * (x - x_shift))

def clip_exp_gain(x, alpha=0.2, x_shift=0):
    if x < max(0,x_shift):
        return 0
    return np.exp(-1 / (alpha * (x - x_shift)))

def clip_step_decay(x, alpha=0.2, x_shift=0):
    if x < 0:
        return 0
    elif x >= 0 and x <= x_shift:
        return 1
    return np.exp(-alpha * (x - x_shift)**2)

def smoothstep_Cinf(x, a=0, b=1, scale=1, alpha=1):
    assert b > a
    assert alpha > 0

    x = np.asarray(x, dtype=float)
    y = np.empty_like(x)
    y[x <= (a + 1e-5)] = 0.0
    y[x >= (b - 1e-5)] = 1.0

    m = (x > (a + 1e-5)) & (x < (b - 1e-5))
    k = (x[m] - a) / (b - a)
    eta_k  = np.exp(-1 / (alpha * k))
    eta_1k = np.exp(-1 / (alpha * (1 - k)))
    y[m] = eta_k / (eta_k + eta_1k)

    return y * scale

def _const(x, c=0, *args, **kwargs):
    x = np.asarray(x, dtype=float)
    return np.ones_like(x) * c

class StepGain():
    def __init__(self, points={}, alpha=1):
        self._alpha = alpha
        self._points = points
        self._increment =self._get_increment(self._points)
        self.fun = self._get_fun(self._increment)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        for i in range(len(self.fun)):
            y += self.fun[i](x)
        return y
    
    def _get_increment(self, point_dict):
        increment = {}
        prev_y = None
        for x, y in sorted(point_dict.items()):
            if prev_y is None:
                increment[x] = y
            else:
                increment[x] = y - prev_y
            prev_y = y
        return increment
    
    def _get_fun(self, increment_dict):
        fun = []
        prve_key = None
        for idx, (key, value) in enumerate(increment_dict.items()):
            if idx == 0:
                fun.append(partial(_const, c=value))
            else:
                fun.append(
                    partial(smoothstep_Cinf, a=prve_key, b=key, scale=value, alpha=self._alpha))
            prve_key = key
        return fun
    