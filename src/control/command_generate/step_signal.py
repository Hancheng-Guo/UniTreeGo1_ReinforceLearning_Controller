import numpy as np
from collections import deque


class SSProcess:
    def __init__(self, switch_time=2.5, switch_step=1.0, dt=0.01, seed=None):
        self.x = 0.
        self.t = 0.
        self.dt = float(dt)
        self.switch_time = float(switch_time)
        self.switch_step = float(switch_step)
        self.x_target = 0.
        self.rng = np.random.default_rng(seed=seed)
        

    def step(self, amp, avg, smooth_time):
        self.t += self.dt
        if self.t > self.switch_time:
            self.t = 0
            self.x = self.x_target
            rand = self.rng.random()
            if rand < 0.7:
                self.x_target += self.switch_step
            elif rand < 0.9:
                self.x_target -= self.switch_step
            else:
                self.x_target = 0
            self.x_target = np.clip(self.x_target, avg - amp, avg + amp)
        if self.t >= smooth_time:
            return self.x_target
        else:
            return (self.x +
                    (self.x_target - self.x) *
                    (3 * (self.t/smooth_time)**2 - 2 * (self.t/smooth_time)**3))
    
    def reset(self):
        self.x = 0.
        self.t = 0.
        self.x_target = 0.
        return self.x
    
