import numpy as np
from collections import deque


class OUProcess:
    def __init__(self, theta=0.1, dt=0.02, smooth_len=15, blend_len=50, seed=None):
        self.x = 0.
        self.theta = float(theta)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed=seed)
        self.sigma = 1
        self.std = np.sqrt(self.sigma**2 / (2 * self.theta - self.theta**2 * self.dt))
        self.smooth_len = smooth_len
        self.memory = deque(np.zeros(self.smooth_len), maxlen=self.smooth_len)
        self.blend_len = blend_len
        self.n_steps = 0

    def step(self, amp, avg):
        self.n_steps += 1
        eps = self.rng.standard_normal()
        self.x += -self.theta * self.x * self.dt + self.sigma * np.sqrt(self.dt) * eps
        self.memory.append(np.clip(self.x / (2 * self.std), -1, 1) * amp + avg)
        return np.mean(self.memory) * min(self.n_steps, self.blend_len) / self.blend_len
    
    def reset(self):
        self.x = 0.
        self.n_steps = 0
        self.memory = deque(np.zeros(self.smooth_len), maxlen=self.smooth_len)
        return 0.