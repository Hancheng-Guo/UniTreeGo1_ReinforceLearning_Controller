import numpy as np
from collections import deque


class OUProcess:
    def __init__(self, theta=1.0, dt=0.02, order=5, seed=None):
        self.x = 0.
        self.theta = float(theta)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed=seed)
        self.sigma = np.sqrt(1 / self.theta)
        self.order = order
        self.memory = deque(np.zeros(self.order-1), maxlen=self.order-1)

    def step(self, amp, avg):
        eps = self.rng.standard_normal()
        self.x += -self.theta * self.x * self.dt + np.sqrt(self.dt) * eps
        y = self.x / (3 * self.sigma) * amp + avg
        self.memory.append(y)
        return np.sum(self.memory) / self.order
    
    def reset(self):
        self.x = 0.
        self.memory.clear()