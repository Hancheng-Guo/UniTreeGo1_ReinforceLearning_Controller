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

    def step(self):
        eps = self.rng.standard_normal()
        self.x += -self.theta * self.x * self.dt + np.sqrt(self.dt) * eps
        y = (np.sum(self.memory) + self.x) / self.order
        self.memory.append(y)
        return np.clip(y / (3 * self.sigma), -1.0, 1.0)
    
    def reset(self):
        self.x = 0.
        self.memory.clear()


class UniTreeGo1ControlGenerator:
    def __init__(self, env,
                 generator_theta=1.0,
                 generator_smooth_order=5,
                 generator_schedule=[],
                 **kwargs):
        self.env = env
        self.dt = env.dt * env.frame_skip
        self.schedule = generator_schedule
        self.controllers = [
            OUProcess(theta=generator_theta, dt=self.dt, order=generator_smooth_order) for _ in self.schedule
        ]

    def __len__(self):
        return len(self.controllers)

    def get(self):
        control_vector = []
        for i, controller in enumerate(self.controllers):
            amp = self.schedule[i]["amp"][int(self.env.stage)]
            avg = self.schedule[i]["avg"][int(self.env.stage)]
            control_item = np.clip(amp * controller.step() + avg,
                                   self.schedule[i]["clip"][0],
                                   self.schedule[i]["clip"][1])
            control_vector.append(control_item)
        return np.array(control_vector)
    
    def reset(self):
        for controller in self.controllers:
            controller.reset()


class UniTreeGo1ControlROS:
    def __init__(self, env, **kwargs):
        self.env = env

    def __len__(self):
        return 3

    def get(self):
        return np.concatenate((4, 0, 0))

    def reset(self):
        pass