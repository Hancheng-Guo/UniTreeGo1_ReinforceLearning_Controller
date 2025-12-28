import numpy as np

from src.control.command_generate.ornstein_uhlenbeck import OUProcess


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
            control_item = np.clip(controller.step(amp, avg),
                                   self.schedule[i]["clip"][0],
                                   self.schedule[i]["clip"][1])
            control_vector.append(control_item)
        return np.array(control_vector)
    
    def reset(self):
        for controller in self.controllers:
            controller.reset()
