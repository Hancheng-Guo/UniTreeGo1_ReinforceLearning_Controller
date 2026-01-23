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
        self.controllers = {
            key: OUProcess(theta=generator_theta, dt=self.dt, order=generator_smooth_order) for key in self.schedule.keys()
        }

    def __len__(self):
        return len(self.controllers)

    def get(self):
        control_dict = {}
        for i, (key, controller) in enumerate(self.controllers.items()):
            amp = self.schedule[key]["amp"][int(self.env.stage)]
            avg = self.schedule[key]["avg"][int(self.env.stage)]
            control_item = np.clip(controller.step(amp, avg),
                                   self.schedule[key]["clip"][0],
                                   self.schedule[key]["clip"][1])
            control_dict[key] = control_item
        return self._dict_to_vector(self._z_angular_velocity_check(control_dict))
    
    def reset(self):
        control_vector = []
        for _, controller in self.controllers.items():
            control_vector.append(controller.reset())
        return np.array(control_vector)
    
    def _z_angular_velocity_check(self, control_dict) -> dict:
        velocity = np.sqrt(control_dict["robot_x_velocity"]**2 + control_dict["robot_y_velocity"]**2)
        z_angular_velocity_mx = np.pi * (0.85 * np.exp(-velocity**2) + 0.15)
        control_dict["z_angular_velocity"] = np.clip(control_dict["z_angular_velocity"],
                                                     -z_angular_velocity_mx, z_angular_velocity_mx)
        return control_dict
    
    def _dict_to_vector(self, control_dict) -> np.array:
        return np.array([control_dict[key] for key in self.controllers.keys()])
