import numpy as np

from src.control.command_generate.ornstein_uhlenbeck import OUProcess


class UniTreeGo1ControlGenerator:
    def __init__(self, env,
                 generator_theta=1.0,
                 generator_smooth_len=7,
                 generator_blend_len=25,
                 generator_schedule=[],
                 generator_random=False,
                 disable_len=50, 
                 **kwargs):
        self.env = env
        self.dt = env.dt * env.frame_skip
        self.schedule = generator_schedule
        self.random = generator_random
        self.controllers = {
            key: OUProcess(theta=generator_theta,
                           dt=self.dt,
                           smooth_len=generator_smooth_len,
                           blend_len=generator_blend_len) for key in self.schedule.keys()
        }
        self.disable_len = disable_len
        self.disable_count = 0
        self.controllers_enable = [True for _ in self.schedule.keys()]
        self.p_controllers_disable = 0.05
        self.p_disable = 0.001 / self.disable_len

    def __len__(self):
        return len(self.controllers)

    def get(self):
        if self.random and (np.random.rand() < self.p_disable):
            self.disable_count = self.disable_len

        if self.disable_count <= 0:
            control_dict = {}
            for i, (key, controller) in enumerate(self.controllers.items()):
                if self.controllers_enable[i]:
                    amp = self.schedule[key]["amp"][int(self.env.stage)]
                    avg = self.schedule[key]["avg"][int(self.env.stage)]
                    control_item = np.clip(controller.step(amp, avg),
                                        self.schedule[key]["clip"][0],
                                        self.schedule[key]["clip"][1])
                    control_dict[key] = control_item
                else:
                    control_dict[key] = 0.
            return self._dict_to_vector(self._z_angular_velocity_check(control_dict))
        else:
            self.disable_count -= 1
            return np.zeros(len(self.controllers))

    
    def reset(self):
        if self.random:
            for i in range(len(self.controllers_enable)):
                self.controllers_enable[i] = (np.random.rand() < self.p_controllers_disable)

        control_vector = []
        for i, (_, controller) in enumerate(self.controllers.items()):
            if self.controllers_enable[i]:
                control_vector.append(controller.reset())
            else:
                control_vector.append(0.)
        return np.array(control_vector)
    
    def _z_angular_velocity_check(self, control_dict) -> dict:
        if ("z_angular_velocity" not in control_dict or
            "robot_x_velocity" not in control_dict or
            "robot_y_velocity" not in control_dict):
            return control_dict
        
        velocity = np.sqrt(control_dict["robot_x_velocity"]**2 + control_dict["robot_y_velocity"]**2)
        z_angular_velocity_mx = np.pi * (0.85 * np.exp(-velocity**2) + 0.15)
        control_dict["z_angular_velocity"] = np.clip(control_dict["z_angular_velocity"],
                                                     -z_angular_velocity_mx, z_angular_velocity_mx)
        return control_dict
    
    def _dict_to_vector(self, control_dict) -> np.array:
        return np.array([control_dict[key] for key in self.controllers.keys()])
