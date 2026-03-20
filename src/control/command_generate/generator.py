import numpy as np
from itertools import product
import random

from src.control.command_generate.ornstein_uhlenbeck import OUProcess
from src.control.command_generate.step_signal import SSProcess


class UniTreeGo1ControlGeneratorBase:
    def __init__(self, env, **kwargs):
        self.env = env
        self.dt = env.dt * env.frame_skip
        self.controllers = {}

    def __len__(self):
        return len(self.controllers)
    
    def get(self):
        raise NotImplementedError()
    
    def reset(self):
        raise NotImplementedError()
    
    def _cmd_check(self, control_dict):
        if ("z_angular_velocity" not in control_dict or
            "robot_x_velocity" not in control_dict or
            "robot_y_velocity" not in control_dict):
            return control_dict
        
        x_lin_vel = control_dict["robot_x_velocity"]
        y_lin_vel = control_dict["robot_y_velocity"]
        z_ang_vel = control_dict["z_angular_velocity"]

        y_lin_vel_mx = np.exp(-x_lin_vel**2)
        z_ang_vel_mx = np.exp(-(x_lin_vel**2 + 10 * y_lin_vel**2))

        check_mark = True
        if y_lin_vel > y_lin_vel_mx or y_lin_vel < -y_lin_vel_mx:
            check_mark = False
            control_dict["robot_y_velocity"] = np.clip(y_lin_vel, -y_lin_vel_mx, y_lin_vel_mx)
        if z_ang_vel > z_ang_vel_mx or z_ang_vel < -z_ang_vel_mx:
            check_mark = False
            control_dict["z_angular_velocity"] - np.clip(z_ang_vel, -z_ang_vel_mx, z_ang_vel_mx)

        return control_dict, check_mark
    
    def _dict_to_vector(self, control_dict) -> np.array:
        return np.array([control_dict[key] for key in control_dict.keys()])
    

class UniTreeGo1ControlOUGenerator(UniTreeGo1ControlGeneratorBase):
    def __init__(self, env,
                 generator_theta=1.0,
                 generator_smooth_len=7,
                 generator_blend_len=25,
                 generator_schedule=[],
                 generator_random=False,
                 disable_len=50, 
                 **kwargs):
        super().__init__(env, **kwargs)
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
            control_dict, _ = self._cmd_check(control_dict)
            return self._dict_to_vector(control_dict)
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


class UniTreeGo1ControlStepGenerator(UniTreeGo1ControlGeneratorBase):
    def __init__(self, env,
                 generator_switch_time=10.0,
                 generator_smooth_time=[],
                 generator_schedule=[],
                 generator_random=False,
                 generator_enable_p=[],
                 **kwargs):
        super().__init__(env, **kwargs)
        self.schedule = generator_schedule
        self.switch_time = generator_switch_time
        self.smooth_time = generator_smooth_time

        self.random = generator_random
        self.enable_p = generator_enable_p
        self.controllers = {
            key: SSProcess(switch_time=self.switch_time, switch_step=value["step"],
                           dt=self.dt) for key, value in self.schedule.items()
        }
        self.enable_mask = np.array([False for _ in self.schedule.keys()])

    def get(self):
        control_dict = {}
        smooth_time = self.smooth_time[int(self.env.stage)]
        for i, (key, controller) in enumerate(self.controllers.items()):
            amp = self.schedule[key]["amp"][int(self.env.stage)]
            avg = self.schedule[key]["avg"][int(self.env.stage)]

            control_item = np.clip(controller.step(amp, avg, smooth_time),
                                self.schedule[key]["clip"][0],
                                self.schedule[key]["clip"][1])
            control_dict[key] = control_item * self.enable_mask[i]
        control_dict, _ = self._cmd_check(control_dict)
        control_vector = self._dict_to_vector(control_dict)
        return control_vector

    
    def reset(self):
        if self.random:
            enable_count = 0
            enable_p = 0
            enable_rand = np.random.rand()
            for p in self.enable_p[int(self.env.stage)]:
                if enable_rand < (enable_p + p) and enable_rand >= enable_p:
                    break
                else:
                    enable_count += 1
                    enable_p += p
            self.enable_mask = np.concatenate((np.ones(enable_count, dtype=bool),
                                               np.zeros(len(self.controllers) - enable_count, dtype=bool)))
            np.random.shuffle(self.enable_mask)

        for _, controller in self.controllers.items():
            controller.reset()
        return self.get()


class UniTreeGo1ControlConstGenerator(UniTreeGo1ControlGeneratorBase):
    def __init__(self, env,
                 const_command: dict| None = None,
                 generator_schedule: list| None = None,
                 **kwargs):
        super().__init__(env, **kwargs)
        assert generator_schedule is not None or const_command is not None, "no command provided"
        if const_command is not None:
            assert "robot_x_velocity" in const_command, "robot_x_velocity not exist in command"
            assert "robot_y_velocity" in const_command, "robot_y_velocity not exist in command"
            assert "z_angular_velocity" in const_command, "z_angular_velocity not exist in command"
            self.random = False
            const_command, _ = self._cmd_check(const_command)
            self.const_command = self._dict_to_vector(const_command)
        else:
            self.random = True
            self.schedule = generator_schedule
            self.state_old = None
            self.legal_commands = []
            self.const_command = None
            self.reset()
        assert len(self.const_command) != 0, "const_command is empty"

    def __len__(self):
        if self.random:
            return len(self.schedule)
        else:
            return len(self.const_command)
        
    def get(self):
        return self.const_command

    def reset(self):
        if self.random:
            if self.state_old is None or self.state_old != int(self.env.stage):
                self.state_old = int(self.env.stage)
                self.legal_commands = self._get_legal_commands()
            self.const_command = random.choice(self.legal_commands)
        return self.const_command

    def _get_legal_commands(self):
        command_range = {}
        for name, info in self.schedule.items():
            amp = info["amp"][int(self.env.stage)]
            avg = info["avg"][int(self.env.stage)]
            step = info["step"][int(self.env.stage)]
            command_range[name] = np.arange(avg - amp, avg + amp + step, step)

        keys = list(command_range.keys())

        commands = []
        for values in product(*(command_range[k] for k in keys)):
            commands.append(dict(zip(keys, values)))

        legal_commands = []
        for command in commands:
            _, check_mark = self._cmd_check(command)
            if check_mark:
                legal_commands.append(self._dict_to_vector(command))

        return legal_commands
