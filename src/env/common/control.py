from src.control.base import UniTreeGo1ControlOUGenerator, UniTreeGo1ControlUDP
from src.control.base import UniTreeGo1ControlStepGenerator, UniTreeGo1ControlConstGenerator

class UniTreeGo1Control:
    def __init__(self, env, control_config):
        self.env = env
        if control_config["control_type"] == "udp":
            self.controller = UniTreeGo1ControlUDP(self.env, **control_config)
        elif control_config["control_type"] == "step":
            self.controller = UniTreeGo1ControlStepGenerator(self.env, **control_config)
        elif control_config["control_type"] == "smooth":
            self.controller = UniTreeGo1ControlOUGenerator(self.env, **control_config)
        else: #"const"
            self.controller = UniTreeGo1ControlConstGenerator(self.env, **control_config)
            
    
    def __len__(self):
        return len(self.controller)

    def get(self):
        return self.controller.get()
    
    def reset(self):
        self.controller.reset()