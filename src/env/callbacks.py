from stable_baselines3.common.callbacks import BaseCallback
from time import sleep
from src.tensorboard.init_log import init_log, update_log
from config import CONFIG

class RenderCallback(BaseCallback):
    def __init__(self, demo_env,
                 demo_freq=CONFIG["demo"]["log_freq"],
                 verbose=CONFIG["algorithm"]["verbose"]):
        super().__init__(verbose)
        self.demo_env = demo_env
        self.demo_freq = demo_freq
    
    def _init_step_loop(self):
        obs, info = self.demo_env.reset()
        log_data = init_log()

        def _step_loop():
            nonlocal obs, log_data
            action, obs_predict = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.demo_env.step(action)
            done = terminated or truncated

            log_data = update_log(log_data, action, obs_predict, obs, reward, terminated, truncated, info)

            sleep(0.02)
            return done
        
        def _step_done():
            nonlocal log_data
            for key, value in log_data.__dict__.items():
                self.logger.record(CONFIG["path"]["tensorboard_log"] + key, value)
            self.logger.dump(self.num_timesteps)

        return _step_loop, _step_done


        
    def _on_step(self) -> bool:
        print("current step: %d" % self.num_timesteps)
        if self.num_timesteps % self.demo_freq == 0:
            step_loop, step_done = self._init_step_loop()
            while not step_loop(): pass
            step_done()
        return True