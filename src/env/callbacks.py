from stable_baselines3.common.callbacks import BaseCallback
from src.render.render_tensorboard import init_log, update_log
from src.config.config import CONFIG

# self.locals.info 中有自定义类中的信息

class RenderCallback(BaseCallback):
    def __init__(self, demo_env,
                 demo_freq=CONFIG["demo"]["log_freq"],
                 verbose=CONFIG["algorithm"]["verbose"]):
        super().__init__(verbose)
        self.demo_env = demo_env
        n_stpes_per_epoch = CONFIG["algorithm"]["n_steps"] * CONFIG["train"]["n_envs"]
        self.demo_freq = n_stpes_per_epoch if (demo_freq <= n_stpes_per_epoch) else (demo_freq // n_stpes_per_epoch)
    
    def _init_step_loop(self):
        obs, info = self.demo_env.reset()
        log_data = init_log()

        def _step_loop():
            nonlocal obs, log_data
            action, obs_predict = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.demo_env.step(action)
            done = terminated or truncated

            log_data = update_log(log_data, action, obs_predict, obs, reward, terminated, truncated, info)

            # sleep(0.02)
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
    
class AdaptiveLRCallback(BaseCallback):
    def __init__(self, model,
                 smooth_step_len=200,
                 kl_min=0.01, kl_max=0.25,
                 lr_min=1e-6, lr_max=5e-3,
                 factor=2, verbose=0):
        super().__init__(verbose)
        self.model = model
        self.kl_min = kl_min
        self.kl_max = kl_max
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.factor = factor
        self.target_lr = None
        self.current_lr = None
        self.smooth_step_len = smooth_step_len
        self.smooth_step_left = 0

    def _on_training_start(self):
        current_lr = self.model.lr_schedule(self.model._current_progress_remaining)
        self.target_lr = current_lr
        self.current_lr = current_lr
        def dynamic_lr_schedule(progress):
            return self.current_lr
        self.model.lr_schedule = dynamic_lr_schedule

    def _on_step(self) -> bool:
        self.current_lr = self.model.lr_schedule(self.model._current_progress_remaining)
        if self.smooth_step_left > 0:
            next_lr = self.current_lr + (self.target_lr - self.current_lr) / self.smooth_step_left
            self.current_lr = next_lr
            self.smooth_step_left -= 1
        return True
    
    def _on_rollout_end(self) -> bool:
        kl = self.logger.name_to_value.get("train/approx_kl")
        if kl is not None:
            self.current_lr = self.model.lr_schedule(self.model._current_progress_remaining)
            if self.current_lr == self.target_lr:
                if kl < self.kl_min:
                    self.target_lr = min(self.current_lr * self.factor, self.lr_max)
                    self.smooth_step_left = self.smooth_step_len
                elif kl > self.kl_max:
                    self.target_lr = max(self.current_lr / self.factor, self.lr_min)
                    self.smooth_step_left = self.smooth_step_len
        return True
