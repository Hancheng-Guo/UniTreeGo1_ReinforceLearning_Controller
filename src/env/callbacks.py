import os
import shutil
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule, ConstantSchedule
# from src.render.render_tensorboard import init_log, update_log
from src.utils.update_checkpoints_tree import update_checkpoints_tree
from src.config.config import CONFIG, save_CONFIG

# self.locals.info has info of customize-env


class CustomCheckpointCallback(BaseCallback):
    def __init__(self,
                 save_name: str,
                 save_dir: str,
                 base_name: str = None,
                 note: str = "",
                 save_freq: int = CONFIG["train"]["checkpoint_freq"],
                 save_vecnormalize: bool = True,
                 verbose: int = 2,
                 ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_name = save_name
        self.save_dir = save_dir
        self.base_name = base_name
        self.note = note
        self.save_vecnormalize = save_vecnormalize
        self.save_count = 1

    def _on_training_start(self):
        self.save_freq = (-self.save_freq % self.model.n_envs) + self.save_freq

    @property
    def _counted_save_name(self) -> str:
        return f"{self.save_name}_{self.save_count}"

    def _save_checkpoint(self) -> bool:
        lr_schedule_tmp = self.model.lr_schedule
        lr_tmp = self.model.lr_schedule(self.model._current_progress_remaining)
        self.model.learning_rate = lr_tmp
        self.model.lr_schedule = FloatSchedule(ConstantSchedule(lr_tmp))
        # save model
        model_path = os.path.join(self.save_dir, f"mdl_{self._counted_save_name}.zip")
        self.model.save(model_path)
        if self.verbose >= 2:
            print(f"Saving model to {model_path}")
        # save vecnormalized env
        env_path = os.path.join(self.save_dir, f"env_{self._counted_save_name}.pkl")
        if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            self.model.get_vec_normalize_env().save(env_path)
            if self.verbose >= 2:
                print(f"Saving vecnormalized env to {env_path}")
        # update checkpoints tree
        update_checkpoints_tree(child=self._counted_save_name, parent=self.base_name, note=self.note)
        self.base_name = self._counted_save_name
        # save config
        config_path = os.path.join(self.save_dir, f"cfg_{self._counted_save_name}.yaml")
        save_CONFIG(config_path)
        if self.verbose >= 2:
            print(f"Saving config to {config_path}")
        # save origin py.file of customize env
        backup_path = os.path.join(self.save_dir, f"bkp_{self._counted_save_name}.py")
        shutil.copy2(CONFIG["path"]["env_class_py"], backup_path)
        if self.verbose >= 2:
            print(f"Saving origin py.file of customize env to {backup_path}")

        self.model.lr_schedule = lr_schedule_tmp
        self.save_count += 1
        return True
        
    def _on_step(self) -> bool:
        if (self.n_calls * self.model.n_envs) % self.save_freq == 0:
            self._save_checkpoint()
        return True

    def _on_training_end(self) -> bool:
        self._save_checkpoint()
        return True
    

class AdaptiveLRCallback(BaseCallback):
    def __init__(self, smooth_step_len=2000,
                 kl_min=0.01, kl_max=0.1,
                 lr_min=1e-6, lr_max=5e-3,
                 factor=2, verbose=0):
        super().__init__(verbose)
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
        return True
        
    def _on_step(self) -> bool:
        current_lr = self.model.lr_schedule(self.model._current_progress_remaining)
        if self.smooth_step_left > 0:
            next_lr = current_lr + (self.target_lr - current_lr) / self.smooth_step_left
            self.current_lr = next_lr
            self.smooth_step_left -= 1
        return True
    
    def _on_rollout_end(self) -> bool:
        kl = self.logger.name_to_value.get("train/approx_kl")
        if kl is not None:
            current_lr = self.model.lr_schedule(self.model._current_progress_remaining)
            if current_lr == self.target_lr:
                if kl < self.kl_min:
                    self.target_lr = min(current_lr * self.factor, self.lr_max)
                    self.smooth_step_left = self.smooth_step_len
                elif kl > self.kl_max:
                    self.target_lr = max(current_lr / self.factor, self.lr_min)
                    self.smooth_step_left = self.smooth_step_len
        return True


class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_rollout_steps = None
        self.current_step = None
        self.barlen = 50
        self.diglen = None
        self.splen = 3
        self.sppos = None

    def _on_training_start(self):
        self.total_rollout_steps = self.model.n_steps * self.model.n_envs
        self.diglen = len(f"{self.total_rollout_steps}")

    def _on_rollout_start(self) -> None:
        self.current_step = 0
        self.sppos = 0
        return True

    def _on_step(self) -> bool:
        self.current_step += 1
        frac = self.current_step * self.model.n_envs / self.total_rollout_steps
        a_len = int(frac * self.barlen)

        a = "\u2588" * a_len
        b = "\u2591" * (self.barlen + self.splen) + " " * self.splen
        b = (b[-(self.current_step % (self.barlen + 2 * self.splen)):] + 
             b[:-(self.current_step % (self.barlen + 2 * self.splen))])
        b = b[a_len:-(2 * self.splen)]
        c = f"{(self.current_step * self.model.n_envs):>{self.diglen}d}"
        d = f"{self.total_rollout_steps:>{self.diglen}d}"
        print(f" Rollout {(frac * 100):^3.0f}% {a}{b} {c}/{d} steps", end="\r")

        return True
    
    def _on_rollout_end(self) -> bool:
        print("\033[2K\r", end="")
        return True
