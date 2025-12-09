import os
import shutil
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule, ConstantSchedule
from src.utils.update_checkpoints_tree import update_checkpoints_tree
from src.config.config import CONFIG, save_CONFIG


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

    def _on_training_start(self)  -> bool:
        self.save_freq = (-self.save_freq % self.model.n_envs) + self.save_freq
        return True

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