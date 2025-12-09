from stable_baselines3.common.callbacks import BaseCallback
from src.utils.progress_bar import ProgressBar


class ProgressBarCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.bar = None
        self.current_step = None

    def _on_training_start(self) -> bool:
        self.bar = ProgressBar(total=self.model.n_steps * self.model.n_envs,
                               custom_str="Rollout")
        return True

    def _on_rollout_start(self) -> bool:
        self.bar.reset()
        self.current_step = 0
        return True

    def _on_step(self) -> bool:
        self.current_step += 1
        self.bar.update(self.current_step * self.model.n_envs)
        return True
    
    def _on_rollout_end(self) -> bool:
        self.bar.clear()
        return True
