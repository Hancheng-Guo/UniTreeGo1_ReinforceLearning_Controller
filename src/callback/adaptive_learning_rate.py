from stable_baselines3.common.callbacks import BaseCallback


class AdaptiveLRCallback(BaseCallback):
    def __init__(self, smooth_step_len=2000,
                 kl_min=0.01, kl_max=0.2,
                 lr_min=1e-6, lr_max=5e-3,
                 factor=2, verbose=0,
                 **kwargs):
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

    def _on_training_start(self) -> bool:
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