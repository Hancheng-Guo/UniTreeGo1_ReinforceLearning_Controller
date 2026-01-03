from stable_baselines3.common.callbacks import BaseCallback
from src.callback.common.test_base_callback import TestBaseCallback


class ProgressBar():
    def __init__(self,
                 total: int,
                 custom_str: str = "",
                 highlight_len: int = 3,
                 bar_len :int = 40,
                 call_times_total: int = None,
                 ):
        self.i = 0
        self.total = total
        self.custom_str = f" {custom_str}" if custom_str else ""
        self.hl_len = highlight_len
        self.bar_len = bar_len
        self.dig_len = len(f"{self.total}")
        self.loop = "\u2591" * (self.bar_len + self.hl_len) + " " * self.hl_len
        self.call_times = 0
        if call_times_total is not None:
            self.call_times_total = call_times_total
            self.call_times_len = len(f"{call_times_total}")
        else:
            self.call_times_total = None
            self.call_times_len = 0

    def reset(self):
        self.i = 0
        self.call_times += 1

    def update(self, done):
        self.i += 1
        end_str = "<OUT OF RANGE!>\r" if done > self.total else "\r"
        done = self.total if done > self.total else done
            
        frac = done / self.total
        done_len = int(frac * self.bar_len)
        loop_i = self.i % (self.bar_len + 2 * self.hl_len)
        
        if self.call_times_total is not None:
            call_times_str = f"[{self.call_times:>{self.call_times_len}d}/{self.call_times_total}] "
        else:
            call_times_str = ""

        # Sample:
        #  ┌──────────────────────────────────────────────────────────────────────┐
        #  | > Rollout [  1/100] 99 % ██████████░░░   ░░░░   99/1000 steps <OUT OF RANGE!>|
        #  |└   P1    ┘└   P2   ┘└P3 ┘└   P4   ┘└   P5   ┘└      P6       ┘└   end_str   ┘|
        #  └──────────────────────────────────────────────────────────────────────┘
        print((f" >{self.custom_str} "
               f"{call_times_str}"
               f"{(frac * 100):^3.0f}% "
               f"{'\u2588' * done_len}"
               f"{(self.loop[-loop_i:] + self.loop[:-loop_i])[done_len:-(2 * self.hl_len)]} "
               f"{f'{done:>{self.dig_len}d}'}/{f'{self.total:>{self.dig_len}d}'} steps "),
               end=end_str)
        return True

    def clear(self):
        print("\033[2K\r", end="")
    

class TrainProgressCallback(BaseCallback):
    def __init__(self,
                 verbose: int = 0,
                 **kwargs):
        super().__init__(verbose)
        self.bar = None
        self.current_step = None

    def _on_training_start(self, **kwargs) -> bool:
        rollout_steps = self.model.n_steps * self.model.n_envs
        rollout_times_total = (self.model._total_timesteps + rollout_steps - 1) // rollout_steps
        self.bar = ProgressBar(total=self.model.n_steps * self.model.n_envs,
                               custom_str="Rollout",
                               call_times_total=rollout_times_total)
        return True

    def _on_rollout_start(self, **kwargs) -> bool:
        self.bar.reset()
        self.current_step = 0
        return True

    def _on_step(self, **kwargs) -> bool:
        self.current_step += 1
        self.bar.update(self.current_step * self.model.n_envs)
        return True
    
    def _on_rollout_end(self, **kwargs) -> bool:
        self.bar.clear()
        return True


class TestProgressCallback(TestBaseCallback):
    def __init__(self,
                 n_tests: int,
                 max_steps: int):
        self.bar = ProgressBar(total=max_steps,
                               custom_str="Darwing",
                               call_times_total=n_tests)
        self.i = None
        
    def _on_test_start(self, **kwargs) -> bool:
        self.bar.reset()
        self.i = 0
        return True
    
    def _on_test_step(self, **kwargs) -> bool:
        self.i += 1
        self.bar.update(self.i)
        return True
    
    def _on_test_end(self, **kwargs) -> bool:
        self.bar.clear()
        return True
