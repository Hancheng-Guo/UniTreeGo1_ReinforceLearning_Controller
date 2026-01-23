from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from typing import Any, Optional, TypeVar, Union

class IterBaseCallback(BaseCallback):
    def __init__(self, verbose: int = 0, **kwargs):
        super().__init__(verbose)

    def _on_iteration_start(self, **kwargs) -> bool:
        return True
    
    def _on_iteration_end(self, **kwargs) -> bool:
        return True
    
    def on_training_start(self, locals_: dict[str, Any], globals_: dict[str, Any]):
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        if hasattr(self, "model") and self.model is not None:
            self.num_timesteps = self.model.num_timesteps
        return self._on_training_start()
    

class IterCallBackList(IterBaseCallback, CallbackList):
    def __init__(self, callbacks):
        super().__init__(callbacks)

    def on_iteration_start(self, **kwargs) -> bool:
        ret = True
        for callback in self.callbacks:
            fn = getattr(callback, "_on_iteration_start", None)
            if fn is not None:
                ret = ret and fn(**kwargs)
        return ret

    def on_iteration_end(self, **kwargs) -> bool:
        ret = True
        for callback in self.callbacks:
            fn = getattr(callback, "_on_iteration_end", None)
            if fn is not None:
                ret = ret and fn(**kwargs)
        return ret

