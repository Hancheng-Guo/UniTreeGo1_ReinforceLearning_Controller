class EnvBaseCallback:
    def __init__(self, **kwargs):
        pass

    def _on_training_start(self, **kwargs) -> bool:
        return True
    
    def _on_episode_start(self, **kwargs) -> bool:
        return True

    def _on_step(self, **kwargs) -> bool:
        return True
    
    def _on_episode_end(self, **kwargs) -> bool:
        return True
    
    def _on_training_end(self, **kwargs) -> bool:
        return True