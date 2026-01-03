class TestBaseCallback:
    def __init__(self, **kwargs):
        pass

    def _on_test_start(self, **kwargs) -> bool:
        return True
    
    def _on_test_step(self, **kwargs) -> bool:
        return True

    def _on_test_end(self, **kwargs) -> bool:
        return True