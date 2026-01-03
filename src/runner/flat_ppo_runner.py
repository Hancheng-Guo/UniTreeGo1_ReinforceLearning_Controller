import os
import numpy as np
import yaml

from src.callback.base import CustomCheckpointCallback
from src.callback.base import AdaptiveLRCallback
from src.callback.base import TrainProgressCallback, TestProgressCallback
from src.callback.base import CustomTensorboardCallback
from src.callback.base import StageScheduleCallback
from src.callback.base import RenderSaverCallback

from src.runner.common.ppo_runner import PPOTrainer, PPOTester

class FlatPPORunner(PPOTrainer, PPOTester):
    def __init__(self,
                 base_name: str = None):
        with open("./src/config/flat_ppo_config.yaml", "r") as f:
           config = yaml.safe_load(f)
        super().__init__(config, base_name)

    def train(self,
              config_inheritance: bool = False,
              note_skip: bool = False,
              tensorboard_skip : bool = False):
        
        self.get_note(note_skip)    # Get training note information
        self.get_save_name()        # Get save name and directory
        self.maybe_run_tensorboard(tensorboard_skip)    # Start TensorBoard thread for logging
        self.get_algorithm_kwargs(config_inheritance)   # Prepare algorithm parameters
        self.make_train_env("FlatLocomotionEnv")        # Create parallel training environment
        self.load_model_with_train_env(tensorboard_skip)    # Load model and environment

        self.get_callback_kwargs()  # Get callback function parameters
        self.get_training_kwargs()  # Get training parameters
        # Start model training process
        self.model.learn(**self.training_kwargs,
                         callback=[TrainProgressCallback(**self.callback_kwargs),
                                   StageScheduleCallback(**self.callback_kwargs),
                                   CustomTensorboardCallback(**self.callback_kwargs),
                                   AdaptiveLRCallback(**self.callback_kwargs),
                                   CustomCheckpointCallback(**self.callback_kwargs)])
        
        self.train_env_close()  # Clean up resources

    def test(self, n_tests=3, max_steps=1000):
        
        if self.base_name:
            self.make_test_env("FlatLocomotionEnv") # Create vectorized environment for testing
            self.load_model_with_test_env()         # Load pre-trained model and environment
            self.register_callbacks([TestProgressCallback(n_tests, max_steps),
                                     RenderSaverCallback(self)])

            for i in range(n_tests):
                self._dispatch("_on_test_start")
                obs = self.test_env.reset()
                for j in range(max_steps):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.test_env.step(action)
                    if done:
                        break
                    self._dispatch("_on_test_step")
                self._dispatch("_on_test_end")
                
            self.test_env.close()   # Clean up resources
    
    def load_model_with_test_env(self, **kwargs):
        super().load_model_with_test_env(**kwargs)
        self.test_env.envs[0].env.env.env.env.stage = np.load(os.path.join(self.base_dir,
                                                                           f"cst_{self.base_name}.npy"))



