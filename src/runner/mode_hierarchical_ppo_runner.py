import os
import numpy as np
import yaml
from torch import nn
import matplotlib.pyplot as plt
from src.callback.base import HierarchicalStageScheduleCallback
from src.callback.base import HierarchicalCheckpointCallback
from src.callback.base import AdaptiveLRCallback
from src.callback.base import TestProgressCallback, IterProgressCallback
from src.callback.base import CustomTensorboardCallback
from src.callback.base import RenderSaverCallback
from src.config.base import update_config, get_config
from stable_baselines3.common.env_util import make_vec_env
from src.runner.common.ppo_runner import PPOTrainer, PPOTester
from src.runner.common.logger_proxy import LoggerProxy
from src.callback.base import IterCallBackList
from stable_baselines3.common.vec_env import VecNormalize


class ModeHierarchicalPPORunner(PPOTrainer, PPOTester):
    def __init__(self,
                 base_name: str = None):
        with open("./src/config/hierarchical_ppo_config.yaml", "r") as f:
           config = yaml.safe_load(f)
        super().__init__(config, base_name)

    def train(self,
              config_inheritance: bool = False,
              note_skip: bool = False,
              tensorboard_skip : bool = False):
        
        self.get_note(note_skip)    # Get training note information
        self.get_save_name()        # Get save name and directory
        self.run_tensorboard(tensorboard_skip)      # Start TensorBoard thread for logging
        self.inherite_config(config_inheritance)    # Inherit config
        self.get_algorithm_kwargs()                 # Prepare algorithm parameters
        self.make_loco_train_env("ModeConditionedLocomotionEnv")    # Create parallel locomotion training environment
        self.make_mode_train_env("ModeSelectionEnv")        # Create parallel mode training environment
        self.load_model_with_train_env(tensorboard_skip)    # Load model and environment

        self.get_callback_kwargs()  # Get callback function parameters
        self.get_training_kwargs()  # Get training parameters
        # Start model training process
        self.learn(com_callback=[IterProgressCallback(**self.com_callback_kwargs),
                                 HierarchicalStageScheduleCallback(**self.com_callback_kwargs),
                                 HierarchicalCheckpointCallback(**self.com_callback_kwargs)],
                   loco_callback=[CustomTensorboardCallback(**self.loco_callback_kwargs),
                                  AdaptiveLRCallback(**self.loco_callback_kwargs)],
                   mode_callback=[CustomTensorboardCallback(**self.mode_callback_kwargs),
                                  AdaptiveLRCallback(**self.mode_callback_kwargs)])
        self.train_env_close()  # Clean up resources

    def learn(self,
              com_callback: list = [],
              loco_callback: list = [],
              mode_callback: list = [],):

        self.loco_total_timesteps, _ = self.loco_model._setup_learn(**self.loco_training_kwargs)
        self.mode_total_timesteps, _ = self.mode_model._setup_learn(**self.mode_training_kwargs)

        assert self.loco_model.env is not None and self.mode_model.env is not None

        iteration = 0
        loco_rollout_count = 0
        mode_rollout_count = 0
        self.loco_model._logger = LoggerProxy(self.loco_model._logger,
                                              {"train": "loco_train", "rollout": "loco_rollout"})
        self.mode_model._logger = LoggerProxy(self.mode_model._logger,
                                              {"train": "mode_train", "rollout": "mode_rollout"})
        
        self.loco_callback = IterCallBackList([*loco_callback, *com_callback])
        self.mode_callback = IterCallBackList([*mode_callback, *com_callback])
        self.loco_callback.init_callback(self.loco_model)
        self.mode_callback.init_callback(self.mode_model)

        self.loco_callback.on_training_start(locals(), globals())
        self.mode_callback.on_training_start(locals(), globals())
        while iteration < self.total_iterations:
            iteration += 1

            train_loco = self.loco_callback.on_iteration_start(model=self.loco_model)
            while train_loco:
                continue_training = self.loco_model.collect_rollouts(env=self.loco_model.env,
                                                                     callback=self.loco_callback,
                                                                     rollout_buffer=self.loco_model.rollout_buffer,
                                                                     n_rollout_steps=self.loco_model.n_steps)
                loco_rollout_count += 1
                if not continue_training:
                    iteration = self.total_iterations
                    break

                self.loco_model._update_current_progress_remaining(self.loco_model.num_timesteps, self.loco_total_timesteps)

                assert self.loco_model.ep_info_buffer is not None
                self.loco_model.dump_logs(loco_rollout_count)
                self.loco_model.train()

                if self.loco_model.num_timesteps % self.loco_iteration_steps == 0:
                    break
            self.loco_callback.on_iteration_end()

            train_mode = self.mode_callback.on_iteration_start(model=self.mode_model)
            while train_mode:
                continue_training = self.mode_model.collect_rollouts(env=self.mode_model.env,
                                                                     callback=self.mode_callback,
                                                                     rollout_buffer=self.mode_model.rollout_buffer,
                                                                     n_rollout_steps=self.mode_model.n_steps)
                mode_rollout_count += 1
                if not continue_training:
                    iteration = self.total_iterations
                    break

                self.mode_model._update_current_progress_remaining(self.mode_model.num_timesteps, self.mode_total_timesteps)

                assert self.mode_model.ep_info_buffer is not None
                self.mode_model.dump_logs(mode_rollout_count)
                self.mode_model.train()

                if self.mode_model.num_timesteps % self.mode_iteration_steps == 0:
                    break
            self.mode_callback.on_iteration_end()

        self.loco_model._logger = self.loco_model._logger._logger
        self.mode_model._logger = self.mode_model._logger._logger
        self.loco_callback.on_training_end()
        self.mode_callback.on_training_end()


    def get_algorithm_kwargs(self):
        """Extract and prepare algorithm hyperparameters from configuration for RL algorithm initialization.
        
        This function extracts algorithm-related parameters like steps, batch size, etc.
        It also processes the activation function in the policy network by 
        converting string representations to actual neural network activation functions.

        :Updates:
        - :self.algorithm_kwargs: get algorithm hyperparameters including n_steps, batch_size, learning_rate, etc.
        - :self.loco_algorithm_kwargs: algorithm kwargs for locomotion policy including activation function mapping
        - :self.mode_algorithm_kwargs: algorithm kwargs for mode policy including activation function mapping
        """
        # Extract algorithm-related parameters from the configuration
        self.algorithm_kwargs = get_config(config=self.config, field="algorithm", try_keys=["device", "verbose"])

        self.loco_algorithm_kwargs = get_config(config=self.config, field="algorithm", try_keys=["loco"])["loco"]
        activation_fn = self.loco_algorithm_kwargs["policy_kwargs"].get("activation_fn", "")
        self.loco_algorithm_kwargs["policy_kwargs"]["activation_fn"] = nn.ELU if activation_fn == "ELU" else nn.Tanh

        self.mode_algorithm_kwargs = get_config(config=self.config, field="algorithm", try_keys=["mode"])["mode"]
        activation_fn = self.mode_algorithm_kwargs["policy_kwargs"].get("activation_fn", "")
        self.mode_algorithm_kwargs["policy_kwargs"]["activation_fn"] = nn.ELU if activation_fn == "ELU" else nn.Tanh

    def get_training_kwargs(self):
        self.total_iterations = self.config["train"]["total_iterations"]
        self.mode_iteration_steps = self.config["train"]["mode_iteration_steps"]
        self.loco_iteration_steps = self.config["train"]["loco_iteration_steps"]
        self.mode_training_kwargs = {
            "total_timesteps": self.config["train"]["mode_iteration_steps"] * self.config["train"]["total_iterations"],
            "reset_num_timesteps": False,
            "tb_log_name": f"log_mode_{self.save_name}",
        }
        self.loco_training_kwargs = {
            "total_timesteps": self.config["train"]["loco_iteration_steps"] * self.config["train"]["total_iterations"],
            "reset_num_timesteps": False,
            "tb_log_name": f"log_loco_{self.save_name}",
        }

    def get_callback_kwargs(self):
        """
        Creates and returns a dictionary of keyword arguments for initializing various callbacks
        used during the PPO training process.

        :Updates:
        - :self.callback_kwargs: A dictionary containing keyword arguments for various training callbacks
        """
        try:
            # Attempt to load the base stage from .npy file
            base_stage_path = os.path.join(self.base_dir, f"cst_{self.base_name}.npy")
            base_stage = np.load(base_stage_path)
        except:
            # Default to idle stage if file doesn't exist or can't be loaded
            base_stage = 0.
        
        loco_rollout_steps = self.loco_model.n_steps * self.loco_model.n_envs
        loco_iteration_steps = self.config["train"]["loco_iteration_steps"]
        loco_iteration_rollouts = (loco_iteration_steps + loco_rollout_steps - 1) // loco_rollout_steps


        mode_rollout_steps = self.mode_model.n_steps * self.mode_model.n_envs
        mode_iteration_steps = self.config["train"]["mode_iteration_steps"]
        mode_iteration_rollouts = (mode_iteration_steps + mode_rollout_steps - 1) // mode_rollout_steps

        # n_iterations = (self.config["train"]["total_timesteps"] + iteration_steps - 1) // iteration_steps
        

        self.com_callback_kwargs = {
            "loco_model": self.loco_model,
            "mode_model": self.mode_model,
            # for ProgressBar
            "n_rollouts": self.config["train"]["total_iterations"] * (loco_iteration_rollouts + mode_iteration_rollouts),
            "loco_rollout_steps": loco_rollout_steps,
            "mode_rollout_steps": mode_rollout_steps,
            # for StageScheduleCallback
            "base_stage": base_stage,
            "loco_iteration_rollouts": loco_iteration_rollouts,
            "mode_iteration_rollouts": mode_iteration_rollouts,
            # for CustomCheckpointCallback
            "note": self.note,
            "config": self.config,
            "save_name": self.save_name,
            "save_dir": self.save_dir,
            "save_freq_iterations": self.config["train"]["checkpoint_freq_iterations"],
            "loco_env_py_path": self.config["path"]["loco_env_py"],
            "mode_env_py_path": self.config["path"]["mode_env_py"],
            "checkpoints_path": self.config["path"]["output"],
            "base_name": self.base_name
        }

        self.loco_callback_kwargs = {
            # for CustomTensorboardCallback
            "log_prefix": "loco_custom",
            "log_freq": self.config["train"]["custom_loco_log_freq"],
            "tensorboard_items": self.config["tensorboard_items"]["loco"],
            # for AdaptiveLRCallback
            "lr_prefix": "loco_train",
            "init_lr": self.config["algorithm"]["loco"]["learning_rate"],
        }

        self.mode_callback_kwargs = {
            # for CustomTensorboardCallback
            "log_prefix": "mode_custom",
            "log_freq": self.config["train"]["custom_mode_log_freq"],
            "tensorboard_items": self.config["tensorboard_items"]["mode"],
            # for AdaptiveLRCallback
            "lr_prefix": "mode_train",
            "init_lr": self.config["algorithm"]["mode"]["learning_rate"],
        }

      
    def make_loco_train_env(self, env_name):
        self.make_train_env(env_name)
        self.loco_train_env = self.train_env

    def make_mode_train_env(self, env_name):
        self.make_train_env(env_name)
        self.mode_train_env = self.train_env

    def load_model_with_train_env(self,
                                  tensorboard_skip: bool = False,
                                  **kwargs):
        kwargs.update(tensorboard_log=None if tensorboard_skip else self.save_dir)
        self.mode_model, self.mode_train_env = super().load_model(self.mode_train_env,
                                                                  {**self.algorithm_kwargs, **self.mode_algorithm_kwargs},
                                                                  **kwargs)
        self.loco_model, self.loco_train_env = super().load_model(self.loco_train_env,
                                                                  {**self.algorithm_kwargs, **self.loco_algorithm_kwargs},
                                                                  **kwargs)
        self.display_train_env(self.loco_train_env.venv.envs[0].env)
        
        for env in self.loco_train_env.venv.envs:
            env.env.env.env.env.mode_model = self.mode_model
        
        self.loco_env_for_mode = []
        for env in self.mode_train_env.venv.envs:
            loco_env = self.make_gym_env("ModeConditionedLocomotionEnv")
            loco_env.training = False
            loco_env.norm_reward = False
            self.loco_env_for_mode.append(loco_env)
            env.env.env.env.env.loco_env = loco_env
            env.env.env.env.env.loco_normalize_obs = self.loco_train_env.normalize_obs
            env.env.env.env.env.loco_discontruct_obs = loco_env.env.env.env._discontruct_obs
            env.env.env.env.env.loco_model = self.loco_model
        
            

    def train_env_close(self):
        if self.tensorboard_thread:
            self.tensorboard_thread.stop()
        for env in self.loco_env_for_mode:
            env.close()
        self.loco_train_env.close()
        self.mode_train_env.close()
        plt.close('all')

        print(f"\nModel {self.save_name} training accomplished!\n")

        self.base_name = self.save_name
        self.check_base_name()

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



