import os
import re
import mujoco
import numpy as np
import gymnasium as gym
from torch import nn
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from src.callback.base import ThreadTensorBoard
import matplotlib.pyplot as plt

from src.runner.common.display_obs import get_qpos, get_qvel
from src.callback.base import Stage

from stable_baselines3.common.env_util import make_vec_env
from src.config.base import update_config, get_config
from src.runner.common.modify_model_camera import modify_model_camera

class PPORunner:
    def __init__(self,
                 config: dict,
                 base_name: str = None):
        self.config = config
        self.base_name = base_name
        self.check_base_name()

    def check_base_name(self):
        """Check and set the base directory and name format according to different file naming patterns.

        This function handles two main file naming formats:
        1. Direct datetime format: 'YYYY-MM-DD_HH-MM-SS_N' (e.g., 2022-01-01_12-30-45_1)
        2. File format: 'mdl' or 'env_YYYY-MM-DD_HH-MM-SS_N.(zip|pkl)' (e.g., mdl_2022-01-01_12-30-45_1.zip)

        :Updates:
        - :self.base_name: get the format 'YYYY-MM-DD_HH-MM-SS_N' with highest matching index N or None
        - :self.base_dir: get the directory path of the legal base name or None
        """
        if self.base_name:
            # Check if the base name matches the date_time_number format, e.g., 2022-01-01_12-30-45_1
            pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)$')
            match = pattern.match(self.base_name)
            if match:
                self.base_dir = os.path.join(self.config["path"]["output"], match.group(1))
                return
            else:
                # If the first pattern doesn't match, try matching the filename format like mdl or env_date_time_number.zip or pkl
                pattern = re.compile(r'^(mdl|env)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)\.(zip|pkl)$')
                index = []
                # Iterate through all files in the base name directory to find matching files
                for filename in os.listdir(os.path.join(self.config["path"]["output"], self.base_name)):
                    match = pattern.match(filename)
                    if match:
                        # Extract the numeric part from the filename and add to the index list
                        index.append(int(match.group(3)))
                # Sort the index in descending order
                index.sort(reverse=True)
                # Get the maximum value of duplicate indices (containing both .zip and .pkl)
                for i in range(len(index) - 1):
                    if index[i] == index[i + 1]:
                        self.base_dir = os.path.join(self.config["path"]["output"], self.base_name)
                        self.base_name = f"{self.base_name}_{index[i]}"
                        return
        self.base_name = None
        self.base_dir = None

    def make_gym_env(self,
                     env_name: str,
                     *args, **kwargs):
        """
        Creates and configures a gym environment with specific parameters for the reinforcement learning task.
        Sets up environment parameters including model files, camera settings, control configurations, and reward configurations.
        
        Args:
            env_name (str): The name of the environment to create
            *args: Variable length argument list passed to gym.make()
            **kwargs: Arbitrary keyword arguments passed to gym.make(), will be extended with additional configuration parameters
            
        Returns:
            gym.Env: A configured gym environment instance
        """

        # Set basic environment parameters
        kwargs["id"] = env_name
        kwargs["main_body"] = "trunk"
        kwargs["include_cfrc_ext_in_observation"] = True
        kwargs["exclude_current_positions_from_observation"] = False

        # Configure environment model and training parameters
        kwargs["xml_file"] = self.config["path"]["model_dir_modified"] + "scene.xml"
        kwargs["frame_skip"] = self.config["train"]["frame_skip"]
        kwargs["reset_noise_scale"] = self.config["train"]["reset_noise_scale"]
        kwargs["max_episode_steps"] = self.config["train"]["max_episode_steps"]

        # Configure demo visualization parameters
        kwargs["plt_n_cols"]  = self.config["demo"]["plt_n_cols"]
        kwargs["plt_n_lines"] = self.config["demo"]["plt_n_lines"]
        kwargs["plt_x_range"] = self.config["demo"]["plt_x_range"]
        
        # Modify model camera if needed based on configuration
        if not self.config["is"]["model_camera_modified"]:
            modify_model_camera(dir_original=self.config["path"]["model_dir_original"],
                                dir_modified=self.config["path"]["model_dir_modified"],
                                camera_pos=self.config["demo"]["camera_pos"],
                                camera_xyaxes=self.config["demo"]["camera_xyaxes"])

        # Add control and reward configurations to environment
        kwargs["control_config"] = self.config["control"]
        kwargs["reward_config"] = self.config["reward"]

        return gym.make(*args, **kwargs)
    
    def load_model(self,
                   env: gym.Env,
                   algorithm_kwargs: dict = {},
                   **kwargs):
        """Load an existing model or create a new one based on configuration.
        
        This method handles loading a pre-trained model from disk if a base name 
        is specified in the configuration. Otherwise, it creates a new model with
        default parameters. It also handles environment normalization appropriately.
        
        Args:
            env (gym.Env): The environment to be used with the model
            algorithm_kwargs (dict): Additional keyword arguments for the algorithm
            **kwargs: Additional keyword arguments to pass to PPO init function
    
        Returns:
            tuple: A tuple containing (model, env) where model is the loaded or 
               newly created PPO model and env is the normalized environment
        """
        if self.base_name:
            base_model = os.path.join(self.base_dir, f"mdl_{self.base_name}.zip")
            base_env = os.path.join(self.base_dir, f"env_{self.base_name}.pkl")
            env = VecNormalize.load(base_env, env)
            algorithm_kwargs.pop("policy", None)
            algorithm_kwargs.pop("policy_kwargs", None)
            algorithm_kwargs.pop("learning_rate", None)
            model = PPO.load(base_model, env=env, **algorithm_kwargs, **kwargs)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            model = PPO(env=env, **algorithm_kwargs, **kwargs)
        return model, env
    
    def register_callbacks(self, callbacks: list = []):
        self.callbacks = callbacks

    def _dispatch(self, event_name, *args, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, event_name, None)
            if fn is not None:
                fn(*args, **kwargs)


class PPOTrainer(PPORunner):
    def __init__(self,
                 config: dict,
                 base_name: str = None):
        super().__init__(config, base_name)

    def get_note(self, note_skip: bool=False):
        """
        Get notes for the current training model
        
        Args:
            note_skip (bool): Flag to skip note input. If True, return empty string;
                            If False, ask user to enter note text
        
        :Updates:
        - :self.note: The note text entered by user, or empty string if skipped
        """
        self.note = "" if note_skip else input("\nPlease enter the notes for the current training model:\n > ")

    def get_save_name(self):
        """Generates a unique save name and directory based on current timestamp.
        
        Sets self.save_name to current datetime in format 'YYYY-MM-DD_HH-MM-SS'
        and self.save_dir to the joined path of output directory and save name.
        Creates the save directory if it doesn't exist.

        :Updates:
        - :self.save_name: get save name with current time formated 'YYYY-MM-DD_HH-MM-SS'
        - :self.save_dir: create and get the directory path of the save name
        """
        self.save_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(self.config["path"]["output"], self.save_name)
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def maybe_run_tensorboard(self, tensorboard_skip: bool=False):
        if not tensorboard_skip:
            self.tensorboard_thread = ThreadTensorBoard(self.save_dir)
            self.tensorboard_thread.run()
        else:
            self.tensorboard_thread = None

    def get_algorithm_kwargs(self, config_inheritance: bool=False):
        """Extract and prepare algorithm hyperparameters from configuration for RL algorithm initialization.
        
        This function handles loading base configuration if available and extracts 
        algorithm-related parameters like steps, batch size, learning rate, etc.
        It also processes the activation function in the policy network by 
        converting string representations to actual neural network activation functions.

        Args:
            config_inheritance (bool): Flag indicating whether to inherit base configuration
        
        :Updates:
        - :self.algorithm_kwargs: get algorithm hyperparameters including n_steps, batch_size, learning_rate, etc.
        """
        # If base configuration name is provided and configuration inheritance is needed, load the base configuration file
        if self.base_name and config_inheritance:
            base_config = os.path.join(self.config["path"]["output"], self.base_dir, f"cfg_{self.base_name}.yaml")
            update_config(self.config, base_config)
        # Extract algorithm-related parameters from the configuration
        self.algorithm_kwargs = get_config(config=self.config, field="algorithm",
                                           try_keys=["n_steps", "batch_size", "n_epochs", "clip_range", "gamma",
                                                     "gae_lambda", "device", "verbose", "vf_coef",
                                                     "learning_rate", "policy", "policy_kwargs"])
        # Process activation function in the policy network, converting string representation to actual neural network activation function
        activation_fn = self.algorithm_kwargs["policy_kwargs"].get("activation_fn", "")
        self.algorithm_kwargs["policy_kwargs"]["activation_fn"] = nn.ELU if activation_fn == "ELU" else nn.Tanh

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
            base_stage = Stage.idle
        
        # Construct the callback keyword arguments dictionary
        self.callback_kwargs = {
            # for StageScheduleCallback
            "base_stage": base_stage,
            # for CustomTensorboardCallback
            "log_freq": self.config["train"]["custom_log_freq"],
            # for AdaptiveLRCallback
            "init_lr": self.config["algorithm"]["learning_rate"],
            # for CustomCheckpointCallback
            "note": self.note,
            "config": self.config,
            "save_name": self.save_name,
            "save_dir": self.save_dir,
            "save_freq": self.config["train"]["checkpoint_freq"],
            "env_py_path": self.config["path"]["env_py"],
            "checkpoint_tree_file_path": self.config["path"]["checkpoint_tree"],
            "checkpoints_path": self.config["path"]["output"],
            "base_name": self.base_name
        }

    def get_training_kwargs(self):
        """Prepare and set the training keyword arguments.
        
        This method creates a dictionary of parameters required for training the 
        PPO model, including the total number of timesteps and the tensorboard 
        log name based on the save name configuration.
        
        :Updates:
        - :self.training_kwargs: Sets the training keyword arguments dictionary
        """
        self.training_kwargs = {
            "total_timesteps": self.config["train"]["total_timesteps"],
            "tb_log_name": f"log_{self.save_name}",
        }


    def make_train_env(self, env_name: str):
        """Create and configure the training environment.

        This method creates a vectorized training environment using the specified 
        environment name. It wraps the gym environment with necessary wrappers
        and creates multiple parallel environments based on the configuration.

        Args:
            env_name (str): Name of the environment to create

        :Updates:
        - :self.train_env: Sets the training environment to a vectorized 
                           environment with n_envs parallel environments
        """
        self.train_env = make_vec_env(lambda: self.make_gym_env(env_name),
                                      n_envs=self.config["train"]["n_envs"])
        
    def load_model_with_train_env(self,
                                  tensorboard_skip: bool = False,
                                  **kwargs):
        """Load a model with the training environment and display optional visualizations.
        
        This method loads a model using the training environment and applies various 
        visualizations based on configuration settings. It can display information 
        about the robot body parts, observation space, and action space if enabled 
        in the configuration.
        It also handles tensorboard logging based on the tensorboard_skip parameter.
        
        Args:
            tensorboard_skip (bool): Whether to skip tensorboard logging, defaults to False
            **kwargs: Additional keyword arguments to pass to the parent load_model method
    
        :Updates:
        - :self.model: Sets the model to the loaded PPO model
        - :self.train_env: Updates the training environment with loaded normalized environment
        """
        kwargs.update(tensorboard_log=None if tensorboard_skip else self.save_dir)
        self.model, self.train_env = super().load_model(self.train_env, self.algorithm_kwargs, **kwargs)
        self.display_train_env()
        

    def display_train_env(self):
        """Display information about the training environment.
        
        This method displays information about the training environment, including 
        the body parts, observation space, and action space.
        """
        gym_env = self.train_env.venv.envs[0].env
        model = gym_env.unwrapped.model

        if self.config["is"]["model_body_part_visiable"]:
            n_fill = max([len(model.body(i).name) for i in range(model.nbody)]) + 1
            print("\n=== BODY ===")
            for ibody in range(model.nbody):
                print("[%3d] %s > pos: %s" % (ibody, model.body(ibody).name.rjust(n_fill),
                                              ["%+.4f" % x for x in model.body(ibody).pos]))

        if self.config["is"]["model_obs_space_visiable"]:
            print("\n=== OBSERVATION SPACE ===")
            iobs = 0
            n_fill = max([len(model.joint(i).name) for i in range(model.njnt)]) + 1
            obs_fill = len(f"{gym_env.observation_space.shape[0]}")
            for obs_name, obs_len in gym_env.unwrapped.observation_structure.items():
                if obs_name == "skipped_qpos":
                    continue
                print(f"[{iobs:>{obs_fill}}-{iobs + obs_len - 1:>{obs_fill}}] > {obs_name}")
                if obs_name == "qpos":
                    qpos_str = get_qpos(model, n_fill)[gym_env.unwrapped.observation_structure['skipped_qpos']:]
                    for i in range(len(qpos_str)):
                        print(f"    [{iobs:>{obs_fill}}] {qpos_str[i]}")
                        iobs += 1
                elif obs_name == "qvel":
                    qvel_str = get_qvel(model, n_fill)
                    for i in range(len(qvel_str)):
                        print(f"    [{iobs:>{obs_fill}}] {qvel_str[i]}")
                        iobs += 1
                else:
                    iobs += obs_len

        if self.config["is"]["model_action_space_visiable"]:
            print("\n=== ACTION SPACE ===")
            n_fill = len(f"{model.nu}")
            for i in range(model.nu):
                act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i][0])
                print(f"[{i:>{n_fill}}] > actuator: {act_name},\tjoint: {joint_name}")


    def train_env_close(self):
        """Close the training environment and clean up resources.
        
        This method properly closes the training environment, stops any running 
        tensorboard threads, closes all matplotlib plots, and prints a 
        completion message.
        """
        if self.tensorboard_thread:
            self.tensorboard_thread.stop()
        self.train_env.close()
        plt.close('all')

        print(f"\nModel {self.save_name} training accomplished!\n")

        self.base_name = self.save_name
        self.check_base_name()


class PPOTester(PPORunner):
    def __init__(self,
                 config: dict,
                 base_name: str = None):
        super().__init__(config, base_name)

    def make_test_env(self, env_name: str):
        """Create and configure the testing environment.

        This method creates a vectorized testing environment using the specified 
        environment name. It wraps the gym environment with necessary wrappers
        and creates multiple parallel environments based on the configuration.

        Args:
            env_name (str): Name of the environment to create

        :Updates:
        - :self.test_env: Sets the testing environment to a vectorized 
        """
        test_kwargs = {
            "render_mode": self.config["demo"]["demo_type"],
            "width": self.config["demo"]["mjc_render_width"],
            "height": self.config["demo"]["mjc_render_height"],
        }
        self.test_env = make_vec_env(lambda: self.make_gym_env(env_name, **test_kwargs), n_envs=1)

    def load_model_with_test_env(self, **kwargs):
        """Load a model with the test environment.
        
        This method loads a model using the test environment, which is typically 
        used for evaluating the trained policy in a separate environment instance.
        It also configures the test environment to disable training mode and 
        reward normalization.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the parent load_model method

        :Updates:
        - :self.model: Sets the model to the loaded PPO model
        - :self.test_env: Updates the test environment with loaded normalized environment
        """
        algorithm_kwargs = get_config(config=self.config, field="algorithm", try_keys=["device"])
        self.model, self.test_env = super().load_model(self.test_env, algorithm_kwargs, **kwargs)
        self.test_env.training = False
        self.test_env.norm_reward = False

    def test_env_close(self):
        """Close the test environment and clean up resources.
        
        This method properly closes the test environment, closes all
        matplotlib plots, and prints a completion message.
        """
        self.test_env.close()
        plt.close('all')
        print("\nModel %s test accomplished!\n" % self.base_name)
