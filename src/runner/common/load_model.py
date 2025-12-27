import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from src.runner.common.display_action import display_action
from src.runner.common.display_body import display_body
from src.runner.common.display_obs import display_obs


def load_model(env, base_name, base_dir, config, algorithm_kwargs={}, **kwargs):
    if base_name:
        base_model = os.path.join(base_dir, f"mdl_{base_name}.zip")
        base_env = os.path.join(base_dir, f"env_{base_name}.pkl")
        env = VecNormalize.load(base_env, env)
        algorithm_kwargs.pop("policy", None)
        algorithm_kwargs.pop("policy_kwargs", None)
        algorithm_kwargs.pop("learning_rate", None)
        model = PPO.load(base_model, env=env, **algorithm_kwargs, **kwargs)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        model = PPO(env=env, **algorithm_kwargs, **kwargs)
        
    if config["is"]["model_body_part_visiable"]:
        display_body(env)
    if config["is"]["model_obs_space_visiable"]:
        display_obs(env)
    if config["is"]["model_action_space_visiable"]:
        display_action(env)

    try:
        base_stage = os.path.join(base_dir, f"cst_{base_name}.npy")
    except:
        base_stage = None
    callback_kwargs = {
        # for StageScheduleCallback
        "base_stage": base_stage,
        # for CustomTensorboardCallback
        "log_freq": config["train"]["custom_log_freq"],
        # for CustomCheckpointCallback
        "save_freq": config["train"]["checkpoint_freq"],
        "env_py_path": config["path"]["env_py"],
        "checkpoint_tree_file_path": config["path"]["checkpoint_tree"],
        "checkpoints_path": config["path"]["output"],
        "base_name": base_name
    }

    return model, env, callback_kwargs