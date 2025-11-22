import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

import src.env.init_env
from config import CONFIG


def make_train_env(*args, **kwargs):
    return gym.make(
        id="MyUniTreeGo1",
        xml_file=CONFIG["path"]["model_xml"],
        forward_reward_weight=CONFIG["train"]["forward_reward_weight"],
        ctrl_cost_weight=CONFIG["train"]["ctrl_cost_weight"],
        contact_cost_weight=CONFIG["train"]["contact_cost_weight"],
        healthy_reward=CONFIG["train"]["healthy_reward_weight"],
        main_body="trunk",
        healthy_z_range=(CONFIG["train"]["healthy_z_min"], CONFIG["train"]["healthy_z_max"]),
        healthy_theta_range=(CONFIG["train"]["healthy_theta_min"], CONFIG["train"]["healthy_theta_max"]),
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=CONFIG["train"]["reset_noise_scale"],
        frame_skip=CONFIG["train"]["frame_skip"],
        max_episode_steps=CONFIG["train"]["max_episode_steps"],
        *args, **kwargs
    )

def make_demo_env(*args, **kwargs):
    param_type = "train" if CONFIG["is"]["param_shared"] else "demo"
    return gym.make(
        id="MyUniTreeGo1",
        xml_file=CONFIG["path"]["model_xml"],
        forward_reward_weight=CONFIG[param_type]["forward_reward_weight"],
        ctrl_cost_weight=CONFIG[param_type]["ctrl_cost_weight"],
        contact_cost_weight=CONFIG[param_type]["contact_cost_weight"],
        healthy_reward=CONFIG[param_type]["healthy_reward_weight"],
        main_body="trunk",
        healthy_z_range=(CONFIG[param_type]["healthy_z_min"], CONFIG[param_type]["healthy_z_max"]),
        healthy_theta_range=(CONFIG[param_type]["healthy_theta_min"], CONFIG[param_type]["healthy_theta_max"]),
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=CONFIG[param_type]["reset_noise_scale"],
        frame_skip=CONFIG[param_type]["frame_skip"],
        max_episode_steps=CONFIG[param_type]["max_episode_steps"],
        render_mode="human",
        *args, **kwargs
    )

def make_env(mode, *args, **kwargs):
    if mode == "train":
        return make_vec_env(make_train_env, n_envs=CONFIG["train"]["n_env"], *args, **kwargs)
    elif mode == "demo":
        return make_demo_env(*args, **kwargs)
    else:
        print("make env error!")
        return
