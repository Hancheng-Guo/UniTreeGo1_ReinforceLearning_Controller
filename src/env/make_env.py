import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from config import CONFIG


def make_train_env(*args, **kwargs):
    return gym.make(
        "Ant-v5",
        xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
        forward_reward_weight=1,
        ctrl_cost_weight=0.05,
        contact_cost_weight=5e-4,
        healthy_reward=1,
        main_body="trunk",
        healthy_z_range=(0.195, 0.75),
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=0.1,
        frame_skip=25,
        max_episode_steps=1000,
        *args, **kwargs
    )

def make_demo_env(*args, **kwargs):
    return gym.make(
        "Ant-v5",
        xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
        forward_reward_weight=1,
        ctrl_cost_weight=0.05,
        contact_cost_weight=5e-4,
        healthy_reward=1,
        main_body="trunk",
        healthy_z_range=(0.195, 0.75),
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=0.1,
        frame_skip=25,
        max_episode_steps=1000,
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
