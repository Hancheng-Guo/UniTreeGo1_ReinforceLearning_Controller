import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register

from src.renders.mujoco import modify_model_camera


register(
    id="MyUniTreeGo1",
    entry_point="src.env.unitree_go1:UniTreeGo1Env",
)

def make_gym_env(config, env_mode="train", *args, **kwargs):

    kwargs["id"] = "MyUniTreeGo1"
    kwargs["main_body"] = "trunk"
    kwargs["include_cfrc_ext_in_observation"] = True
    kwargs["exclude_current_positions_from_observation"] = False

    kwargs["xml_file"] = config["path"]["model_dir_modified"] + "scene.xml"
    kwargs["frame_skip"] = config["train"]["frame_skip"]
    kwargs["reset_state"] = config["train"]["reset_state"]
    kwargs["reset_noise_scale"] = config["train"]["reset_noise_scale"]
    kwargs["max_episode_steps"] = config["train"]["max_episode_steps"]

    kwargs["plt_n_lines"] = config["demo"]["plt_n_lines"]
    kwargs["plt_x_range"] = config["demo"]["plt_x_range"]
    if not config["is"]["model_camera_modified"]:
        modify_model_camera(dir_original=config["path"]["model_dir_original"],
                            dir_modified=config["path"]["model_dir_modified"],
                            camera_pos=config["demo"]["camera_pos"],
                            camera_xyaxes=config["demo"]["camera_xyaxes"])

    kwargs["reward_config"] = config["reward"]

    return gym.make(*args, **kwargs)


def make_train_env(config, *args, **kwargs):
    return make_gym_env(config, *args, **kwargs)


def make_demo_env(config, *args, **kwargs):
    env_mode = "train" if config["is"]["param_shared"] else "demo"
    return make_gym_env(config, env_mode=env_mode,
                        render_mode=config["demo"]["demo_type"],
                        width=config["demo"]["mjc_render_width"],
                        height=config["demo"]["mjc_render_height"],
                        *args, **kwargs)

    
def make_env(mode, config={}, *args, **kwargs):

    if not config:
        print("config is empty!")
        return
    
    if mode == "train":
        return make_vec_env(lambda: make_train_env(config),
                            n_envs=config["train"]["n_envs"],
                            *args, **kwargs)
    elif mode == "demo":
        return make_vec_env(lambda: make_demo_env(config),
                            n_envs=1,
                            *args, **kwargs)
    else:
        print("make env error!")
        return
