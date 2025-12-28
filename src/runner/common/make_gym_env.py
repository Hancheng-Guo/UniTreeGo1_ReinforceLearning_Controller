import gymnasium as gym
from gymnasium.envs.registration import register

from src.runner.common.modify_model_camera import modify_model_camera


register(
    id="MyUniTreeGo1",
    entry_point="src.env.unitree_go1:UniTreeGo1Env",
)

def make_gym_env(config, *args, **kwargs):

    kwargs["id"] = "MyUniTreeGo1"
    kwargs["main_body"] = "trunk"
    kwargs["include_cfrc_ext_in_observation"] = True
    kwargs["exclude_current_positions_from_observation"] = False

    kwargs["xml_file"] = config["path"]["model_dir_modified"] + "scene.xml"
    kwargs["frame_skip"] = config["train"]["frame_skip"]
    kwargs["reset_noise_scale"] = config["train"]["reset_noise_scale"]
    kwargs["max_episode_steps"] = config["train"]["max_episode_steps"]

    kwargs["plt_n_cols"]  = config["demo"]["plt_n_cols"]
    kwargs["plt_n_lines"] = config["demo"]["plt_n_lines"]
    kwargs["plt_x_range"] = config["demo"]["plt_x_range"]
    if not config["is"]["model_camera_modified"]:
        modify_model_camera(dir_original=config["path"]["model_dir_original"],
                            dir_modified=config["path"]["model_dir_modified"],
                            camera_pos=config["demo"]["camera_pos"],
                            camera_xyaxes=config["demo"]["camera_xyaxes"])

    kwargs["control_config"] = config["control"]
    kwargs["reward_config"] = config["reward"]

    return gym.make(*args, **kwargs)



    
