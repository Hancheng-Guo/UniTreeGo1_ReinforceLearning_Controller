import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register

from src.renders.mujoco import modify_model_camera


register(
    id="MyUniTreeGo1",
    entry_point="src.env.unitree_go1:UniTreeGo1Env",
)

def make_gym_env(config, env_mode="train", *args, **kwargs):

    if not config["is"]["model_camera_modified"]:
        modify_model_camera(dir_original=config["path"]["model_dir_original"],
                            dir_modified=config["path"]["model_dir_modified"],
                            camera_pos=config["demo"]["camera_pos"],
                            camera_xyaxes=config["demo"]["camera_xyaxes"])

    kwargs["id"] = "MyUniTreeGo1"
    kwargs["main_body"] = "trunk"
    kwargs["include_cfrc_ext_in_observation"] = True
    kwargs["exclude_current_positions_from_observation"] = False

    kwargs["xml_file"] = config["path"]["model_dir_modified"] + "scene.xml"
    kwargs["frame_skip"] = config["train"]["frame_skip"]
    kwargs["reset_state"] = config["train"]["reset_state"]
    kwargs["healthy_reward_weight"] = config["train"]["healthy_reward_weight"]
    kwargs["posture_reward_weight"] = config["train"]["posture_reward_weight"]
    kwargs["forward_reward_weight"] = config["train"]["forward_reward_weight"]
    kwargs["gait_reward_alpha"] = config["train"]["gait_reward_alpha"]
    kwargs["state_reward_weight"] = config["train"]["state_reward_weight"]
    kwargs["state_reward_alpha"] = config["train"]["state_reward_alpha"]
    kwargs["state_reward_hold"] = config["train"]["state_reward_hold"]
    kwargs["ctrl_cost_weight"] = config["train"]["ctrl_cost_weight"]
    kwargs["contact_cost_weight"] = config["train"]["contact_cost_weight"]

    kwargs["plt_n_lines"] = config["demo"]["plt_n_lines"]
    kwargs["plt_x_range"] = config["demo"]["plt_x_range"]
    
    kwargs["healthy_z_range"] = (config[env_mode]["healthy_z_min"],
                                 config[env_mode]["healthy_z_max"])
    kwargs["healthy_z_target"] = config[env_mode]["healthy_z_target"]
    kwargs["healthy_pitch_range"] = (config[env_mode]["healthy_pitch_min"],
                                     config[env_mode]["healthy_pitch_max"])
    kwargs["reset_noise_scale"] = config[env_mode]["reset_noise_scale"]
    kwargs["max_episode_steps"] = config[env_mode]["max_episode_steps"]

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
