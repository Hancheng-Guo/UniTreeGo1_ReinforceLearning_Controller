import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.envs.registration import register

from src.utils.set_mujoco import modify_model_camera
from src.config.config import CONFIG


register(
    id="MyUniTreeGo1",
    entry_point="src.env.unitree_go1:UniTreeGo1Env",
)

def make_gym_env(env_mode="train", *args, **kwargs):

    if not CONFIG["is"]["model_camera_modified"]:
        modify_model_camera()

    kwargs["id"] = "MyUniTreeGo1"
    kwargs["main_body"] = "trunk"
    kwargs["include_cfrc_ext_in_observation"] = True
    kwargs["exclude_current_positions_from_observation"] = False

    kwargs["xml_file"] = CONFIG["path"]["model_dir_modified"] + "scene.xml"
    kwargs["ctrl_cost_weight"] = CONFIG["train"]["ctrl_cost_weight"]
    kwargs["contact_cost_weight"] = CONFIG["train"]["contact_cost_weight"]
    kwargs["forward_reward_weight"] = CONFIG["train"]["forward_reward_weight"]
    kwargs["healthy_reward_weight"] = CONFIG["train"]["healthy_reward_weight"]
    kwargs["state_reward_weight"] = CONFIG["train"]["state_reward_weight"]
    kwargs["posture_reward_weight"] = CONFIG["train"]["posture_reward_weight"]
    kwargs["frame_skip"] = CONFIG["train"]["frame_skip"]
    
    kwargs["healthy_z_range"] = (CONFIG[env_mode]["healthy_z_min"], CONFIG[env_mode]["healthy_z_max"])
    kwargs["healthy_pitch_range"] = (CONFIG[env_mode]["healthy_pitch_min"], CONFIG[env_mode]["healthy_pitch_max"])
    kwargs["reset_noise_scale"] = CONFIG[env_mode]["reset_noise_scale"]
    kwargs["max_episode_steps"] = CONFIG[env_mode]["max_episode_steps"]

    return gym.make(*args, **kwargs)


def make_train_env(*args, **kwargs):
    return make_gym_env(*args, **kwargs)


def make_demo_env(*args, **kwargs):
    env_mode = "train" if CONFIG["is"]["param_shared"] else "demo"
    return make_gym_env(env_mode,
                        render_mode=CONFIG["demo"]["demo_type"],
                        width=CONFIG["demo"]["mjc_render_width"],
                        height=CONFIG["demo"]["mjc_render_height"],
                        *args, **kwargs)

    
def make_env(mode, *args, **kwargs):
    if mode == "train":
        return make_vec_env(make_train_env, n_envs=CONFIG["train"]["n_envs"], *args, **kwargs)
    elif mode == "demo":
        return make_vec_env(lambda: make_demo_env(*args, **kwargs), n_envs=1)
    else:
        print("make env error!")
        return
