import mujoco
import os
import shutil
import gymnasium as gym
import xml.etree.ElementTree as ET
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register

from src.config.config import CONFIG


register(
    id="MyUniTreeGo1",
    entry_point="src.env.unitree_go1:UniTreeGo1Env",
)

def modify_model_camera():

    dir_original = CONFIG["path"]["model_dir_original"]
    dir_modified = CONFIG["path"]["model_dir_modified"]
    os.makedirs(dir_modified, exist_ok=True)
    for root, dirs, files in os.walk(dir_original):
        rel_path = os.path.relpath(root, dir_original)
        dst_path = os.path.join(dir_modified, rel_path)
        os.makedirs(dst_path, exist_ok=True)
        for file in files:
            shutil.copy2(os.path.join(root, file), os.path.join(dst_path, file))

    xml_tree = ET.parse(dir_modified + "go1.xml")

    camera = None
    for cam in xml_tree.getroot().iter("camera"):
        if cam.attrib.get("name") == "tracking":
            camera = cam
            break
    if camera is None:
        raise ValueError(f"Camera 'tracking' not found in XML.")
    
    camera.set("pos", CONFIG["demo"]["camera_pos"])
    camera.set("xyaxes", CONFIG["demo"]["camera_xyaxes"])
    xml_tree.write(dir_modified + "go1.xml")


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
                        render_mode="human",
                        width=CONFIG["demo"]["mjc_render_width"],
                        height=CONFIG["demo"]["mjc_render_height"],
                        *args, **kwargs)

    
def make_env(mode, *args, **kwargs):
    if mode == "train":
        return make_vec_env(make_train_env, n_envs=CONFIG["train"]["n_envs"], *args, **kwargs)
    elif mode == "demo":
        return make_vec_env(make_demo_env, n_envs=1, *args, **kwargs)
    else:
        print("make env error!")
        return
