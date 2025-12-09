import mujoco
import os
import shutil
import xml.etree.ElementTree as ET

from src.config.config import CONFIG

def set_tracking_camera(env):
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking")
    if env.render_mode == "human":
        env.mujoco_renderer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        env.mujoco_renderer.viewer.cam.fixedcamid = camera_id
    env.camera_id = camera_id
    env.mujoco_renderer.camera_id = camera_id

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
