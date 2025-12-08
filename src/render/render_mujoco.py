import mujoco

def set_tracking_camera(env):
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking")
    if env.render_mode == "human":
        env.mujoco_renderer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        env.mujoco_renderer.viewer.cam.fixedcamid = camera_id
    env.camera_id = camera_id
    env.mujoco_renderer.camera_id = camera_id