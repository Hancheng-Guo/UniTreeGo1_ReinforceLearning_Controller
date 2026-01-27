import mujoco
from src.callback.common.env_base_callback import EnvBaseCallback

class CustomMujocoCallback(EnvBaseCallback):
    def __init__(self, render_mode: str):
        self.render_mode = render_mode

        
    def _on_training_start(self, env, **kwargs) -> bool:
        self.env = env
        self.camera_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking")
        self.env.mjc_img = None
        return True

    def _on_episode_start(self, **kwargs) -> bool:
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            self.reset()
        return True


    def _on_step(self, **kwargs) -> bool:
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            self.env.mjc_img = self.env.render(self.render_mode)
        return True
    
    
    def reset(self):
        self._reset_tracking_camera()


    def _reset_tracking_camera(self):
        if self.render_mode == "human":
            self.env.mujoco_renderer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.env.mujoco_renderer.viewer.cam.fixedcamid = self.camera_id
        self.env.camera_id = self.camera_id
        self.env.mujoco_renderer.camera_id = self.camera_id
        