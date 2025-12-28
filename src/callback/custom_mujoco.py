import mujoco


class CustomMujocoCallback():
    def __init__(self, render_mode):
        self.render_mode = render_mode

        
    def _on_training_start(self, env, *args, **kwargs):
        self.env = env
        self.camera_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking")
        self.env.mjc_img = None

    def _on_episode_start(self, *args, **kwargs):
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            self.reset()


    def _on_step(self, *args, **kwargs):
        if self.render_mode in {"human", "rgb_array", "depth_array", "rgbd_tuple"}:
            self.env.mjc_img = self.env.render(self.render_mode)

    
    def reset(self):
        self._reset_tracking_camera()


    def _reset_tracking_camera(self):
        if self.render_mode == "human":
            self.env.mujoco_renderer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.env.mujoco_renderer.viewer.cam.fixedcamid = self.camera_id
        self.env.camera_id = self.camera_id
        self.env.mujoco_renderer.camera_id = self.camera_id
        