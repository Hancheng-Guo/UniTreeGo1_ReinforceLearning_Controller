import imageio
import os
import io
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.callback.common.test_base_callback import TestBaseCallback



class RenderSaver():
    def __init__(self, test_env, test_name: str, test_dir: str):
        self.test_env = test_env
        self.test_name = test_name
        self.test_dir = test_dir
        self.my_env = test_env.venv.envs[0].env.env.env.env
        self.world_dt = self.my_env.dt * self.my_env.frame_skip

        self.save_dir = os.path.join(test_dir, f"demo_{self.test_name}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        _, plt_index = self._get_next_filename("plt", "gif")
        _, mjc_index = self._get_next_filename("mjc", "gif")
        _, mp4_index = self._get_next_filename("", "mp4")
        self.target_index = max(plt_index, mjc_index, mp4_index) + 1

        self.plt_frames = []
        self.mjc_frames = []

    def _get_next_filename(self, prefix: str, ext: str): # e.g. prefix="img", ext="gif"
        if prefix != "":
            prefix = f"{prefix}_"
        pattern = re.compile(rf"{prefix}{self.test_name}\[(\d+)\].{ext}$")
        max_num = 0
        for filename in os.listdir(self.save_dir): 
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
        return f"{prefix}{max_num + 1}{ext}", max_num
    
    def reset(self):
        self.my_env.reset()
        self.plt_frames = []
        self.mjc_frames = []

    def append(self):
        if self.test_env.render_mode == "human":
            plt_fig = plt.gcf()
            buffer = io.BytesIO()
            plt_fig.canvas.print_png(buffer)
            buffer.write(buffer.getvalue())
            plt_img = Image.open(buffer)
            self.plt_frames.append(plt_img.convert("RGB"))

            mjc_img = self.my_env.render("rgb_array")
            self.mjc_frames.append(Image.fromarray(mjc_img))
        elif self.test_env.render_mode in ["rgb_array", "depth_array", "rgbd_tuple",]:
            self.mjc_frames.append(Image.fromarray(self.my_env.mjc_img))
            self.plt_frames.append(self.my_env.plt_img.convert("RGB"))
            

    def save(self):

        if self.test_env.render_mode is not None:
            fps = 1/self.world_dt

            # plt_path = os.path.join(self.save_dir, f"plt_{self.test_name}[{self.target_index}].gif")
            # imageio.mimsave(plt_path, self.plt_frames, fps=fps, loop=0, subrectangles=True, optimize=True)
            # print(f"Saving matplot fig of demo to {plt_path}")

            # mjc_path = os.path.join(self.save_dir, f"mjc_{self.test_name}[{self.target_index}].gif")
            # imageio.mimsave(mjc_path, self.mjc_frames, fps=fps, loop=0, subrectangles=True, optimize=True)
            # print(f"Saving mujoco render of demo to {mjc_path}")

            com_frames = []
            mjc_new_width = int(self.mjc_frames[0].width / self.mjc_frames[0].height * self.plt_frames[0].height)
            mjc_new_height = self.plt_frames[0].height
            for i in range(max(len(self.mjc_frames),len(self.plt_frames))):
                mjc_frame = self.mjc_frames[i].resize((mjc_new_width, mjc_new_height), Image.LANCZOS)
                plt_frame = self.plt_frames[i]
                com_frame = np.hstack((np.array(mjc_frame), np.array(plt_frame)))
                h, w, _ = com_frame.shape
                com_frame = com_frame[:(h - h % 2), :(w - w % 2), :]
                com_frames.append(Image.fromarray(com_frame))
                
            com_path = os.path.join(self.save_dir, f"{self.test_name}[{self.target_index}].mp4")
            imageio.mimsave(com_path, com_frames, fps=fps, codec="libx264", quality=6, macro_block_size=None)
            print(f"Saving render of demo to {com_path}")

        self.target_index += 1


class RenderSaverCallback(TestBaseCallback):
    def __init__(self, ppo_tester, **kwargs):
        self.render_saver = RenderSaver(ppo_tester.test_env,
                                        ppo_tester.base_name,
                                        ppo_tester.base_dir)

    def _on_test_start(self, **kwargs) -> bool:
        self.render_saver.reset()
        return True
    
    def _on_test_step(self, **kwargs) -> bool:
        self.render_saver.append()
        return True

    def _on_test_end(self, **kwargs) -> bool:
        print(" > Render Saving...", end="\r")
        self.render_saver.save()
        return True