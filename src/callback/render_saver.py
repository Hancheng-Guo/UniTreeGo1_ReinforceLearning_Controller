import imageio
import os
import io
import re
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
        self.target_index = max(plt_index, mjc_index) + 1

        self.plt_frames = []
        self.mjc_frames = []

    def _get_next_filename(self, prefix: str, ext: str): # e.g. prefix="img", ext="gif"
        pattern = re.compile(rf"{prefix}_{self.test_name}\[(\d+)\].{ext}$")
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
            self.plt_frames.append(plt_img)

            mjc_img = self.my_env.render("rgb_array")
            self.mjc_frames.append(mjc_img)
        else:
            self.mjc_frames.append(self.my_env.mjc_img)
            self.plt_frames.append(self.my_env.plt_img)

    def save(self):
        plt_path = os.path.join(self.save_dir,
                                f"plt_{self.test_name}[{self.target_index}].gif")
        imageio.mimsave(plt_path, self.plt_frames, fps=1/self.world_dt,
                        loop=0, subrectangles=True, palettesize=4, optimize=True)
        print(f"Saving matplot fig of demo to {plt_path}")

        mjc_path = os.path.join(self.save_dir,
                                f"mjc_{self.test_name}[{self.target_index}].gif")
        imageio.mimsave(mjc_path, self.mjc_frames, fps=1/self.world_dt,
                        loop=0, subrectangles=True, palettesize=8, optimize=True)
        print(f"Saving mujoco render of demo to {mjc_path}")

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