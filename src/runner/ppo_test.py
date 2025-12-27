import imageio
import os
import io
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from stable_baselines3.common.env_util import make_vec_env

from src.runner.common.make_gym_env import make_gym_env
from src.runner.common.check_base_name import check_base_name
from src.runner.common.load_model import load_model
from src.callback.base import ProgressBar
from src.config.base import CONFIG


class TestSaver():
    def __init__(self, test_env, test_name, test_dir):
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

    def _get_next_filename(self, prefix, ext): # e.g. prefix="img", ext="gif"
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


def make_demo_env(config, *args, **kwargs):
    return make_gym_env(config,
                        render_mode=config["demo"]["demo_type"],
                        width=config["demo"]["mjc_render_width"],
                        height=config["demo"]["mjc_render_height"],
                        *args, **kwargs)


def ppo_test(test_name=None, n_tests=3, max_steps=1000):

    test_name, test_dir = check_base_name(test_name, config=CONFIG)
    bar = ProgressBar(total=max_steps, custom_str="Darwing", call_times_total=n_tests)

    if test_name:
        test_env = make_vec_env(lambda: make_demo_env(CONFIG), n_envs=1)
        model, test_env, callback_kwargs = load_model(test_env, test_name, test_dir, CONFIG)
        test_env.envs[0].env.env.env.env.stage = np.load(callback_kwargs["base_stage"])
        test_env.training = False
        test_env.norm_reward = False
        obs = test_env.reset()

        test_saver = TestSaver(test_env, test_name, test_dir)
        for i in range(n_tests):
            bar.reset()
            for j in range(max_steps):
                bar.update(j + 1)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                if done:
                    break
                test_saver.append()
            bar.clear()
            print(" > Render Saving...", end="\r")
            test_saver.save()
            test_saver.reset()
            obs = test_env.reset()
 
        test_env.close()
        plt.close('all')
        print("\nModel %s test accomplished!\n" % test_name)
    return test_name
