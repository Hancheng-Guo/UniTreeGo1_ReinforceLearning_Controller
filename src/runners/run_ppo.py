import imageio
import os
import io
import numpy as np
import re
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from datetime import datetime
from PIL import Image

from src.utils.progress_bar import ProgressBar
from src.renders.tensorboard import ThreadTensorBoard
from src.env.make_env import make_env
from src.utils.display_model import display_obs, display_body, display_action
from src.callbacks.calbacks import CustomCheckpointCallback, AdaptiveLRCallback, ProgressBarCallback, CustomTensorboardCallback, StageScheduleCallback
from src.config.config import CONFIG, update_CONFIG, get_CONFIG


def check_base_name(base_name):
    if base_name:
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)$')
        match = pattern.match(base_name)
        if match:
            base_dir = os.path.join(CONFIG["path"]["output"], match.group(1))
            return base_name, base_dir
        else:
            pattern = re.compile(r'^(mdl|env)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d+)\.(zip|pkl)$')
            index = []
            for filename in os.listdir(os.path.join(CONFIG["path"]["output"], base_name)):
                match = pattern.match(filename)
                if match:
                    index.append(int(match.group(3)))
            index.sort(reverse=True)
            for i in range(len(index) - 1):
                if index[i] == index[i + 1]:
                    base_dir = os.path.join(CONFIG["path"]["output"], base_name)
                    base_name = f"{base_name}_{index[i]}"
                    return base_name, base_dir
    return None, None

def get_note(note_skip):
    note = "" if note_skip else input("\nPlease enter the notes for the current training model:\n > ")
    return note

def get_save_name():
    save_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(CONFIG["path"]["output"], save_name)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    return save_name, save_dir

def get_optional_kwargs(base_name, base_dir, config_inheritance):
    if base_name and config_inheritance:
        base_config = os.path.join(CONFIG["path"]["output"],
                                   base_dir,
                                   f"cfg_{base_name}.yaml")
        update_CONFIG(base_config)
    optional_kwargs = get_CONFIG(field="algorithm",
                                 try_keys=["n_steps", "batch_size", "n_epochs",
                                           "clip_range", "gamma", "gae_lambda",
                                           "device", "verbose", "vf_coef",])
    return optional_kwargs

def load_model(env, base_name, base_dir, config, **kwargs):
    if base_name:
        base_model = os.path.join(base_dir, f"mdl_{base_name}.zip")
        base_env = os.path.join(base_dir, f"env_{base_name}.pkl")
        env = VecNormalize.load(base_env, env)
        model = PPO.load(base_model, env=env, **kwargs)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        model = PPO(policy=CONFIG["algorithm"]["policy"],
                    learning_rate=CONFIG["algorithm"]["learning_rate_init"],
                    env=env,
                    **kwargs)
        
    if CONFIG["is"]["model_body_part_visiable"]:
        display_body(env)
    if CONFIG["is"]["model_obs_space_visiable"]:
        display_obs(env)
    if CONFIG["is"]["model_action_space_visiable"]:
        display_action(env)

    try:
        base_stage = os.path.join(base_dir, f"cst_{base_name}.npy")
    except:
        base_stage = None
    callback_kwargs = {
        # for StageScheduleCallback
        "base_stage": base_stage,
        # for CustomTensorboardCallback
        "log_freq": CONFIG["train"]["custom_log_freq"],
        # for CustomCheckpointCallback
        "save_freq": CONFIG["train"]["checkpoint_freq"],
        "env_py_path": CONFIG["path"]["env_py"],
        "checkpoint_tree_file_path": CONFIG["path"]["checkpoint_tree"],
        "checkpoints_path": CONFIG["path"]["output"],
        "base_name": base_name
    }

    return model, env, callback_kwargs

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
        self.my_env.plt_render.reset()
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


def ppo_train(base_name=None, config_inheritance=False, note_skip=False):

    note = get_note(note_skip)
    base_name, base_dir = check_base_name(base_name)
    save_name, save_dir = get_save_name()

    tensorboard_thread = ThreadTensorBoard(log_path=CONFIG["path"]["output"])
    tensorboard_thread.run()
    
    optional_kwargs = get_optional_kwargs(base_name, base_dir, config_inheritance)
    train_env = make_env("train", config=CONFIG)
    model, train_env, callback_kwargs = load_model(train_env, base_name, base_dir,
                                                   tensorboard_log=save_dir,
                                                   config=CONFIG,
                                                   **optional_kwargs)
    model.learn(total_timesteps=CONFIG["algorithm"]["total_timesteps"],
                tb_log_name=f"log_{save_name}",
                callback=[StageScheduleCallback(**callback_kwargs),
                          ProgressBarCallback(**callback_kwargs),
                          CustomTensorboardCallback(**callback_kwargs),
                          AdaptiveLRCallback(**callback_kwargs),
                          CustomCheckpointCallback(save_name, save_dir, note,
                                                   **callback_kwargs)
                         ],)
    
    tensorboard_thread.stop()
    train_env.close()
    plt.close('all')

    print("\nModel %s traning accomplished!\n" % save_name)
    return save_name


def ppo_test(test_name=None, n_tests=3, max_steps=1000, mode=""):

    test_name, test_dir = check_base_name(test_name)
    bar = ProgressBar(total=max_steps, custom_str="Darwing")

    if test_name:
        test_env = make_env("demo", config=CONFIG)
        model, test_env, callback_kwargs = load_model(test_env, test_name, test_dir,
                                                      config=CONFIG)
        test_env.envs[0].env.env.env.env.stage = np.load(callback_kwargs["base_stage"])
        test_env.training = False
        test_env.norm_reward = False
        obs = test_env.reset()

        test_saver = TestSaver(test_env, test_name, test_dir)
        for i in range(n_tests):
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
