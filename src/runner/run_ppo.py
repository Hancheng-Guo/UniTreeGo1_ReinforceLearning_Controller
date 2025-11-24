import imageio
import os
import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime
from PIL import Image

from config import CONFIG
from src.render.render_tensorboard import init_tensorboard
from src.env.make_env import make_env
from src.env.display_model import display_model
from src.env.callbacks import RenderCallback
from src.common.get_next_filename import get_next_filename
from src.common.update_checkpoints_tree import update_checkpoints_tree

def ppo_train(base_model_name=None, demo=False):

    init_tensorboard()

    model_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_env = make_env("train")
    if base_model_name:
        model = PPO.load(CONFIG["path"]["checkpoints"] + base_model_name + ".zip",
                         env=train_env,)
    else:
        model = PPO(policy=CONFIG["algorithm"]["policy"],
                    env=train_env,
                    n_steps=CONFIG["algorithm"]["n_steps"],
                    batch_size=CONFIG["algorithm"]["batch_size"],
                    n_epochs=CONFIG["algorithm"]["n_epochs"],
                    gamma=CONFIG["algorithm"]["gamma"],
                    gae_lambda=CONFIG["algorithm"]["gae_lambda"],
                    device=CONFIG["algorithm"]["device"],
                    verbose=CONFIG["algorithm"]["verbose"],
                    tensorboard_log=CONFIG["path"]["tensorboard"],
                    )
    display_model(train_env)
    
    if demo:
        demo_env  = make_env("demo")
        model.learn(total_timesteps=CONFIG["algorithm"]["total_timesteps"],
                    tb_log_name=model_name,
                    callback=RenderCallback(demo_env),
                    )
        demo_env.close()
    else:
        model.learn(total_timesteps=CONFIG["algorithm"]["total_timesteps"],
                    tb_log_name=model_name,
                    )
    
    model.save(CONFIG["path"]["checkpoints"] + model_name + ".zip")
    update_checkpoints_tree(child=model_name, parent=base_model_name)

    train_env.close()
    plt.close('all')

    return model_name


def ppo_test(model_name=None, max_steps=1000):

    if model_name:
        env  = make_env("demo")
        model = PPO.load(CONFIG["path"]["checkpoints"] + model_name,
                         env=env,
                         )
        obs, info = env.reset()

        world_dt = env.env.env.env.dt *env.env.env.env.frame_skip
        plt_frames = []
        mjc_frames = []
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            plt_fig = plt.gcf()
            buffer = io.BytesIO()
            plt_fig.canvas.print_png(buffer)
            buffer.write(buffer.getvalue())
            plt_img = Image.open(buffer)
            plt_frames.append(plt_img)
            
            mjc_fig = env.env.env.env.render("rgb_array")
            mjc_frames.append(mjc_fig)

            if terminated or truncated:
                obs, info = env.reset()
                break

        filepath = CONFIG["path"]["demo"] + model_name.split('.')[0] + "/"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        _, plt_index = get_next_filename(filepath, "plt_", ".gif")
        _, mjc_index = get_next_filename(filepath, "mjc_", ".gif")
        
        imageio.mimsave(filepath + "plt_%d.gif" % (1 + max(plt_index, mjc_index)), plt_frames, fps=1/world_dt)
        imageio.mimsave(filepath + "mjc_%d.gif" % (1 + max(plt_index, mjc_index)), mjc_frames, fps=1/world_dt)
        env.close()
        plt.close('all')

        return model_name
