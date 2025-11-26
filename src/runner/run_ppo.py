import imageio
import os
import io
import matplotlib
import shutil
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime
from PIL import Image

from src.config.config import CONFIG, update_CONFIG, save_CONFIG
from src.render.render_tensorboard import init_tensorboard
from src.env.make_env import make_env
from src.env.display_model import display_model
from src.env.callbacks import RenderCallback
from src.utils.get_next_filename import get_next_filename
from src.utils.update_checkpoints_tree import update_checkpoints_tree

def ppo_train(base_model_name=None, demo=False):

    note = input("\nPlease enter the notes for the current training model:\n")
    init_tensorboard()

    model_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_env = make_env("train")
    if base_model_name:
        update_CONFIG(CONFIG["path"]["config_backup"] + base_model_name + ".yaml")
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
        demo_env  = make_env("demo", demo_type="multiple")
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
    update_checkpoints_tree(child=model_name, parent=base_model_name, note=note)
    shutil.copy2(CONFIG["path"]["env_class_py"], CONFIG["path"]["env_backup"] + model_name + ".py")
    save_CONFIG(CONFIG["path"]["config_backup"] + model_name + ".yaml")
    
    train_env.close()
    plt.close('all')

    print("\nModel %s traning accomplished!\n" % model_name)
    return model_name


def ppo_test(model_name=None, n_tests=3, max_steps=1000):

    if model_name:
        env = make_env("demo", demo_type="single")
        model = PPO.load(CONFIG["path"]["checkpoints"] + model_name,
                         env=env,
                         )
        obs, info = env.reset()

        world_dt = env.env.env.env.dt *env.env.env.env.frame_skip
        filepath = CONFIG["path"]["demo"] + model_name.split('.')[0] + "/"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        _, plt_index = get_next_filename(filepath, "plt_", ".gif")
        _, mjc_index = get_next_filename(filepath, "mjc_", ".gif")
        target_index = max(plt_index, mjc_index) + 1

        for i in range(n_tests):
            plt_frames = []
            mjc_frames = []

            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()
                    break

                plt_fig = plt.gcf()
                buffer = io.BytesIO()
                plt_fig.canvas.print_png(buffer)
                buffer.write(buffer.getvalue())
                plt_img = Image.open(buffer)
                plt_frames.append(plt_img)
                
                mjc_fig = env.env.env.env.render("rgb_array")
                mjc_frames.append(mjc_fig)

            imageio.mimsave(filepath + "plt_%d.gif" % target_index, plt_frames, fps=1/world_dt, loop=0)
            imageio.mimsave(filepath + "mjc_%d.gif" % target_index, mjc_frames, fps=1/world_dt, loop=0)
            target_index += 1
                    
        env.close()
        plt.close('all')

    print("\nModel %s test accomplished!\n" % model_name)
    return model_name
