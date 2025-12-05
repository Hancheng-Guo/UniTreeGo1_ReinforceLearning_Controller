import imageio
import os
import io
import matplotlib
import shutil
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from datetime import datetime
from PIL import Image

from src.config.config import CONFIG, update_CONFIG, save_CONFIG, get_CONFIG
from src.render.render_tensorboard import ThreadTensorBoard
from src.env.make_env import make_env
from src.env.display_model import display_model
from src.env.callbacks import AdaptiveLRCallback
from src.utils.get_next_filename import get_next_filename
from src.utils.update_checkpoints_tree import update_checkpoints_tree


def ppo_train(base_model_name=None, config_inheritance=True, note_skip=False):

    if note_skip:
        note = ""
    else:
        note = input("\nPlease enter the notes for the current training model:\n > ")
    tensorboard_thread = ThreadTensorBoard()
    tensorboard_thread.run()

    train_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_env = make_env("train")
    if base_model_name and config_inheritance:
        update_CONFIG(CONFIG["path"]["config_backup"] + base_model_name + ".yaml")
    optional_kwargs = get_CONFIG(field="algorithm",
                                 try_keys=["n_steps", "batch_size", "n_epochs",
                                           "clip_range", "gamma", "gae_lambda",
                                           "device", "verbose", "vf_coef",
                                           ])
    if base_model_name:
        base_checkpoint = CONFIG["path"]["checkpoints"] + base_model_name
        train_env = VecNormalize.load(base_checkpoint + ".pkl", train_env)
        model = PPO.load(base_checkpoint + ".zip", env=train_env, **optional_kwargs)
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        model = PPO(policy=CONFIG["algorithm"]["policy"],
                    env=train_env,
                    tensorboard_log=CONFIG["path"]["tensorboard"],
                    learning_rate=CONFIG["algorithm"]["learning_rate_init"],
                    **optional_kwargs)
    display_model(train_env)
    
    model.learn(total_timesteps=CONFIG["algorithm"]["total_timesteps"],
                tb_log_name=train_time,
                callback=[AdaptiveLRCallback(),
                          ],
                )
    
    model.save(CONFIG["path"]["checkpoints"] + train_time + ".zip")
    train_env.save(CONFIG["path"]["checkpoints"] + train_time + ".pkl")
    update_checkpoints_tree(child=train_time, parent=base_model_name, note=note)
    shutil.copy2(CONFIG["path"]["env_class_py"], CONFIG["path"]["env_backup"] + train_time + ".py")
    save_CONFIG(CONFIG["path"]["config_backup"] + train_time + ".yaml")
    
    tensorboard_thread.stop()
    train_env.close()
    plt.close('all')

    print("\nModel %s traning accomplished!\n" % train_time)
    return train_time


def ppo_test(model_name=None, n_tests=3, max_steps=1000):

    if model_name:
        env = make_env("demo", demo_type="single")
        env = VecNormalize.load(CONFIG["path"]["checkpoints"] + model_name + ".pkl", env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(CONFIG["path"]["checkpoints"] + model_name,
                         env=env,
                         device=CONFIG["algorithm"]["device"]
                         )
        obs = env.reset()

        # world_dt = env.env.env.env.dt * env.env.env.env.frame_skip
        my_env = env.venv.envs[0].env.env.env.env
        world_dt = my_env.dt * my_env.frame_skip
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
                obs, reward, done, info = env.step(action)
                # obs, reward, terminated, truncated = env.step(action)
                if done:
                    break

                plt_fig = plt.gcf()
                buffer = io.BytesIO()
                plt_fig.canvas.print_png(buffer)
                buffer.write(buffer.getvalue())
                plt_img = Image.open(buffer)
                plt_frames.append(plt_img)
                
                mjc_fig = my_env.render("rgb_array")
                mjc_frames.append(mjc_fig)

            my_env.plt_endline()
            obs = env.reset()
            imageio.mimsave(filepath + "plt_%d.gif" % target_index, plt_frames, fps=1/world_dt,
                            loop=0, subrectangles=True, palettesize=4, optimize=True)
            mjc_frames = [np.array(frame)[::2, ::2] for frame in mjc_frames]
            imageio.mimsave(filepath + "mjc_%d.gif" % target_index, mjc_frames, fps=1/world_dt,
                            loop=0, subrectangles=True, palettesize=8, optimize=True)
            target_index += 1
                    
        env.close()
        plt.close('all')

    print("\nModel %s test accomplished!\n" % model_name)
    return model_name
