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
from src.env.callbacks import RenderCallback, AdaptiveLRCallback
from src.utils.get_next_filename import get_next_filename
from src.utils.update_checkpoints_tree import update_checkpoints_tree


def ppo_train(base_model_name=None, config_inheritance=True, demo=False, note_skip=False):

    if note_skip:
        note = ""
    else:
        note = input("\nPlease enter the notes for the current training model:\n > ")
    tensorboard_thread = ThreadTensorBoard()
    tensorboard_thread.run()

    model_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_env = make_env("train")
    if base_model_name and config_inheritance:
        update_CONFIG(CONFIG["path"]["config_backup"] + base_model_name + ".yaml")
    optional_kwargs = get_CONFIG(field="algorithm",
                                 try_keys=["n_steps", "batch_size", "n_epochs",
                                           "learning_rate", "clip_range", "gamma", 
                                           "gae_lambda", "device", "verbose", "vf_coef",
                                           ])
    if base_model_name:
        model = PPO.load(CONFIG["path"]["checkpoints"] + base_model_name + ".zip",
                         env=VecNormalize.load(
                             CONFIG["path"]["checkpoints"] + base_model_name + ".pkl",
                             train_env),
                         **optional_kwargs)
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        model = PPO(policy=CONFIG["algorithm"]["policy"],
                    env=train_env,
                    tensorboard_log=CONFIG["path"]["tensorboard"],
                    **optional_kwargs)
    display_model(train_env)
    
    if demo:
        demo_env = make_env("demo", demo_type="multiple")
        model.learn(total_timesteps=CONFIG["algorithm"]["total_timesteps"],
                    tb_log_name=model_name,
                    callback=[RenderCallback(demo_env),
                              AdaptiveLRCallback(model)],
                    )
        demo_env.close()
    else:
        model.learn(total_timesteps=CONFIG["algorithm"]["total_timesteps"],
                    tb_log_name=model_name,
                    callback=[AdaptiveLRCallback(model)],
                    )
    
    model.save(CONFIG["path"]["checkpoints"] + model_name + ".zip")
    train_env.save(CONFIG["path"]["checkpoints"] + model_name + ".pkl")
    update_checkpoints_tree(child=model_name, parent=base_model_name, note=note)
    shutil.copy2(CONFIG["path"]["env_class_py"], CONFIG["path"]["env_backup"] + model_name + ".py")
    save_CONFIG(CONFIG["path"]["config_backup"] + model_name + ".yaml")
    
    tensorboard_thread.stop()
    train_env.close()
    plt.close('all')

    print("\nModel %s traning accomplished!\n" % model_name)
    return model_name


def ppo_test(model_name=None, n_tests=3, max_steps=1000):

    if model_name:
        env = make_env("demo", demo_type="single")
        model = PPO.load(CONFIG["path"]["checkpoints"] + model_name,
                         env=env,
                         device=CONFIG["algorithm"]["device"]
                         )
        obs, info = env.reset()

        world_dt = env.env.env.env.dt * env.env.env.env.frame_skip
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
                    break

                plt_fig = plt.gcf()
                buffer = io.BytesIO()
                plt_fig.canvas.print_png(buffer)
                buffer.write(buffer.getvalue())
                plt_img = Image.open(buffer)
                plt_frames.append(plt_img)
                
                mjc_fig = env.env.env.env.render("rgb_array")
                mjc_frames.append(mjc_fig)

            env.env.env.env.plt_endline()
            obs, info = env.reset()
            imageio.mimsave(filepath + "plt_%d.gif" % target_index, plt_frames, fps=1/world_dt,
                            loop=0, subrectangles=True, palettesize=4, optimize=True)
            # mjc_frames = [
            #     Image.fromarray(frame).resize(
            #         (frame.shape[1] // 2, frame.shape[0] // 2),
            #         Image.Resampling.BOX) for frame in mjc_frames
            # ]
            mjc_frames = [np.array(frame)[::2, ::2] for frame in mjc_frames]
            imageio.mimsave(filepath + "mjc_%d.gif" % target_index, mjc_frames, fps=1/world_dt,
                            loop=0, subrectangles=True, palettesize=8, optimize=True)
            target_index += 1
                    
        env.close()
        plt.close('all')

    print("\nModel %s test accomplished!\n" % model_name)
    return model_name
