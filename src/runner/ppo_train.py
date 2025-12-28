import os
import matplotlib.pyplot as plt
from datetime import datetime
from torch import nn
from stable_baselines3.common.env_util import make_vec_env

from src.runner.common.make_gym_env import make_gym_env
from src.runner.common.check_base_name import check_base_name
from src.runner.common.load_model import load_model
from src.callback.base import CustomCheckpointCallback
from src.callback.base import AdaptiveLRCallback
from src.callback.base import ProgressBarCallback
from src.callback.base import CustomTensorboardCallback, ThreadTensorBoard
from src.callback.base import StageScheduleCallback
from src.config.base import CONFIG, update_CONFIG, get_CONFIG


def get_note(note_skip):
    note = "" if note_skip else input("\nPlease enter the notes for the current training model:\n > ")
    return note


def get_save_name(config):
    save_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(config["path"]["output"], save_name)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    return save_name, save_dir


def get_algorithm_kwargs(base_name, base_dir, config, config_inheritance):
    if base_name and config_inheritance:
        base_config = os.path.join(config["path"]["output"],
                                   base_dir,
                                   f"cfg_{base_name}.yaml")
        update_CONFIG(base_config)
    algorithm_kwargs = get_CONFIG(config=config,
                                  field="algorithm",
                                  try_keys=["n_steps", "batch_size", "n_epochs",
                                            "clip_range", "gamma", "gae_lambda",
                                            "device", "verbose", "vf_coef",
                                            "learning_rate", "policy", "policy_kwargs"])
    activation_fn = algorithm_kwargs["policy_kwargs"].get("activation_fn", "")
    algorithm_kwargs["policy_kwargs"]["activation_fn"] = nn.ELU if activation_fn == "ELU" else nn.Tanh
    return algorithm_kwargs


def make_train_env(config, *args, **kwargs):
    return make_gym_env(config, *args, **kwargs)


def ppo_train(base_name=None, config_inheritance=False, note_skip=False):

    note = get_note(note_skip)
    base_name, base_dir = check_base_name(base_name, config=CONFIG)
    save_name, save_dir = get_save_name(config=CONFIG)

    tensorboard_thread = ThreadTensorBoard(log_path=save_dir)
    tensorboard_thread.run()
    
    algorithm_kwargs = get_algorithm_kwargs(base_name, base_dir, CONFIG, config_inheritance)
    train_env = make_vec_env(lambda: make_train_env(CONFIG), n_envs=CONFIG["train"]["n_envs"])
    model, train_env, callback_kwargs = load_model(train_env, base_name, base_dir, CONFIG,
                                                   tensorboard_log=save_dir,
                                                   algorithm_kwargs=algorithm_kwargs)
    model.learn(total_timesteps=CONFIG["train"]["total_timesteps"],
                tb_log_name=f"log_{save_name}",
                callback=[ProgressBarCallback(**callback_kwargs),
                          StageScheduleCallback(**callback_kwargs),
                          CustomTensorboardCallback(**callback_kwargs),
                          AdaptiveLRCallback(**callback_kwargs),
                          CustomCheckpointCallback(save_name, save_dir, note,
                                                   **callback_kwargs)
                         ])
    
    tensorboard_thread.stop()
    train_env.close()
    plt.close('all')

    print("\nModel %s traning accomplished!\n" % save_name)
    return save_name
