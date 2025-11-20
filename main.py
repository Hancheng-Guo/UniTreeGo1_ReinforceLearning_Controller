from stable_baselines3 import PPO
from datetime import datetime

from tools.display_state import display_state
from tools.display_body import display_body
from src.env.make_env import make_env
from src.env.callbacks import RenderCallback
from src.tensorboard.init_tensorboard import init_tensorboard
from config import CONFIG


def main():

    init_tensorboard()

    train_env = make_env("train")
    demo_env  = make_env("demo")
    display_state(train_env)
    display_body(train_env)

    model = PPO("MlpPolicy", train_env, n_steps=256, verbose=1, tensorboard_log=CONFIG["path"]["tensorboard"])
    model.learn(total_timesteps=10_000_000, log_interval=1, callback=RenderCallback(demo_env, demo_freq=100))

    model.save(CONFIG["path"]["tensorboard"] + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

if __name__ == "__main__":
    main()


