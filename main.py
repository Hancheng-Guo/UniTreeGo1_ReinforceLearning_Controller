from stable_baselines3 import PPO
from datetime import datetime

from config import CONFIG
from src.render.render_tensorboard import init_tensorboard
from src.env.make_env import make_env
from tools.display_model import display_model
from src.env.callbacks import RenderCallback


def main():

    init_tensorboard()

    train_env = make_env("train")
    demo_env  = make_env("demo")
    display_model(train_env)

    model = PPO(policy=CONFIG["algorithm"]["policy"],
                env=train_env,
                n_steps=CONFIG["algorithm"]["n_steps"],
                batch_size=CONFIG["algorithm"]["batch_size"],
                n_epochs=CONFIG["algorithm"]["n_epochs"],
                gamma=CONFIG["algorithm"]["gamma"],
                gae_lambda=CONFIG["algorithm"]["gae_lambda"],
                device=CONFIG["algorithm"]["device"],
                verbose=CONFIG["algorithm"]["verbose"],
                tensorboard_log=CONFIG["path"]["tensorboard"])
    model.learn(total_timesteps=CONFIG["algorithm"]["total_timesteps"],
                callback=RenderCallback(demo_env))
    model.save(CONFIG["path"]["tensorboard"] + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


if __name__ == "__main__":
    main()


