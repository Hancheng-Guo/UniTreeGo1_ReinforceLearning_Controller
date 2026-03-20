import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONWARNINGS"] = "ignore:pkg_resources is deprecated as an API"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
from glfw import GLFWError
import warnings
warnings.filterwarnings("ignore", category=GLFWError)
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")

from src.runner.base import FastFlatPPORunner, TrackFlatPPORunner, ModeHierarchicalPPORunner


if __name__ == "__main__":

    ### Train A New Flat PPO model for rapid forward.
    flat_ppo = FastFlatPPORunner(base_name=None)
    flat_ppo.train()
    flat_ppo.test(n_tests=3)
    
    ### Train A Existed Flat PPO model for rapid forward.
    # flat_ppo = FastFlatPPORunner(base_name="2025-12-08_05-18-09_6")
    # flat_ppo.train(config_inheritance=False, note_skip=True)
    # flat_ppo.test(n_tests=3)
