import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONWARNINGS"] = "ignore:pkg_resources is deprecated as an API"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
from glfw import GLFWError
import warnings
warnings.filterwarnings("ignore", category=GLFWError)

from src.runners.run_ppo import ppo_train, ppo_test


if __name__ == "__main__":


    ### Test Existing Model.
    # model_name = "2025-12-08_05-18-09_6"
    # model_name = ppo_test(test_name=model_name, n_tests=1)


    ### Train A New Model without Test.
    model_name = ppo_train(base_model_name=None)


    ### Continue Training Specified Model with Test.
    # model_name = "2025-11-25_22-33-36" # your model name
    # model_name = ppo_train(base_name=model_name, config_inheritance=False)
    # model_name = ppo_test(test_name=model_name)
