from src.runner.run_ppo import ppo_train, ppo_test


if __name__ == "__main__":


    ### Test Existing Model.
    # model_name = "2025-11-25_22-33-36" # your model name
    # model_name = ppo_test(model_name=model_name)


    ### Train A New Model without Demo and Test.
    model_name = ppo_train(base_model_name=None, demo=False)


    ### Continue Training Specified Model with demo And Test.
    # base_model_name = "2025-11-25_22-33-36" # your model name
    # model_name = ppo_train(base_model_name=base_model_name, demo=True)
    # model_name = ppo_test(model_name=model_name)
