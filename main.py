from src.runner.run_ppo import ppo_train, ppo_test


if __name__ == "__main__":

    # model_name = "2025-11-25_13-57-54"
    # model_name = ppo_test(model_name=model_name)

    demo = False
    # demo = True
    base_model_name = "2025-11-25_13-57-54"
    model_name = ppo_train(base_model_name=base_model_name, demo=demo)
    model_name = ppo_test(model_name=model_name)

    print("\nModel %s traning accomplished!\n" % model_name)
