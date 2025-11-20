import stable_baselines3

def extract_gym_env(env):
    if isinstance(env, stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv):
        return env.envs[env.num_envs - 1].env
    else: # gymnasium.wrappers.common.TimeLimit
        return env
        