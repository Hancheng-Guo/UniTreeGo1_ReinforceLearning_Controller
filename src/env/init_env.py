from gymnasium.envs.registration import register

register(
    id="MyUniTreeGo1",
    entry_point="src.env.unitree_go1:UniTreeGo1Env",
)