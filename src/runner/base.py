from src.runner.flat_ppo_runner import FlatPPORunner


__all__ = [
    "FlatPPORunner"
    ]


from gymnasium.envs.registration import register


register(
    id="FlatLocomotionEnv",
    entry_point="src.env.flat_locomotion_env:FlatLocomotionEnv",
)