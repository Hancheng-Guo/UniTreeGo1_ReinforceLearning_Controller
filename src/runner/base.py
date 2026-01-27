from src.runner.flat_ppo_runner import FlatPPORunner
from src.runner.mode_hierarchical_ppo_runner import ModeHierarchicalPPORunner


__all__ = [
    "FlatPPORunner",
    "ModeHierarchicalPPORunner",
    ]


from gymnasium.envs.registration import register


register(id="FlatLocomotionEnv",
         entry_point="src.env.flat_locomotion_env:FlatLocomotionEnv")
register(id="ModeConditionedLocomotionEnv",
         entry_point="src.env.mode_conditioned_locomotion_env:ModeConditionedLocomotionEnv")
register(id="ModeSelectionEnv",
         entry_point="src.env.mode_selection_env:ModeSelectionEnv")