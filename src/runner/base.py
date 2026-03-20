from src.runner.flat_ppo_runner import FlatPPORunner, FastFlatPPORunner, TrackFlatPPORunner
from src.runner.mode_hierarchical_ppo_runner import ModeHierarchicalPPORunner


__all__ = [
    "FlatPPORunner",
    "FastFlatPPORunner",
    "TrackFlatPPORunner",
    "ModeHierarchicalPPORunner",
    ]


from gymnasium.envs.registration import register


register(id="FastFlatLocomotionEnv",
         entry_point="src.env.fast_flat_locomotion_env:FlatLocomotionEnv")
register(id="TrackFlatLocomotionEnv",
         entry_point="src.env.track_flat_locomotion_env:FlatLocomotionEnv")
register(id="ModeConditionedLocomotionEnv",
         entry_point="src.env.mode_conditioned_locomotion_env:ModeConditionedLocomotionEnv")
register(id="ModeSelectionEnv",
         entry_point="src.env.mode_selection_env:ModeSelectionEnv")