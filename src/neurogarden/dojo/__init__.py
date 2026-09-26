"""The dojo: train against the engine in lockstep. Importing registers the environment."""

import gymnasium

from .env import NeuroGardenEnv
from .wrappers import TinyObservation

ENV_ID = "NeuroGarden/Drosoville-v0"

if ENV_ID not in gymnasium.registry:
    # No max_episode_steps: the env truncates itself, so Gymnasium adds no second time limit.
    gymnasium.register(id=ENV_ID, entry_point="neurogarden.dojo.env:NeuroGardenEnv")


def make(**kwargs) -> gymnasium.Env:
    """gymnasium.make for Drosoville; takes the NeuroGardenEnv constructor arguments."""
    return gymnasium.make(ENV_ID, **kwargs)


__all__ = ["ENV_ID", "NeuroGardenEnv", "TinyObservation", "make"]
