"""The brain interface and a helper to run one episode."""

from __future__ import annotations

from typing import Protocol

import gymnasium
import numpy as np

from neurogarden.dojo.stats import EpisodeStats
from neurogarden.engine.rng import SplitMix64

# XOR mask decorrelating a brain's seed from the world seed it is paired with.
_BRAIN_SEED_XOR = 0x9E3779B97F4A7C15


class Brain(Protocol):
    """Decides. Receives the raw channel dict: the same payload a network client will get."""

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, observation: dict[str, np.ndarray]) -> int: ...


def brain_seed(seed: int | None) -> int | None:
    """Seed for a brain's own RNG, decorrelated from the world seed it is paired with.

    Brain and world both use SplitMix64; resetting both from the same integer would
    draw the identical u64 stream. None stays None (an unseeded episode).
    """
    if seed is None:
        return None
    return SplitMix64(seed ^ _BRAIN_SEED_XOR).next_u64()


def run_episode(env: gymnasium.Env, brain: Brain, seed: int | None = None) -> EpisodeStats:
    """Run the brain until the fly dies or the episode is truncated."""
    observation, info = env.reset(seed=seed)
    brain.reset(brain_seed(seed))
    while True:
        observation, _, terminated, truncated, info = env.step(brain.act(observation))
        if terminated or truncated:
            return info["stats"]
