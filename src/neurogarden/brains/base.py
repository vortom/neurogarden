"""The brain interface and a helper to run one episode."""

from __future__ import annotations

from typing import Protocol

import gymnasium
import numpy as np

from neurogarden.dojo.stats import EpisodeStats


class Brain(Protocol):
    """Decides. Receives the raw channel dict: the same payload a network client will get."""

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, observation: dict[str, np.ndarray]) -> int: ...


def run_episode(env: gymnasium.Env, brain: Brain, seed: int | None = None) -> EpisodeStats:
    """Run the brain until the fly dies or the episode is truncated."""
    observation, info = env.reset(seed=seed)
    brain.reset(seed)
    while True:
        observation, _, terminated, truncated, info = env.step(brain.act(observation))
        if terminated or truncated:
            return info["stats"]
