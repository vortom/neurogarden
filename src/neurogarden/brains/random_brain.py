"""The baseline: uniform random actions."""

from __future__ import annotations

import numpy as np

from neurogarden.engine.body import Action
from neurogarden.engine.rng import SplitMix64


class RandomBrain:
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = SplitMix64(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = SplitMix64(self._seed if seed is None else seed)

    def act(self, observation: dict[str, np.ndarray]) -> int:
        return self._rng.randbelow(len(Action))
