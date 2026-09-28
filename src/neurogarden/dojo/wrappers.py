"""Observation wrappers for small networks."""

from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium import spaces

from neurogarden.engine.body import VISION_SIZE
from neurogarden.engine.tiles import Resource, Terrain

from .features import DEFAULT_AGE_SCALE, TINY_SIZE, tiny_features

_VISION_CLASSES = (len(Terrain), len(Resource), 3)  # 3 = occupant classes: none/self/other


class TinyObservation(gymnasium.ObservationWrapper):
    """Flat float32 vector in [0, 1]: smell (15) + touch (4) + body (5) + env (1) = 25.

    include_vision=True appends the 7x7 view one-hot encoded (539 more values).
    """

    def __init__(self, env: gymnasium.Env, include_vision: bool = False) -> None:
        super().__init__(env)
        self._include_vision = include_vision
        self._age_scale = env.unwrapped.config.max_age or DEFAULT_AGE_SCALE
        size = TINY_SIZE
        if include_vision:
            size += VISION_SIZE * VISION_SIZE * sum(_VISION_CLASSES)
        self.observation_space = spaces.Box(0.0, 1.0, (size,), np.float32)

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        parts = [tiny_features(observation, self._age_scale)]
        if self._include_vision:
            for layer, classes in enumerate(_VISION_CLASSES):
                one_hot = np.eye(classes, dtype=np.float32)[observation["vision"][:, :, layer]]
                parts.append(one_hot.ravel())
        return np.concatenate(parts)
