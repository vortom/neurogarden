"""The tiny feature vector: what a small network sees of a fly's observation.

Shared by the `TinyObservation` wrapper (training) and the evolved brain (living), so a
policy learned in the dojo reads exactly the same numbers in the garden.
"""

from __future__ import annotations

import numpy as np

from neurogarden.engine.config import LIGHT_MAX, NEED_MAX

DEFAULT_AGE_SCALE = 12000  # ten days: past that, "old" is as old as the feature gets
TINY_SIZE = 15 + 4 + 5 + 1  # smell (3 scents x own + N/E/S/W) + touch + body + light
FEATURES_VERSION = 1  # bump when the vector's layout or scaling changes


def tiny_features(observation: dict[str, np.ndarray], age_scale: int = DEFAULT_AGE_SCALE):
    """Flat float32 vector in [0, 1]: smell (15) + touch (4) + body (5) + env (1) = 25."""
    body = observation["body"].astype(np.float32)
    body[:4] /= NEED_MAX
    body[4] = min(body[4] / age_scale, 1.0)
    return np.concatenate(
        [
            observation["smell"].astype(np.float32).ravel() / NEED_MAX,
            np.minimum(observation["touch"], 1).astype(np.float32),
            body,
            observation["env"].astype(np.float32) / LIGHT_MAX,
        ]
    )
