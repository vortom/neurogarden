"""Observation payloads <-> the numpy channel dicts brains consume."""

from __future__ import annotations

import numpy as np

from .messages import BodyInfo, Channels


def encode_channels(observation: dict[str, np.ndarray]) -> Channels:
    return Channels(**{name: array.tolist() for name, array in observation.items()})


def decode_channels(channels: Channels, body: BodyInfo) -> dict[str, np.ndarray]:
    """The exact dict `World.observe` produces: shape and dtype come from the catalog."""
    decoded = {}
    for name, info in body.channels.items():
        array = np.array(getattr(channels, name), dtype=np.dtype(info.dtype))
        decoded[name] = array.reshape(tuple(info.shape))
    return decoded
