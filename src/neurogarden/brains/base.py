"""The brain interface and a helper to run one episode."""

from __future__ import annotations

import logging
import re
from typing import Protocol

import gymnasium
import numpy as np

from neurogarden.dojo.stats import EpisodeStats
from neurogarden.engine.rng import SplitMix64
from neurogarden.protocol.messages import SAY_MAX

log = logging.getLogger("neurogarden.brains")

# XOR mask decorrelating a brain's seed from the world seed it is paired with.
_BRAIN_SEED_XOR = 0x9E3779B97F4A7C15


class Brain(Protocol):
    """Decides. Receives the raw channel dict: the same payload a network client will get.

    A brain may also have a `thought() -> str`: what it would say, shown over its fly as a
    speech bubble by the runners that host it (see `think`).
    """

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, observation: dict[str, np.ndarray]) -> int: ...


THOUGHT_EVERY = 5  # ticks between looks at a brain's thought, when it has one
THOUGHT_REPEAT = 100  # ticks after which an unchanged thought is said again (bubbles fade at 150)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")  # what the protocol refuses in a `say`


def think(brain) -> str | None:
    """What a brain would say, if it has a `thought()`, cut to what a bubble holds."""
    thought = getattr(brain, "thought", None)
    if thought is None:
        return None
    try:
        text = str(thought())
    except Exception:  # a brain that cannot speak still gets to act
        log.exception("a brain failed to think aloud")
        return None
    return _CONTROL.sub("", text)[:SAY_MAX] or None


class Mouth:
    """Decides when a brain's thought is worth a `say`: when it changed, or now and then so
    the bubble does not fade — never every tick, since a say is a message on the wire."""

    def __init__(self, brain) -> None:
        self._brain = brain
        self._said: str | None = None
        self._said_at = -THOUGHT_REPEAT

    def speak(self, tick: int) -> str | None:
        """The text to say at this tick, or None to stay quiet."""
        if tick % THOUGHT_EVERY:
            return None
        text = think(self._brain)
        if text is None or (text == self._said and tick - self._said_at < THOUGHT_REPEAT):
            return None
        self._said, self._said_at = text, tick
        return text


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
