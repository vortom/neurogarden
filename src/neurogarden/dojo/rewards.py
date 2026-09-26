"""Reward functions. Learner-side: the engine knows nothing about reward."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from neurogarden.engine.config import NEED_MAX
from neurogarden.engine.events import Event


@dataclass(frozen=True)
class BodyState:
    satiety: int
    hydration: int
    energy: int
    health: int
    age: int

    @classmethod
    def from_observation(cls, observation: dict[str, np.ndarray]) -> BodyState:
        return cls(*(int(value) for value in observation["body"]))


RewardFn = Callable[[BodyState, BodyState, list[Event], bool], float]

MAX_DRIVE = 3.0


def drive(body: BodyState) -> float:
    """Distance from the homeostatic set point; 0 = all needs full, 3 = all empty.

    Health is excluded on purpose: it is a consequence, not a drive.
    """
    needs = (body.satiety, body.hydration, body.energy)
    return sum(((NEED_MAX - need) / NEED_MAX) ** 2 for need in needs)


def wellbeing(prev: BodyState, body: BodyState, events: list[Event], died: bool) -> float:
    """Dense, bounded, positive while alive, aligned with lifespan. The default."""
    return 0.0 if died else 1.0 - drive(body) / MAX_DRIVE


def survival(prev: BodyState, body: BodyState, events: list[Event], died: bool) -> float:
    return 0.0 if died else 1.0


def make_homeostatic(death_penalty: float = 10.0) -> RewardFn:
    """Drive reduction, after Keramati & Gutkin.

    Undiscounted, these rewards telescope to drive(start) - drive(end), and most
    steps are slightly negative, so use a discount factor below 1 and keep the
    death penalty large enough that dying early never pays.
    """

    def homeostatic(prev: BodyState, body: BodyState, events: list[Event], died: bool) -> float:
        reward = drive(prev) - drive(body)
        return reward - death_penalty if died else reward

    return homeostatic


REWARDS: dict[str, RewardFn] = {
    "wellbeing": wellbeing,
    "survival": survival,
    "homeostatic": make_homeostatic(),
}


def resolve(reward: str | RewardFn) -> RewardFn:
    if callable(reward):
        return reward
    try:
        return REWARDS[reward]
    except KeyError:
        raise ValueError(f"unknown reward {reward!r}; choose from {sorted(REWARDS)}") from None
