"""Bodies: what an agent can sense and do, described as data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .config import LIGHT_MAX, NEED_MAX

VISION_SIZE = 7
INT32_MAX = 2**31 - 1

# Direction index 0=N, 1=E, 2=S, 3=W as (dx, dy); y grows south.
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class Action(IntEnum):
    IDLE = 0
    MOVE_N = 1
    MOVE_E = 2
    MOVE_S = 3
    MOVE_W = 4
    CONSUME = 5
    REST = 6


MOVE_DIRECTION = {Action.MOVE_N: 0, Action.MOVE_E: 1, Action.MOVE_S: 2, Action.MOVE_W: 3}


@dataclass(frozen=True)
class ChannelSpec:
    shape: tuple[int, ...]
    dtype: str
    low: int
    high: int


@dataclass(frozen=True)
class BodyConfig:
    name: str
    index: int  # stable integer used in the state hash
    frame: str
    actions: tuple[Action, ...]
    channels: dict[str, ChannelSpec]


BODIES = {
    "fly": BodyConfig(
        name="fly",
        index=1,
        frame="allocentric",
        actions=tuple(Action),
        channels={
            "smell": ChannelSpec((3, 5), "int16", 0, NEED_MAX),
            "vision": ChannelSpec((VISION_SIZE, VISION_SIZE, 3), "uint8", 0, 255),
            "touch": ChannelSpec((4,), "uint8", 0, 255),
            "body": ChannelSpec((5,), "int32", 0, INT32_MAX),
            "env": ChannelSpec((1,), "int16", 0, LIGHT_MAX),
        },
    ),
}


def get_body(name: str) -> BodyConfig:
    try:
        return BODIES[name]
    except KeyError:
        raise ValueError(f"unknown body {name!r}") from None


def observation_spec(body: str) -> dict[str, ChannelSpec]:
    return dict(get_body(body).channels)


def action_names(body: str) -> tuple[str, ...]:
    return tuple(action.name.lower() for action in get_body(body).actions)
