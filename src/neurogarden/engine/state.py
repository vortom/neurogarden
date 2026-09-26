"""The complete world state as plain integer data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .rng import SplitMix64
from .tiles import WALKABLE, ParsedMap, Resource, Terrain, find_tiles


@dataclass
class Agent:
    id: int
    body: str
    x: int
    y: int
    facing: int
    satiety: int
    hydration: int
    energy: int
    health: int
    age: int = 0
    alive: bool = True
    bumped: bool = False


@dataclass
class WorldState:
    config: Config
    rng: SplitMix64
    terrain: np.ndarray  # (H, W) uint8
    resource_kind: np.ndarray  # (H, W) uint8
    resource_amount: np.ndarray  # (H, W) int16
    resource_age: np.ndarray  # (H, W) int32
    occupant: np.ndarray  # (H, W) int32, 0 = empty
    agents: dict[int, Agent] = field(default_factory=dict)
    next_agent_id: int = 1
    tick: int = 0
    # Derived data below: never snapshotted or hashed.
    fruit_version: int = 0  # bumped whenever the set of fruit tiles changes
    trees: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        self.trees = find_tiles(self.terrain, Terrain.TREE)

    @property
    def height(self) -> int:
        return self.terrain.shape[0]

    @property
    def width(self) -> int:
        return self.terrain.shape[1]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.terrain[y, x] in WALKABLE

    def living(self) -> list[Agent]:
        """Living agents in ascending id order."""
        return [self.agents[i] for i in sorted(self.agents) if self.agents[i].alive]


def new_state(parsed: ParsedMap, config: Config, seed: int) -> WorldState:
    shape = parsed.terrain.shape
    state = WorldState(
        config=config,
        rng=SplitMix64(seed),
        terrain=parsed.terrain.copy(),
        resource_kind=np.zeros(shape, dtype=np.uint8),
        resource_amount=np.zeros(shape, dtype=np.int16),
        resource_age=np.zeros(shape, dtype=np.int32),
        occupant=np.zeros(shape, dtype=np.int32),
    )
    for x, y in parsed.fruit:
        state.resource_kind[y, x] = Resource.FRUIT
        state.resource_amount[y, x] = config.fruit_bites
    return state
