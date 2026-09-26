"""Senses: turn world state into what one body perceives (allocentric frame)."""

from __future__ import annotations

from collections import deque

import numpy as np

from .actions import water_adjacent
from .body import DIRECTIONS, INT32_MAX, VISION_SIZE
from .clock import light_at
from .config import LIGHT_MAX, NEED_MAX
from .state import Agent, WorldState
from .tiles import Resource, Terrain, find_tiles

SCENTS = ("fruit", "humidity", "nest")
_CENTRE = VISION_SIZE // 2


def scent_field(
    terrain: np.ndarray, sources: list[tuple[int, int]], scent_range: int
) -> np.ndarray:
    """Multi-source BFS over 4-connected tiles, blocked only by rock.

    Intensity is 1000 * (R - d) // R for path distance d < R, else 0.
    """
    height, width = terrain.shape
    field = np.zeros((height, width), dtype=np.int16)
    passable = (terrain != Terrain.ROCK).tolist()
    distance = [[-1] * width for _ in range(height)]
    queue: deque[tuple[int, int]] = deque()
    for x, y in sources:
        distance[y][x] = 0
        queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        d = distance[y][x]
        field[y, x] = NEED_MAX * (scent_range - d) // scent_range
        if d + 1 >= scent_range:
            continue
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and distance[ny][nx] < 0 and passable[ny][nx]:
                distance[ny][nx] = d + 1
                queue.append((nx, ny))
    return field


class ScentFields:
    """Cached scent fields. Derived data: never snapshotted, rebuilt on restore."""

    def __init__(self, state: WorldState) -> None:
        cfg = state.config
        water = list(find_tiles(state.terrain, Terrain.WATER))
        nest = list(find_tiles(state.terrain, Terrain.NEST))
        self._humidity = scent_field(state.terrain, water, cfg.smell_range_humidity)
        self._nest = scent_field(state.terrain, nest, cfg.smell_range_nest)
        self._fruit = np.zeros_like(self._nest)
        self._fruit_version = -1

    def current(self, state: WorldState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fields in SCENTS order; the fruit field is refreshed when fruit changed."""
        if self._fruit_version != state.fruit_version:
            fruit = [
                (int(x), int(y)) for y, x in np.argwhere(state.resource_kind == Resource.FRUIT)
            ]
            self._fruit = scent_field(state.terrain, fruit, state.config.smell_range_fruit)
            self._fruit_version = state.fruit_version
        return self._fruit, self._humidity, self._nest


def vision_radius(light: int, state: WorldState) -> int:
    cfg = state.config
    span = cfg.vision_radius_day - cfg.vision_radius_night
    return cfg.vision_radius_night + span * light // LIGHT_MAX


def _window(array: np.ndarray, x: int, y: int, r: int, fill: int) -> np.ndarray:
    """(2r+1, 2r+1) cut-out centred on (x, y); out-of-bounds cells get `fill`."""
    height, width = array.shape
    out = np.full((2 * r + 1, 2 * r + 1), fill, dtype=array.dtype)
    x0, x1 = max(0, x - r), min(width, x + r + 1)
    y0, y1 = max(0, y - r), min(height, y + r + 1)
    out[y0 - (y - r) : y1 - (y - r), x0 - (x - r) : x1 - (x - r)] = array[y0:y1, x0:x1]
    return out


def _vision(state: WorldState, agent: Agent, light: int) -> np.ndarray:
    vision = np.zeros((VISION_SIZE, VISION_SIZE, 3), dtype=np.uint8)  # VOID beyond the radius
    r = vision_radius(light, state)
    lo, hi = _CENTRE - r, _CENTRE + r + 1
    vision[lo:hi, lo:hi, 0] = _window(state.terrain, agent.x, agent.y, r, Terrain.ROCK)
    vision[lo:hi, lo:hi, 1] = _window(state.resource_kind, agent.x, agent.y, r, Resource.NONE)
    others = _window(state.occupant, agent.x, agent.y, r, 0) != 0
    vision[lo:hi, lo:hi, 2] = others * 2
    vision[_CENTRE, _CENTRE, 2] = 1  # self
    return vision


def _smell(state: WorldState, scents: ScentFields, agent: Agent) -> np.ndarray:
    smell = np.zeros((len(SCENTS), 5), dtype=np.int16)
    samples = [(agent.x, agent.y)] + [(agent.x + dx, agent.y + dy) for dx, dy in DIRECTIONS]
    for row, field in enumerate(scents.current(state)):
        for column, (x, y) in enumerate(samples):
            if state.in_bounds(x, y):
                smell[row, column] = field[y, x]
    return smell


def observe(state: WorldState, scents: ScentFields, agent: Agent) -> dict[str, np.ndarray]:
    light = light_at(state.tick, state.config)
    x, y = agent.x, agent.y
    touch = [
        int(agent.bumped),
        int(state.resource_kind[y, x]),
        int(water_adjacent(state, x, y)),
        int(state.terrain[y, x] == Terrain.NEST),
    ]
    body = [agent.satiety, agent.hydration, agent.energy, agent.health, min(agent.age, INT32_MAX)]
    return {
        "smell": _smell(state, scents, agent),
        "vision": _vision(state, agent, light),
        "touch": np.array(touch, dtype=np.uint8),
        "body": np.array(body, dtype=np.int32),
        "env": np.array([light], dtype=np.int16),
    }
