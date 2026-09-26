"""Fruit: placement, rot and spawning (tick order, step 5)."""

from __future__ import annotations

import numpy as np

from .events import Event
from .state import WorldState
from .tiles import Resource, Terrain, clip_window


def add_fruit(state: WorldState, x: int, y: int) -> None:
    state.resource_kind[y, x] = Resource.FRUIT
    state.resource_amount[y, x] = state.config.fruit_bites
    state.resource_age[y, x] = 0
    state.fruit_version += 1


def remove_fruit(state: WorldState, x: int, y: int) -> None:
    state.resource_kind[y, x] = Resource.NONE
    state.resource_amount[y, x] = 0
    state.resource_age[y, x] = 0
    state.fruit_version += 1


def fruit_near(state: WorldState, tx: int, ty: int) -> int:
    """Fruits of any origin within tree_radius (Chebyshev) of the tree."""
    x0, x1, y0, y1 = clip_window((state.height, state.width), tx, ty, state.config.tree_radius)
    return int((state.resource_kind[y0:y1, x0:x1] == Resource.FRUIT).sum())


def fruit_candidates(state: WorldState, tx: int, ty: int) -> list[tuple[int, int]]:
    """Ground tiles without a resource near the tree, row-major. Occupied tiles count."""
    x0, x1, y0, y1 = clip_window((state.height, state.width), tx, ty, state.config.tree_radius)
    free = (state.terrain[y0:y1, x0:x1] == Terrain.GROUND) & (
        state.resource_kind[y0:y1, x0:x1] == Resource.NONE
    )
    return [(int(x) + x0, int(y) + y0) for y, x in np.argwhere(free)]


def _place_near(state: WorldState, tx: int, ty: int) -> tuple[int, int] | None:
    candidates = fruit_candidates(state, tx, ty)
    if not candidates:
        return None
    x, y = candidates[state.rng.randbelow(len(candidates))]
    add_fruit(state, x, y)
    return x, y


def seed_initial_fruit(state: WorldState) -> None:
    for tx, ty in state.trees:
        for _ in range(state.config.tree_initial_fruit):
            _place_near(state, tx, ty)


def update_fruit(state: WorldState, events: list[Event]) -> None:
    cfg = state.config
    is_fruit = state.resource_kind == Resource.FRUIT
    state.resource_age[is_fruit] += 1
    for y, x in np.argwhere(is_fruit & (state.resource_age >= cfg.fruit_lifetime)):
        remove_fruit(state, int(x), int(y))
        events.append(Event(state.tick, "fruit_rotted", None, {"x": int(x), "y": int(y)}))
    for tx, ty in state.trees:
        if fruit_near(state, tx, ty) >= cfg.tree_max_fruit:
            continue
        if state.rng.randbelow(1000) >= cfg.fruit_spawn_permille:
            continue
        placed = _place_near(state, tx, ty)
        if placed is not None:
            events.append(
                Event(state.tick, "fruit_spawned", None, {"x": placed[0], "y": placed[1]})
            )
