from collections import deque

import numpy as np
import pytest

from neurogarden.engine import maps
from neurogarden.engine.tiles import WALKABLE, Terrain, find_tiles, parse_map


def test_drosoville_is_bundled():
    assert "drosoville" in maps.available()


def test_unknown_map_is_a_value_error():
    with pytest.raises(ValueError):
        maps.load("atlantis")


@pytest.fixture(scope="module")
def terrain():
    return parse_map(maps.load("drosoville")).terrain


def components(mask):
    """Number of 4-connected components of True cells."""
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        count += 1
        queue = deque([(x, y)])
        seen[y, x] = True
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                inside = 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                if inside and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((nx, ny))
    return count


def test_size_and_rock_border(terrain):
    assert terrain.shape == (24, 32)
    border = np.concatenate([terrain[0], terrain[-1], terrain[:, 0], terrain[:, -1]])
    assert (border == Terrain.ROCK).all()


def test_one_nest_two_ponds_four_trees(terrain):
    assert len(find_tiles(terrain, Terrain.NEST)) == 1
    assert components(terrain == Terrain.WATER) >= 2
    assert len(find_tiles(terrain, Terrain.TREE)) >= 4


def test_no_tree_within_six_tiles_of_water(terrain):
    water = find_tiles(terrain, Terrain.WATER)
    for tx, ty in find_tiles(terrain, Terrain.TREE):
        nearest = min(max(abs(tx - wx), abs(ty - wy)) for wx, wy in water)
        assert nearest > 6, (tx, ty, nearest)


def test_every_walkable_tile_is_reachable_from_the_nest(terrain):
    walkable = np.isin(terrain, [int(t) for t in WALKABLE])
    assert components(walkable) == 1
