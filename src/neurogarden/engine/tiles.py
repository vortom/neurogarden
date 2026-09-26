"""Terrain and resource enums, and the text-map parser."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Terrain(IntEnum):
    VOID = 0  # never in a map; vision-only ("not seen")
    GROUND = 1
    ROCK = 2
    WATER = 3
    TREE = 4
    NEST = 5


class Resource(IntEnum):
    NONE = 0
    FRUIT = 1


WALKABLE = (Terrain.GROUND, Terrain.NEST)

_CHARS = {
    ".": Terrain.GROUND,
    "#": Terrain.ROCK,
    "~": Terrain.WATER,
    "T": Terrain.TREE,
    "N": Terrain.NEST,
    "F": Terrain.GROUND,  # ground with a full fruit at tick 0
}


class MapError(ValueError):
    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"{message} (line {line}, column {column})")
        self.line = line
        self.column = column


@dataclass(frozen=True)
class ParsedMap:
    terrain: np.ndarray  # (H, W) uint8
    fruit: tuple[tuple[int, int], ...]  # (x, y) of 'F' tiles, row-major
    text: str  # normalised map text


def parse_map(text: str) -> ParsedMap:
    lines = [line.strip() for line in text.strip().splitlines()]
    if not lines or not lines[0]:
        raise MapError("map is empty", 1, 1)
    width = len(lines[0])
    terrain = np.zeros((len(lines), width), dtype=np.uint8)
    fruit: list[tuple[int, int]] = []
    for y, line in enumerate(lines):
        if len(line) != width:
            raise MapError(
                f"row has {len(line)} tiles, expected {width}", y + 1, min(len(line), width) + 1
            )
        for x, char in enumerate(line):
            if char not in _CHARS:
                raise MapError(f"unknown tile {char!r}", y + 1, x + 1)
            terrain[y, x] = _CHARS[char]
            if char == "F":
                fruit.append((x, y))
    return ParsedMap(terrain=terrain, fruit=tuple(fruit), text="\n".join(lines))


def find_tiles(terrain: np.ndarray, kind: Terrain) -> tuple[tuple[int, int], ...]:
    """(x, y) of every tile of the given terrain, in row-major order."""
    return tuple((int(x), int(y)) for y, x in np.argwhere(terrain == kind))


def clip_window(shape: tuple[int, int], x: int, y: int, r: int) -> tuple[int, int, int, int]:
    """Bounds x0, x1, y0, y1 of a (2r+1)x(2r+1) window centred on (x, y), clipped to shape.

    shape is (height, width), matching an array's .shape.
    """
    height, width = shape
    return max(0, x - r), min(width, x + r + 1), max(0, y - r), min(height, y + r + 1)
