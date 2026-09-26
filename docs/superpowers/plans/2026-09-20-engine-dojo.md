# NeuroGarden Engine + Dojo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure, deterministic simulation engine for the world of Drosoville, a Gymnasium environment around it, two example brains and a terminal viewer.

**Architecture:** The engine is a pure library of small modules that operate on one plain-integer `WorldState`; a thin `World` facade exposes `from_map / spawn / observe / step / snapshot / restore / state_hash`. The dojo wraps a `World` as a single-agent Gymnasium environment and owns everything learner-side (reward, stats, fitness). Brains consume the raw observation dict, the same payload a network client will receive in sub-project 2.

**Tech Stack:** Python ≥ 3.12, numpy, gymnasium, pytest, ruff, uv, hatchling.

**Spec:** `docs/superpowers/specs/2026-09-20-engine-dojo-design.md` (binding), with `docs/superpowers/specs/2026-09-20-neurogarden-architecture-design.md` as its parent. Read both before starting.

## Global Constraints

- Python `>=3.12`; runtime dependencies exactly `numpy` and `gymnasium`; dev dependencies `pytest` and `ruff`; src layout; hatchling; managed with `uv`.
- `engine/` imports nothing from `dojo/` or `brains/`, does no I/O except reading a bundled map through `importlib.resources`, reads no wall clock, knows no reward, and uses no library RNG (`random`, `numpy.random`) — only `engine/rng.py`.
- World state is integer-only. Needs and health stay in `[0, 1000]`.
- Coordinates are `(x, y)`, origin top-left, `y` grows south; numpy layers are indexed `[y, x]`. Directions are `0=N, 1=E, 2=S, 3=W` everywhere.
- A brain never receives world coordinates (agent stream ≠ spectator stream).
- `RULES_VERSION = 1`. After Task 14 checks in the golden replay, any change that alters state evolution or observations must bump `RULES_VERSION` and regenerate the golden file in the same commit.
- ruff: line length 100, rules `E, F, I, UP, B`. Before every commit run `uv run ruff check .` and `uv run ruff format --check .`; both must be clean.
- Shell (user's global rule): run each command as its own call; never chain with `&&`, `;` or `|`; prefer absolute paths or `git -C <repo>` over `cd`. Commands below assume the repository root as working directory.
- Git: work on branch `feat/engine-dojo`, created from `docs/neurogarden-design` (the specs and this plan live there). Commit after every task with a conventional-commit message, plus the attribution trailer your session instructions specify.
- Every code block in this plan was run as a throwaway prototype: 164 tests pass on CPython 3.14.2, numpy 2.5.3, gymnasium 1.3.0, pytest 9.1.1, ruff 0.16.8. If a step's result differs from the expected one, stop and find out why; never edit a test to make it fit.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml`, `.gitignore`, `README.md` | Packaging, tooling, front page |
| `src/neurogarden/__init__.py` | Package version |
| `src/neurogarden/engine/rng.py` | SplitMix64: the engine's only randomness |
| `src/neurogarden/engine/config.py` | `Config` dataclass, `RULES_VERSION`, shared constants |
| `src/neurogarden/engine/tiles.py` | `Terrain` / `Resource` enums, text-map parser |
| `src/neurogarden/engine/body.py` | `Action`, directions, body registry, observation spec |
| `src/neurogarden/engine/events.py` | `Event`, coordinate stripping for the agent stream |
| `src/neurogarden/engine/clock.py` | Light, night, day number from the tick |
| `src/neurogarden/engine/state.py` | `Agent`, `WorldState`, `new_state` |
| `src/neurogarden/engine/fruit.py` | Fruit placement, rot, spawning |
| `src/neurogarden/engine/actions.py` | Apply one agent's action |
| `src/neurogarden/engine/metabolism.py` | Needs, health, death |
| `src/neurogarden/engine/senses.py` | Scent fields (cached) and observations |
| `src/neurogarden/engine/snapshot.py` | Snapshot, restore, canonical state hash |
| `src/neurogarden/engine/world.py` | `World` facade and `StepResult` |
| `src/neurogarden/engine/maps/` | Bundled maps and their loader |
| `src/neurogarden/engine/replay.py` | Record and verify replays |
| `src/neurogarden/dojo/rewards.py` | `BodyState`, drive, reward functions |
| `src/neurogarden/dojo/stats.py` | `EpisodeStats`, `StatsTracker`, fitness |
| `src/neurogarden/dojo/render_ansi.py` | Emoji / ASCII frame |
| `src/neurogarden/dojo/env.py` | `NeuroGardenEnv` |
| `src/neurogarden/dojo/wrappers.py` | `TinyObservation` |
| `src/neurogarden/dojo/watch.py` | Terminal viewer CLI |
| `src/neurogarden/brains/` | `Brain` protocol, `run_episode`, `RandomBrain`, `ScriptedBrain` |
| `tests/test_*.py` | One test file per module, flat, each self-contained |

Every file is created whole in exactly one task. The three package `__init__.py` files start as docstrings in Task 1 and are each replaced whole once, in the task that gives them exports.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md` (replaces the one-line README)
- Create: `src/neurogarden/__init__.py`, `src/neurogarden/engine/__init__.py`, `src/neurogarden/dojo/__init__.py`, `src/neurogarden/brains/__init__.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable package `neurogarden` with `neurogarden.__version__ == "0.1.0"`; the commands `uv run pytest` and `uv run ruff`.

- [ ] **Step 1: Create the branch**

Run: `git switch -c feat/engine-dojo docs/neurogarden-design`
Expected: `Switched to a new branch 'feat/engine-dojo'`

- [ ] **Step 2: Create `tests/test_package.py`**

```python
import neurogarden


def test_package_imports_and_has_a_version():
    assert neurogarden.__version__ == "0.1.0"
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "neurogarden"
version = "0.1.0"
description = "Small worlds. Strange minds. A life simulator for small brains."
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.1",
    "gymnasium>=1.0",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "ruff>=0.6",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/neurogarden"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: balance guard and benchmark (tens of seconds)"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 4: Create `.gitignore`**

```text
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
dist/
```

- [ ] **Step 5: Replace `README.md` with a stub (the full README comes in Task 21)**

```markdown
# NeuroGarden

*Small worlds. Strange minds.*

A life simulator for small brains. Work in progress: see `docs/superpowers/specs/`.
```

- [ ] **Step 6: Create `src/neurogarden/__init__.py`**

```python
"""NeuroGarden: small worlds, strange minds."""

__version__ = "0.1.0"
```

- [ ] **Step 7: Create the three sub-package markers**

`src/neurogarden/engine/__init__.py`:

```python
"""The NeuroGarden engine: a pure, deterministic simulation library."""
```

`src/neurogarden/dojo/__init__.py`:

```python
"""The dojo: train against the engine in lockstep."""
```

`src/neurogarden/brains/__init__.py`:

```python
"""Example brains."""
```

- [ ] **Step 8: Install and run the test**

A scaffold has no meaningful red step: until the package directory exists, `uv sync` itself fails.

Run: `uv sync`
Expected: a `.venv` is created and numpy, gymnasium, pytest and ruff are installed. If no wheel exists for the default interpreter, run `uv python pin 3.13` and `uv sync` again.

Run: `uv run pytest tests/test_package.py -q`
Expected: `1 passed`

- [ ] **Step 9: Lint**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`
Expected: `All checks passed!` and no file needing a reformat.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock .gitignore README.md src tests
git commit -m "chore: scaffold the neurogarden package"
```

---

### Task 2: SplitMix64

**Files:**
- Create: `src/neurogarden/engine/rng.py`
- Test: `tests/test_rng.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SplitMix64(seed: int = 0)` with attribute `state: int` (the whole generator state, a `u64`), `next_u64() -> int`, `randbelow(n: int) -> int` (raises `ValueError` for `n <= 0`), `shuffle(items: list) -> None` (Fisher–Yates: `i` from `n-1` down to `1`, `j = randbelow(i+1)`; fewer than two items draw nothing).

- [ ] **Step 1: Create `tests/test_rng.py`**

```python
import pytest

from neurogarden.engine.rng import SplitMix64


def test_reference_vectors_seed_0():
    rng = SplitMix64(0)
    assert [rng.next_u64() for _ in range(3)] == [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
    ]


def test_randbelow_reference_seed_42():
    rng = SplitMix64(42)
    assert [rng.randbelow(1000) for _ in range(5)] == [413, 291, 858, 764, 250]


def test_randbelow_stays_in_range():
    rng = SplitMix64(7)
    assert all(0 <= rng.randbelow(3) < 3 for _ in range(1000))


def test_randbelow_rejects_non_positive():
    with pytest.raises(ValueError):
        SplitMix64(1).randbelow(0)


def test_state_is_one_integer_and_restorable():
    rng = SplitMix64(123)
    rng.next_u64()
    clone = SplitMix64(0)
    clone.state = rng.state
    assert clone.next_u64() == rng.next_u64()


def test_seed_is_reduced_mod_2_64():
    assert SplitMix64(2**64 + 5).state == 5


def test_shuffle_is_deterministic_and_a_permutation():
    a, b = list(range(10)), list(range(10))
    SplitMix64(9).shuffle(a)
    SplitMix64(9).shuffle(b)
    assert a == b
    assert sorted(a) == list(range(10))
    assert a != list(range(10))


def test_shuffle_of_single_item_draws_nothing():
    rng = SplitMix64(5)
    before = rng.state
    rng.shuffle([1])
    assert rng.state == before
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_rng.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.rng'`

- [ ] **Step 3: Create `src/neurogarden/engine/rng.py`**

```python
"""SplitMix64: the only source of randomness inside the engine.

Fully specified so that replays survive dependency upgrades and a future port
to another language can reproduce every draw bit for bit.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1


class SplitMix64:
    __slots__ = ("state",)

    def __init__(self, seed: int = 0) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def randbelow(self, n: int) -> int:
        """Uniform integer in [0, n) by rejection sampling (no modulo bias)."""
        if n <= 0:
            raise ValueError("n must be positive")
        limit = ((1 << 64) // n) * n
        while True:
            r = self.next_u64()
            if r < limit:
                return r % n

    def shuffle(self, items: list) -> None:
        """Fisher-Yates, in place. A list of fewer than two items draws nothing."""
        for i in range(len(items) - 1, 0, -1):
            j = self.randbelow(i + 1)
            items[i], items[j] = items[j], items[i]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_rng.py -q`
Expected: `8 passed`. The reference vectors come from the spec (section 6); if they fail, the constants are mistyped.

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/rng.py tests/test_rng.py
git commit -m "feat(engine): add SplitMix64, the engine's only RNG"
```

---

### Task 3: Config

**Files:**
- Create: `src/neurogarden/engine/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: frozen dataclass `Config` (all fields and defaults from spec section 3.8, plus the day phases `dawn_end=100`, `dusk_start=700`, `night_start=800`), `Config.to_dict() -> dict`, `Config.from_dict(data) -> Config` (`ValueError` on unknown fields); constants `RULES_VERSION = 1`, `NEED_MAX = 1000`, `LIGHT_MAX = 1000`, `NIGHT_LIGHT_THRESHOLD = 500`, `MAX_VISION_RADIUS = 3`. Invalid values raise `ValueError` at construction.

- [ ] **Step 1: Create `tests/test_config.py`**

```python
import dataclasses

import pytest

from neurogarden.engine.config import RULES_VERSION, Config


def test_defaults_are_valid_and_match_spec():
    cfg = Config()
    assert (cfg.initial_satiety, cfg.initial_hydration, cfg.initial_energy) == (700, 700, 800)
    assert (cfg.energy_move, cfg.energy_move_night) == (-2, -4)
    assert cfg.max_age is None
    assert cfg.day_length == 1200
    assert RULES_VERSION == 1


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().day_length = 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"initial_satiety": 0},
        {"initial_health": 1001},
        {"fruit_bites": 0},
        {"starve_damage": -1},
        {"max_age": 0},
        {"dawn_end": 0},
        {"dusk_start": 900},
        {"night_start": 1300},
        {"vision_radius_day": 4},
        {"vision_radius_night": 3, "vision_radius_day": 2},
        {"fruit_spawn_permille": 1001},
        {"tree_initial_fruit": 4},
        {"satiety_drain": 1.5},
        {"fruit_bites": True},
    ],
)
def test_nonsense_is_rejected(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


def test_dict_round_trip():
    cfg = Config(max_age=5000, fruit_bites=2)
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError):
        Config.from_dict({"wings": 2})
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.config'`

- [ ] **Step 3: Create `src/neurogarden/engine/config.py`**

```python
"""Every engine tunable lives in one frozen, validated dataclass."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

# Bump on any change that alters state evolution or observations for identical inputs.
RULES_VERSION = 1

NEED_MAX = 1000
LIGHT_MAX = 1000
NIGHT_LIGHT_THRESHOLD = 500
MAX_VISION_RADIUS = 3

_POSITIVE = (
    "fruit_bites",
    "fruit_lifetime",
    "tree_radius",
    "tree_max_fruit",
    "smell_range_fruit",
    "smell_range_humidity",
    "smell_range_nest",
)
_NON_NEGATIVE = (
    "satiety_drain",
    "hydration_drain",
    "energy_rest",
    "energy_rest_nest",
    "fruit_bite_satiety",
    "drink_hydration",
    "starve_damage",
    "regen_threshold",
    "regen_amount",
    "tree_initial_fruit",
)
_INITIAL = ("initial_satiety", "initial_hydration", "initial_energy", "initial_health")


@dataclass(frozen=True)
class Config:
    # needs
    initial_satiety: int = 700
    initial_hydration: int = 700
    initial_energy: int = 800
    initial_health: int = 1000
    satiety_drain: int = 1
    hydration_drain: int = 1
    # energy delta per action
    energy_idle: int = -1
    energy_move: int = -2
    energy_move_night: int = -4
    energy_consume: int = -1
    energy_rest: int = 8
    energy_rest_nest: int = 12
    # feeding
    fruit_bites: int = 4
    fruit_bite_satiety: int = 150
    drink_hydration: int = 150
    # health
    starve_damage: int = 5
    regen_threshold: int = 300
    regen_amount: int = 1
    max_age: int | None = None
    # time: dawn ramp [0, dawn_end), day, dusk ramp [dusk_start, night_start), night
    day_length: int = 1200
    dawn_end: int = 100
    dusk_start: int = 700
    night_start: int = 800
    # vision
    vision_radius_day: int = 3
    vision_radius_night: int = 1
    # smell
    smell_range_fruit: int = 12
    smell_range_humidity: int = 10
    smell_range_nest: int = 16
    # fruit
    tree_radius: int = 2
    tree_max_fruit: int = 3
    tree_initial_fruit: int = 2
    fruit_spawn_permille: int = 8
    fruit_lifetime: int = 1500

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "max_age" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{f.name} must be an int, got {value!r}")
        if self.max_age is not None and self.max_age <= 0:
            raise ValueError("max_age must be positive or None")
        for name in _INITIAL:
            if not 0 < getattr(self, name) <= NEED_MAX:
                raise ValueError(f"{name} must be in 1..{NEED_MAX}")
        for name in _POSITIVE:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if not 0 < self.dawn_end <= self.dusk_start < self.night_start <= self.day_length:
            raise ValueError("need 0 < dawn_end <= dusk_start < night_start <= day_length")
        if not 0 <= self.vision_radius_night <= self.vision_radius_day <= MAX_VISION_RADIUS:
            raise ValueError(f"need 0 <= night radius <= day radius <= {MAX_VISION_RADIUS}")
        if not 0 <= self.fruit_spawn_permille <= 1000:
            raise ValueError("fruit_spawn_permille must be in 0..1000")
        if self.tree_initial_fruit > self.tree_max_fruit:
            raise ValueError("tree_initial_fruit must not exceed tree_max_fruit")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**data)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config.py -q`
Expected: `18 passed`

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/config.py tests/test_config.py
git commit -m "feat(engine): add validated Config and RULES_VERSION"
```

---

### Task 4: Tiles and the map parser

**Files:**
- Create: `src/neurogarden/engine/tiles.py`
- Test: `tests/test_tiles.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Terrain` (`VOID=0, GROUND=1, ROCK=2, WATER=3, TREE=4, NEST=5`), `Resource` (`NONE=0, FRUIT=1`), `WALKABLE = (Terrain.GROUND, Terrain.NEST)`, `MapError(ValueError)` with `.line` and `.column` (1-based), `ParsedMap(terrain: np.ndarray uint8 (H, W), fruit: tuple[(x, y), ...], text: str)`, `parse_map(text: str) -> ParsedMap`, `find_tiles(terrain, kind) -> tuple[(x, y), ...]` in row-major order.

- [ ] **Step 1: Create `tests/test_tiles.py`**

```python
import numpy as np
import pytest

from neurogarden.engine.tiles import MapError, Terrain, find_tiles, parse_map

SMALL = """
#####
#.F~#
#TN.#
#####
"""


def test_parse_terrain_and_fruit():
    parsed = parse_map(SMALL)
    assert parsed.terrain.shape == (4, 5)
    assert parsed.terrain.dtype == np.uint8
    assert parsed.terrain[1, 1] == Terrain.GROUND
    assert parsed.terrain[1, 2] == Terrain.GROUND  # 'F' is ground
    assert parsed.terrain[1, 3] == Terrain.WATER
    assert parsed.terrain[2, 1] == Terrain.TREE
    assert parsed.terrain[2, 2] == Terrain.NEST
    assert parsed.terrain[0, 0] == Terrain.ROCK
    assert parsed.fruit == ((2, 1),)


def test_text_is_normalised():
    assert parse_map(SMALL).text == "#####\n#.F~#\n#TN.#\n#####"


def test_void_never_appears_in_a_map():
    assert not (parse_map(SMALL).terrain == Terrain.VOID).any()


def test_empty_map_is_an_error():
    with pytest.raises(MapError) as err:
        parse_map("  \n ")
    assert (err.value.line, err.value.column) == (1, 1)


def test_ragged_row_reports_line_and_column():
    with pytest.raises(MapError) as err:
        parse_map("###\n##\n###")
    assert (err.value.line, err.value.column) == (2, 3)


def test_unknown_character_reports_line_and_column():
    with pytest.raises(MapError) as err:
        parse_map("###\n#?#\n###")
    assert (err.value.line, err.value.column) == (2, 2)


def test_find_tiles_is_row_major_xy():
    terrain = parse_map("T.T\n.T.").terrain
    assert find_tiles(terrain, Terrain.TREE) == ((0, 0), (2, 0), (1, 1))
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_tiles.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.tiles'`

- [ ] **Step 3: Create `src/neurogarden/engine/tiles.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_tiles.py -q`
Expected: `7 passed`

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/tiles.py tests/test_tiles.py
git commit -m "feat(engine): add terrain enums and the text-map parser"
```

---

### Task 5: Bodies and events

**Files:**
- Create: `src/neurogarden/engine/body.py`, `src/neurogarden/engine/events.py`
- Test: `tests/test_body_events.py`

**Interfaces:**
- Consumes: `NEED_MAX`, `LIGHT_MAX` from `config.py`.
- Produces (`body.py`): `Action` IntEnum (`IDLE=0, MOVE_N=1, MOVE_E=2, MOVE_S=3, MOVE_W=4, CONSUME=5, REST=6`), `DIRECTIONS = ((0,-1),(1,0),(0,1),(-1,0))`, `MOVE_DIRECTION: dict[Action, int]`, `VISION_SIZE = 7`, `INT32_MAX`, `ChannelSpec(shape, dtype, low, high)`, `BodyConfig(name, index, frame, actions, channels)`, `BODIES`, `get_body(name) -> BodyConfig` (`ValueError` if unknown), `observation_spec(body) -> dict[str, ChannelSpec]`, `action_names(body) -> tuple[str, ...]`.
- Produces (`events.py`): frozen `Event(tick: int, type: str, agent_id: int | None = None, data: dict = {})` (`ValueError` for a type outside `EVENT_TYPES`), `for_agent(event) -> Event` with the keys `x`, `y`, `from`, `to` removed from `data`.

- [ ] **Step 1: Create `tests/test_body_events.py`**

```python
import pytest

from neurogarden.engine.body import (
    DIRECTIONS,
    MOVE_DIRECTION,
    Action,
    action_names,
    get_body,
    observation_spec,
)
from neurogarden.engine.events import Event, for_agent


def test_fly_has_seven_actions_in_spec_order():
    assert action_names("fly") == (
        "idle",
        "move_n",
        "move_e",
        "move_s",
        "move_w",
        "consume",
        "rest",
    )
    assert [int(a) for a in Action] == [0, 1, 2, 3, 4, 5, 6]


def test_directions_are_n_e_s_w_with_y_growing_south():
    assert DIRECTIONS == ((0, -1), (1, 0), (0, 1), (-1, 0))
    assert MOVE_DIRECTION[Action.MOVE_N] == 0
    assert MOVE_DIRECTION[Action.MOVE_W] == 3


def test_observation_spec_shapes_and_dtypes():
    spec = observation_spec("fly")
    assert {name: (s.shape, s.dtype) for name, s in spec.items()} == {
        "smell": ((3, 5), "int16"),
        "vision": ((7, 7, 3), "uint8"),
        "touch": ((4,), "uint8"),
        "body": ((5,), "int32"),
        "env": ((1,), "int16"),
    }


def test_fly_is_allocentric():
    assert get_body("fly").frame == "allocentric"


def test_unknown_body_is_a_value_error():
    with pytest.raises(ValueError):
        get_body("dragon")


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValueError):
        Event(0, "exploded")


def test_for_agent_strips_coordinates_only():
    event = Event(3, "moved", 1, {"from": (1, 1), "to": (1, 2), "direction": 2})
    stripped = for_agent(event)
    assert stripped.data == {"direction": 2}
    assert (stripped.tick, stripped.type, stripped.agent_id) == (3, "moved", 1)
    assert event.data["from"] == (1, 1)  # original untouched
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_body_events.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.body'`

- [ ] **Step 3: Create `src/neurogarden/engine/body.py`**

```python
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
```

- [ ] **Step 4: Create `src/neurogarden/engine/events.py`**

```python
"""Events: what happened during a tick."""

from __future__ import annotations

from dataclasses import dataclass, field

EVENT_TYPES = frozenset(
    {
        "moved",
        "bumped",
        "ate",
        "drank",
        "consume_failed",
        "rested",
        "invalid_action",
        "need_depleted",
        "damaged",
        "died",
        "fruit_spawned",
        "fruit_rotted",
    }
)

# Keys that carry world coordinates. Brains must never see them.
_COORDINATE_KEYS = ("x", "y", "from", "to")


@dataclass(frozen=True)
class Event:
    tick: int
    type: str
    agent_id: int | None = None
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.type!r}")


def for_agent(event: Event) -> Event:
    """Copy of the event with world coordinates stripped (agent stream)."""
    data = {key: value for key, value in event.data.items() if key not in _COORDINATE_KEYS}
    return Event(event.tick, event.type, event.agent_id, data)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_body_events.py -q`
Expected: `7 passed`

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/body.py src/neurogarden/engine/events.py tests/test_body_events.py
git commit -m "feat(engine): describe bodies, actions and events as data"
```

---

### Task 6: Clock and world state

**Files:**
- Create: `src/neurogarden/engine/clock.py`, `src/neurogarden/engine/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `Config`, `LIGHT_MAX`, `NIGHT_LIGHT_THRESHOLD`; `SplitMix64`; `ParsedMap`, `Terrain`, `Resource`, `WALKABLE`, `find_tiles`.
- Produces (`clock.py`): `light_at(tick, config) -> int` in `[0, 1000]`, `is_night(tick, config) -> bool` (`light < 500`), `day_number(tick, config) -> int` (from 1).
- Produces (`state.py`): dataclass `Agent(id, body, x, y, facing, satiety, hydration, energy, health, age=0, alive=True, bumped=False)`; dataclass `WorldState(config, rng, terrain, resource_kind, resource_amount, resource_age, occupant, agents={}, next_agent_id=1, tick=0, fruit_version=0, trees=())` with `width`, `height`, `in_bounds(x, y)`, `walkable(x, y)`, `living() -> list[Agent]` (ascending id); `new_state(parsed, config, seed) -> WorldState`. `fruit_version` and `trees` are derived data: never snapshotted or hashed.

- [ ] **Step 1: Create `tests/test_state.py`**

```python
import numpy as np

from neurogarden.engine.clock import day_number, is_night, light_at
from neurogarden.engine.config import Config
from neurogarden.engine.state import Agent, new_state
from neurogarden.engine.tiles import Resource, parse_map

MAP = """
#####
#.F~#
#TN.#
#####
"""


def make_state(**overrides):
    return new_state(parse_map(MAP), Config(**overrides), seed=1)


def test_layers_have_spec_dtypes_and_shape():
    state = make_state()
    assert (state.width, state.height) == (5, 4)
    assert state.terrain.dtype == np.uint8
    assert state.resource_kind.dtype == np.uint8
    assert state.resource_amount.dtype == np.int16
    assert state.resource_age.dtype == np.int32
    assert state.occupant.dtype == np.int32


def test_map_fruit_starts_full_and_fresh():
    state = make_state(fruit_bites=3)
    assert state.resource_kind[1, 2] == Resource.FRUIT
    assert state.resource_amount[1, 2] == 3
    assert state.resource_age[1, 2] == 0


def test_walkable_is_ground_and_nest_in_bounds_only():
    state = make_state()
    assert state.walkable(1, 1)  # ground
    assert state.walkable(2, 2)  # nest
    assert not state.walkable(3, 1)  # water
    assert not state.walkable(1, 2)  # tree
    assert not state.walkable(0, 0)  # rock
    assert not state.walkable(-1, 1)
    assert not state.walkable(5, 1)


def test_trees_are_derived_from_terrain():
    assert make_state().trees == ((1, 2),)


def test_living_is_ascending_and_skips_the_dead():
    state = make_state()
    for agent_id, alive in ((2, True), (1, True), (3, False)):
        state.agents[agent_id] = Agent(agent_id, "fly", 1, 1, 2, 700, 700, 800, 1000, alive=alive)
    assert [a.id for a in state.living()] == [1, 2]


def test_state_does_not_alias_the_parsed_map():
    parsed = parse_map(MAP)
    state = new_state(parsed, Config(), seed=0)
    state.terrain[1, 1] = 2
    assert parsed.terrain[1, 1] == 1


def test_light_follows_the_spec_table():
    cfg = Config()
    assert light_at(0, cfg) == 0
    assert light_at(50, cfg) == 500
    assert light_at(100, cfg) == 1000
    assert light_at(699, cfg) == 1000
    assert light_at(700, cfg) == 1000
    assert light_at(750, cfg) == 500
    assert light_at(800, cfg) == 0
    assert light_at(1199, cfg) == 0
    assert light_at(1200 + 50, cfg) == 500  # wraps every day


def test_night_is_light_below_500():
    cfg = Config()
    assert is_night(49, cfg)
    assert not is_night(50, cfg)
    assert not is_night(750, cfg)
    assert is_night(751, cfg)
    assert is_night(1000, cfg)


def test_day_number_starts_at_one():
    cfg = Config()
    assert day_number(0, cfg) == 1
    assert day_number(1199, cfg) == 1
    assert day_number(1200, cfg) == 2
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_state.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.clock'`

- [ ] **Step 3: Create `src/neurogarden/engine/clock.py`**

```python
"""Day and night as a pure function of the tick."""

from __future__ import annotations

from .config import LIGHT_MAX, NIGHT_LIGHT_THRESHOLD, Config


def light_at(tick: int, config: Config) -> int:
    phase = tick % config.day_length
    if phase < config.dawn_end:
        return phase * LIGHT_MAX // config.dawn_end
    if phase < config.dusk_start:
        return LIGHT_MAX
    if phase < config.night_start:
        return (config.night_start - phase) * LIGHT_MAX // (config.night_start - config.dusk_start)
    return 0


def is_night(tick: int, config: Config) -> bool:
    return light_at(tick, config) < NIGHT_LIGHT_THRESHOLD


def day_number(tick: int, config: Config) -> int:
    return tick // config.day_length + 1
```

- [ ] **Step 4: Create `src/neurogarden/engine/state.py`**

```python
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
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_state.py -q`
Expected: `9 passed`

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/clock.py src/neurogarden/engine/state.py tests/test_state.py
git commit -m "feat(engine): add the day clock and the integer world state"
```

---

### Task 7: Fruit

**Files:**
- Create: `src/neurogarden/engine/fruit.py`
- Test: `tests/test_fruit.py`

**Interfaces:**
- Consumes: `WorldState` (layers, `trees`, `rng`, `config`, `fruit_version`, `tick`), `Event`, `Terrain`, `Resource`.
- Produces: `add_fruit(state, x, y)` and `remove_fruit(state, x, y)` (both bump `state.fruit_version`), `fruit_near(state, tx, ty) -> int`, `fruit_candidates(state, tx, ty) -> list[(x, y)]` (row-major), `seed_initial_fruit(state)`, `update_fruit(state, events)` (spec 3.7 step 5). RNG draw order is part of the contract: a tree at its maximum draws nothing; otherwise one `randbelow(1000)`; on success with `k > 0` candidates one `randbelow(k)`.

- [ ] **Step 1: Create `tests/test_fruit.py`**

```python
from neurogarden.engine.config import Config
from neurogarden.engine.fruit import (
    add_fruit,
    fruit_candidates,
    fruit_near,
    remove_fruit,
    seed_initial_fruit,
    update_fruit,
)
from neurogarden.engine.state import new_state
from neurogarden.engine.tiles import Resource, parse_map

ORCHARD = """
#######
#.....#
#..T..#
#..N..#
#######
"""


def make_state(map_text=ORCHARD, seed=1, **overrides):
    return new_state(parse_map(map_text), Config(**overrides), seed)


def fruit_tiles(state):
    ys, xs = (state.resource_kind == Resource.FRUIT).nonzero()
    return sorted(zip(xs.tolist(), ys.tolist(), strict=True))


def test_add_and_remove_bump_the_fruit_version():
    state = make_state()
    add_fruit(state, 1, 1)
    assert state.resource_amount[1, 1] == state.config.fruit_bites
    assert state.fruit_version == 1
    remove_fruit(state, 1, 1)
    assert state.resource_kind[1, 1] == Resource.NONE
    assert state.fruit_version == 2


def test_candidates_are_ground_without_resource_in_row_major_order():
    state = make_state(tree_radius=1)
    add_fruit(state, 2, 1)
    # radius 1 around the tree at (3, 2); (3, 3) is the nest, so not ground
    assert fruit_candidates(state, 3, 2) == [(3, 1), (4, 1), (2, 2), (4, 2), (2, 3), (4, 3)]


def test_fruit_near_counts_fruit_of_any_origin():
    state = make_state(tree_radius=1)
    add_fruit(state, 2, 1)
    add_fruit(state, 5, 3)  # outside radius 1
    assert fruit_near(state, 3, 2) == 1


def test_initial_fruit_is_deterministic_per_seed():
    a, b, c = make_state(seed=3), make_state(seed=3), make_state(seed=4)
    for state in (a, b, c):
        seed_initial_fruit(state)
    assert len(fruit_tiles(a)) == a.config.tree_initial_fruit
    assert fruit_tiles(a) == fruit_tiles(b)
    assert a.rng.state == b.rng.state
    assert a.rng.state != c.rng.state


def test_fruit_rots_at_its_lifetime():
    state = make_state(fruit_lifetime=3, fruit_spawn_permille=0)
    add_fruit(state, 1, 1)
    events = []
    update_fruit(state, events)
    update_fruit(state, events)
    assert fruit_tiles(state) == [(1, 1)]
    update_fruit(state, events)
    assert fruit_tiles(state) == []
    assert [(e.type, e.data) for e in events] == [("fruit_rotted", {"x": 1, "y": 1})]


def test_tree_at_its_maximum_draws_nothing():
    state = make_state(tree_max_fruit=1, tree_initial_fruit=1, fruit_spawn_permille=1000)
    add_fruit(state, 2, 2)
    before = state.rng.state
    update_fruit(state, [])
    assert state.rng.state == before
    assert fruit_tiles(state) == [(2, 2)]


def test_spawn_always_happens_at_1000_permille_and_never_at_0():
    sure = make_state(fruit_spawn_permille=1000)
    events = []
    update_fruit(sure, events)
    assert len(fruit_tiles(sure)) == 1
    assert [e.type for e in events] == ["fruit_spawned"]

    never = make_state(fruit_spawn_permille=0)
    update_fruit(never, [])
    assert fruit_tiles(never) == []


def test_no_candidate_means_one_draw_and_no_spawn():
    state = make_state("###\n#T#\n###", fruit_spawn_permille=1000)
    before = state.rng.state
    events = []
    update_fruit(state, events)
    assert events == []
    state_after_one_draw = type(state.rng)(0)
    state_after_one_draw.state = before
    state_after_one_draw.randbelow(1000)
    assert state.rng.state == state_after_one_draw.state
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_fruit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.fruit'`

- [ ] **Step 3: Create `src/neurogarden/engine/fruit.py`**

```python
"""Fruit: placement, rot and spawning (tick order, step 5)."""

from __future__ import annotations

import numpy as np

from .events import Event
from .state import WorldState
from .tiles import Resource, Terrain


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


def _window(state: WorldState, tx: int, ty: int) -> tuple[int, int, int, int]:
    r = state.config.tree_radius
    return (
        max(0, tx - r),
        min(state.width, tx + r + 1),
        max(0, ty - r),
        min(state.height, ty + r + 1),
    )


def fruit_near(state: WorldState, tx: int, ty: int) -> int:
    """Fruits of any origin within tree_radius (Chebyshev) of the tree."""
    x0, x1, y0, y1 = _window(state, tx, ty)
    return int((state.resource_kind[y0:y1, x0:x1] == Resource.FRUIT).sum())


def fruit_candidates(state: WorldState, tx: int, ty: int) -> list[tuple[int, int]]:
    """Ground tiles without a resource near the tree, row-major. Occupied tiles count."""
    x0, x1, y0, y1 = _window(state, tx, ty)
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_fruit.py -q`
Expected: `8 passed`

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/fruit.py tests/test_fruit.py
git commit -m "feat(engine): add fruit placement, rot and spawning"
```

---

### Task 8: Actions

**Files:**
- Create: `src/neurogarden/engine/actions.py`
- Test: `tests/test_actions.py`

**Interfaces:**
- Consumes: `Action`, `DIRECTIONS`, `MOVE_DIRECTION`; `NEED_MAX`; `Event`; `remove_fruit`; `Agent`, `WorldState`; `Resource`, `Terrain`.
- Produces: `water_adjacent(state, x, y) -> bool`; `apply_action(state, agent, action: int, events) -> Action` — clears `agent.bumped`, applies the action (spec 3.6), appends events, and returns the effective action (`Action.IDLE` for an unknown id, with an `invalid_action` event). Energy is *not* touched here; Task 9 owns it.

- [ ] **Step 1: Create `tests/test_actions.py`**

```python
from neurogarden.engine.actions import apply_action, water_adjacent
from neurogarden.engine.body import Action
from neurogarden.engine.config import Config
from neurogarden.engine.fruit import add_fruit
from neurogarden.engine.state import Agent, new_state
from neurogarden.engine.tiles import Resource, parse_map

MAP = """
######
#...~#
#.N..#
#T...#
######
"""


def make(x=2, y=1, satiety=500, hydration=500, **overrides):
    state = new_state(parse_map(MAP), Config(**overrides), seed=1)
    agent = Agent(1, "fly", x, y, 2, satiety, hydration, 800, 1000)
    state.agents[1] = agent
    state.occupant[y, x] = 1
    return state, agent


def test_move_into_free_ground_updates_position_occupant_and_facing():
    state, agent = make()
    events = []
    assert apply_action(state, agent, Action.MOVE_W, events) == Action.MOVE_W
    assert (agent.x, agent.y, agent.facing) == (1, 1, 3)
    assert state.occupant[1, 2] == 0 and state.occupant[1, 1] == 1
    assert [(e.type, e.data) for e in events] == [
        ("moved", {"from": (2, 1), "to": (1, 1), "direction": 3})
    ]


def test_blocked_moves_bump_but_still_turn():
    cases = (  # x, y, action, facing afterwards; blocked by rock, water, tree, rock
        (1, 1, Action.MOVE_N, 0),
        (3, 1, Action.MOVE_E, 1),
        (1, 2, Action.MOVE_S, 2),
        (1, 1, Action.MOVE_W, 3),
    )
    for x, y, action, facing in cases:
        state, agent = make(x=x, y=y)
        events = []
        apply_action(state, agent, action, events)
        assert (agent.x, agent.y) == (x, y)
        assert agent.bumped
        assert agent.facing == facing
        assert [(e.type, e.data) for e in events] == [("bumped", {"direction": facing})]


def test_out_of_bounds_blocks_like_rock():
    state = new_state(parse_map("..\n.."), Config(), seed=1)
    agent = Agent(1, "fly", 0, 0, 2, 500, 500, 800, 1000)
    state.agents[1] = agent
    state.occupant[0, 0] = 1
    apply_action(state, agent, Action.MOVE_N, [])
    assert (agent.x, agent.y) == (0, 0) and agent.bumped


def test_occupied_tile_blocks_movement():
    state, agent = make()
    state.occupant[1, 1] = 2
    apply_action(state, agent, Action.MOVE_W, [])
    assert (agent.x, agent.y) == (2, 1) and agent.bumped


def test_bumped_is_cleared_by_the_next_action():
    state, agent = make(x=1, y=1)
    apply_action(state, agent, Action.MOVE_N, [])
    assert agent.bumped
    apply_action(state, agent, Action.IDLE, [])
    assert not agent.bumped


def test_eating_takes_one_bite_and_removes_the_last_one():
    state, agent = make(fruit_bites=2, fruit_bite_satiety=150)
    add_fruit(state, 2, 1)
    events = []
    apply_action(state, agent, Action.CONSUME, events)
    assert agent.satiety == 650 and state.resource_amount[1, 2] == 1
    apply_action(state, agent, Action.CONSUME, events)
    assert agent.satiety == 800 and state.resource_kind[1, 2] == Resource.NONE
    assert [(e.type, e.data) for e in events] == [
        ("ate", {"bites_left": 1}),
        ("ate", {"bites_left": 0}),
    ]


def test_satiety_and_hydration_are_clamped_at_1000():
    state, agent = make(x=3, y=1, satiety=950, hydration=950)
    add_fruit(state, 3, 1)
    agent.hydration = 1000
    apply_action(state, agent, Action.CONSUME, [])  # eats: satiety lower
    assert agent.satiety == 1000
    agent.hydration = 950
    apply_action(state, agent, Action.CONSUME, [])  # drinks: hydration lower
    assert agent.hydration == 1000


def test_drinking_needs_adjacent_water():
    state, agent = make(x=3, y=1, drink_hydration=150)
    assert water_adjacent(state, 3, 1)
    events = []
    apply_action(state, agent, Action.CONSUME, events)
    assert agent.hydration == 650
    assert [e.type for e in events] == ["drank"]


def test_lower_need_decides_when_both_are_possible_and_tie_eats():
    for satiety, hydration, expected in ((300, 600, "ate"), (600, 300, "drank"), (500, 500, "ate")):
        state, agent = make(x=3, y=1, satiety=satiety, hydration=hydration)
        add_fruit(state, 3, 1)
        events = []
        apply_action(state, agent, Action.CONSUME, events)
        assert [e.type for e in events] == [expected]


def test_consume_with_nothing_available_fails():
    state, agent = make()
    events = []
    apply_action(state, agent, Action.CONSUME, events)
    assert [e.type for e in events] == ["consume_failed"]
    assert (agent.satiety, agent.hydration) == (500, 500)


def test_rest_reports_whether_on_nest():
    state, agent = make(x=2, y=2)
    events = []
    apply_action(state, agent, Action.REST, events)
    assert [(e.type, e.data) for e in events] == [("rested", {"on_nest": True})]


def test_invalid_action_becomes_idle_with_an_event():
    state, agent = make()
    events = []
    assert apply_action(state, agent, 99, events) == Action.IDLE
    assert [(e.type, e.data) for e in events] == [("invalid_action", {"action": "99"})]
    assert (agent.x, agent.y) == (2, 1)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_actions.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.actions'`

- [ ] **Step 3: Create `src/neurogarden/engine/actions.py`**

```python
"""Apply one agent's action to the world (tick order, step 3)."""

from __future__ import annotations

from .body import DIRECTIONS, MOVE_DIRECTION, Action
from .config import NEED_MAX
from .events import Event
from .fruit import remove_fruit
from .state import Agent, WorldState
from .tiles import Resource, Terrain


def water_adjacent(state: WorldState, x: int, y: int) -> bool:
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if state.in_bounds(nx, ny) and state.terrain[ny, nx] == Terrain.WATER:
            return True
    return False


def apply_action(state: WorldState, agent: Agent, action: int, events: list[Event]) -> Action:
    """Apply the action and return the effective one (IDLE for an invalid id)."""
    agent.bumped = False
    try:
        act = Action(action)
    except ValueError:
        events.append(Event(state.tick, "invalid_action", agent.id, {"action": str(action)}))
        return Action.IDLE
    if act in MOVE_DIRECTION:
        _move(state, agent, MOVE_DIRECTION[act], events)
    elif act == Action.CONSUME:
        _consume(state, agent, events)
    elif act == Action.REST:
        on_nest = bool(state.terrain[agent.y, agent.x] == Terrain.NEST)
        events.append(Event(state.tick, "rested", agent.id, {"on_nest": on_nest}))
    return act


def _move(state: WorldState, agent: Agent, direction: int, events: list[Event]) -> None:
    dx, dy = DIRECTIONS[direction]
    agent.facing = direction
    tx, ty = agent.x + dx, agent.y + dy
    if state.walkable(tx, ty) and state.occupant[ty, tx] == 0:
        state.occupant[agent.y, agent.x] = 0
        state.occupant[ty, tx] = agent.id
        data = {"from": (agent.x, agent.y), "to": (tx, ty), "direction": direction}
        events.append(Event(state.tick, "moved", agent.id, data))
        agent.x, agent.y = tx, ty
    else:
        agent.bumped = True
        events.append(Event(state.tick, "bumped", agent.id, {"direction": direction}))


def _consume(state: WorldState, agent: Agent, events: list[Event]) -> None:
    cfg = state.config
    x, y = agent.x, agent.y
    can_eat = state.resource_kind[y, x] == Resource.FRUIT and state.resource_amount[y, x] > 0
    can_drink = water_adjacent(state, x, y)
    if can_eat and (not can_drink or agent.satiety <= agent.hydration):
        agent.satiety = min(NEED_MAX, agent.satiety + cfg.fruit_bite_satiety)
        state.resource_amount[y, x] -= 1
        bites_left = int(state.resource_amount[y, x])
        if bites_left == 0:
            remove_fruit(state, x, y)
        events.append(Event(state.tick, "ate", agent.id, {"bites_left": bites_left}))
    elif can_drink:
        agent.hydration = min(NEED_MAX, agent.hydration + cfg.drink_hydration)
        events.append(Event(state.tick, "drank", agent.id))
    else:
        events.append(Event(state.tick, "consume_failed", agent.id))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_actions.py -q`
Expected: `12 passed`

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/actions.py tests/test_actions.py
git commit -m "feat(engine): apply move, consume and rest actions"
```

---

### Task 9: Metabolism

**Files:**
- Create: `src/neurogarden/engine/metabolism.py`
- Test: `tests/test_metabolism.py`

**Interfaces:**
- Consumes: `Action`, `MOVE_DIRECTION`; `is_night`; `NEED_MAX`; `Event`; `Agent`, `WorldState`; `Terrain`.
- Produces: `energy_delta(state, agent, action: Action) -> int`; `metabolise(state, agent, action: Action, events) -> None` (spec 3.7 step 4: drains, energy delta, clamp, `need_depleted`, `damaged`, regeneration, `age += 1`, death with causes `starvation` / `dehydration` / `exhaustion` / `old_age`; a dead agent leaves the occupant layer).

- [ ] **Step 1: Create `tests/test_metabolism.py`**

```python
from neurogarden.engine.body import Action
from neurogarden.engine.config import Config
from neurogarden.engine.metabolism import energy_delta, metabolise
from neurogarden.engine.state import Agent, new_state
from neurogarden.engine.tiles import parse_map

MAP = """
#####
#.N.#
#####
"""


def make(x=1, tick=200, satiety=500, hydration=500, energy=500, health=1000, **overrides):
    state = new_state(parse_map(MAP), Config(**overrides), seed=1)
    state.tick = tick  # 200 = full daylight, 1000 = night
    agent = Agent(1, "fly", x, 1, 2, satiety, hydration, energy, health)
    state.agents[1] = agent
    state.occupant[1, x] = 1
    return state, agent


def test_energy_delta_per_action_by_day():
    state, agent = make()
    assert energy_delta(state, agent, Action.IDLE) == -1
    assert energy_delta(state, agent, Action.MOVE_E) == -2
    assert energy_delta(state, agent, Action.CONSUME) == -1
    assert energy_delta(state, agent, Action.REST) == 8


def test_moving_at_night_costs_double_and_nest_rest_gains_more():
    state, agent = make(x=2, tick=1000)
    assert energy_delta(state, agent, Action.MOVE_W) == -4
    assert energy_delta(state, agent, Action.REST) == 12


def test_one_tick_drains_needs_and_ages():
    state, agent = make()
    metabolise(state, agent, Action.MOVE_E, [])
    assert (agent.satiety, agent.hydration, agent.energy, agent.age) == (499, 499, 498, 1)


def test_energy_is_clamped_at_1000():
    state, agent = make(energy=998)
    metabolise(state, agent, Action.REST, [])
    assert agent.energy == 1000


def test_health_regenerates_only_above_the_threshold():
    state, agent = make(health=900, satiety=301, hydration=301, energy=302)
    metabolise(state, agent, Action.IDLE, [])  # needs become 300, 300, 301
    assert agent.health == 901
    metabolise(state, agent, Action.IDLE, [])  # satiety drops to 299
    assert agent.health == 901


def test_depleted_need_emits_once_and_damages_every_tick():
    state, agent = make(satiety=1)
    events = []
    metabolise(state, agent, Action.IDLE, events)
    metabolise(state, agent, Action.IDLE, events)
    assert agent.health == 990
    assert [e.type for e in events] == ["need_depleted", "damaged", "damaged"]
    assert events[0].data == {"need": "satiety"}
    assert events[1].data == {"amount": 5, "causes": ["starvation"]}


def test_damage_stacks_per_depleted_need():
    state, agent = make(satiety=1, hydration=1, energy=1)
    metabolise(state, agent, Action.IDLE, [])
    assert agent.health == 985


def test_death_lists_every_cause_and_frees_the_tile():
    state, agent = make(satiety=1, hydration=1, health=10)
    events = []
    metabolise(state, agent, Action.IDLE, events)
    assert not agent.alive and agent.health == 0
    assert state.occupant[1, 1] == 0
    assert events[-1].type == "died"
    assert events[-1].data == {"causes": ["starvation", "dehydration"]}


def test_old_age_kills_a_healthy_agent():
    state, agent = make(max_age=2)
    events = []
    metabolise(state, agent, Action.IDLE, events)
    assert agent.alive
    metabolise(state, agent, Action.IDLE, events)
    assert not agent.alive
    assert events[-1].data == {"causes": ["old_age"]}


def test_no_max_age_by_default():
    state, agent = make()
    agent.age = 10**9
    metabolise(state, agent, Action.IDLE, [])
    assert agent.alive
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_metabolism.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.metabolism'`

- [ ] **Step 3: Create `src/neurogarden/engine/metabolism.py`**

```python
"""Needs, health and death (tick order, step 4)."""

from __future__ import annotations

from .body import MOVE_DIRECTION, Action
from .clock import is_night
from .config import NEED_MAX
from .events import Event
from .state import Agent, WorldState
from .tiles import Terrain

# need attribute -> cause of death when it sits at zero
_CAUSES = (("satiety", "starvation"), ("hydration", "dehydration"), ("energy", "exhaustion"))


def _clamp(value: int) -> int:
    return max(0, min(NEED_MAX, value))


def energy_delta(state: WorldState, agent: Agent, action: Action) -> int:
    cfg = state.config
    if action in MOVE_DIRECTION:
        return cfg.energy_move_night if is_night(state.tick, cfg) else cfg.energy_move
    if action == Action.CONSUME:
        return cfg.energy_consume
    if action == Action.REST:
        on_nest = state.terrain[agent.y, agent.x] == Terrain.NEST
        return cfg.energy_rest_nest if on_nest else cfg.energy_rest
    return cfg.energy_idle


def metabolise(state: WorldState, agent: Agent, action: Action, events: list[Event]) -> None:
    cfg = state.config
    before = (agent.satiety, agent.hydration, agent.energy)
    agent.satiety = _clamp(agent.satiety - cfg.satiety_drain)
    agent.hydration = _clamp(agent.hydration - cfg.hydration_drain)
    agent.energy = _clamp(agent.energy + energy_delta(state, agent, action))

    depleted: list[str] = []
    for (need, cause), old in zip(_CAUSES, before, strict=True):
        if getattr(agent, need) == 0:
            depleted.append(cause)
            if old > 0:
                events.append(Event(state.tick, "need_depleted", agent.id, {"need": need}))

    if depleted:
        amount = min(agent.health, cfg.starve_damage * len(depleted))
        agent.health -= amount
        data = {"amount": amount, "causes": list(depleted)}
        events.append(Event(state.tick, "damaged", agent.id, data))
    elif min(agent.satiety, agent.hydration, agent.energy) >= cfg.regen_threshold:
        agent.health = min(NEED_MAX, agent.health + cfg.regen_amount)

    agent.age += 1
    if agent.health == 0:
        _die(state, agent, depleted, events)
    elif cfg.max_age is not None and agent.age >= cfg.max_age:
        _die(state, agent, ["old_age"], events)


def _die(state: WorldState, agent: Agent, causes: list[str], events: list[Event]) -> None:
    agent.alive = False
    state.occupant[agent.y, agent.x] = 0
    events.append(Event(state.tick, "died", agent.id, {"causes": list(causes)}))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_metabolism.py -q`
Expected: `10 passed`

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/metabolism.py tests/test_metabolism.py
git commit -m "feat(engine): add needs, health and death"
```

---

### Task 10: Senses

**Files:**
- Create: `src/neurogarden/engine/senses.py`
- Test: `tests/test_senses.py`

**Interfaces:**
- Consumes: `water_adjacent`; `DIRECTIONS`, `INT32_MAX`, `VISION_SIZE`; `light_at`; `LIGHT_MAX`, `NEED_MAX`; `Agent`, `WorldState`; `Resource`, `Terrain`, `find_tiles`.
- Produces: `SCENTS = ("fruit", "humidity", "nest")`; `scent_field(terrain, sources: list[(x, y)], scent_range: int) -> np.ndarray int16 (H, W)`; `ScentFields(state)` with `current(state) -> (fruit, humidity, nest)` (fruit recomputed only when `state.fruit_version` changed); `vision_radius(light, state) -> int`; `observe(state, scents, agent) -> dict[str, np.ndarray]` with exactly the channels, shapes and dtypes of `observation_spec("fly")`.

- [ ] **Step 1: Create `tests/test_senses.py`**

```python
import numpy as np

from neurogarden.engine.body import observation_spec
from neurogarden.engine.config import Config
from neurogarden.engine.fruit import add_fruit, remove_fruit
from neurogarden.engine.senses import ScentFields, observe, scent_field, vision_radius
from neurogarden.engine.state import Agent, new_state
from neurogarden.engine.tiles import Terrain, parse_map

MAP = """
#########
#.......#
#.#####.#
#...N..~#
#########
"""


def make(x=1, y=1, tick=200, **overrides):
    state = new_state(parse_map(MAP), Config(**overrides), seed=1)
    state.tick = tick
    agent = Agent(1, "fly", x, y, 2, 700, 600, 500, 900, age=7)
    state.agents[1] = agent
    state.occupant[y, x] = 1
    return state, agent


def test_scent_falls_off_linearly_with_path_distance():
    terrain = parse_map(".....").terrain
    field = scent_field(terrain, [(0, 0)], 4)
    assert field.dtype == np.int16
    assert field[0].tolist() == [1000, 750, 500, 250, 0]


def test_rock_blocks_scent_so_distance_is_a_path():
    terrain = parse_map(MAP).terrain
    field = scent_field(terrain, [(3, 1)], 20)
    assert field[2, 3] == 0  # rock never smells
    assert field[1, 5] == 1000 * (20 - 2) // 20
    # (3, 3) is 2 tiles from the source as the crow flies, but 6 around the wall
    assert field[3, 3] == 1000 * (20 - 6) // 20


def test_nearest_source_wins():
    terrain = parse_map(".......").terrain
    field = scent_field(terrain, [(0, 0), (6, 0)], 10)
    assert field[0].tolist() == [1000, 900, 800, 700, 800, 900, 1000]


def test_no_sources_means_an_all_zero_field():
    assert not scent_field(parse_map("...").terrain, [], 5).any()


def test_observation_matches_the_body_spec():
    state, agent = make()
    observation = observe(state, ScentFields(state), agent)
    spec = observation_spec("fly")
    assert set(observation) == set(spec)
    for name, channel in spec.items():
        assert observation[name].shape == channel.shape, name
        assert observation[name].dtype == np.dtype(channel.dtype), name
        assert observation[name].min() >= channel.low and observation[name].max() <= channel.high


def test_smell_rows_are_fruit_humidity_nest_and_columns_own_n_e_s_w():
    state, agent = make(x=6, y=3, smell_range_humidity=10, smell_range_nest=16)
    smell = observe(state, ScentFields(state), agent)["smell"]
    assert smell[0].tolist() == [0, 0, 0, 0, 0]  # no fruit anywhere
    # water at (7, 3): own tile d=1, north (6,2) is rock, east is the water itself
    assert smell[1].tolist() == [900, 0, 1000, 0, 800]
    # nest at (4, 3): own tile d=2, west (5,3) d=1, east (7,3) is water d=3
    assert smell[2].tolist() == [875, 0, 812, 0, 937]


def test_fruit_field_is_cached_until_the_fruit_changes():
    state, agent = make(x=2, y=1)
    scents = ScentFields(state)
    assert observe(state, scents, agent)["smell"][0, 0] == 0
    add_fruit(state, 1, 1)
    assert observe(state, scents, agent)["smell"][0].tolist()[:1] == [916]
    cached = scents.current(state)[0]
    assert scents.current(state)[0] is cached  # no recompute without a change
    remove_fruit(state, 1, 1)
    assert observe(state, scents, agent)["smell"][0, 0] == 0


def test_vision_radius_follows_light():
    state, _ = make()
    assert [vision_radius(light, state) for light in (0, 499, 500, 999, 1000)] == [1, 1, 2, 2, 3]


def test_vision_by_day_is_north_up_with_rock_outside_the_map():
    state, agent = make(x=1, y=1)
    add_fruit(state, 3, 1)
    state.occupant[3, 1] = 2  # another agent two tiles south
    vision = observe(state, ScentFields(state), agent)["vision"]
    assert vision[3, 3].tolist() == [Terrain.GROUND, 0, 1]  # centre: self
    assert vision[3, 5].tolist() == [Terrain.GROUND, 1, 0]  # fruit two tiles east
    assert vision[5, 3].tolist() == [Terrain.GROUND, 0, 2]  # other agent two tiles south
    assert vision[2, 3].tolist() == [Terrain.ROCK, 0, 0]  # border rock to the north
    assert vision[0, 0].tolist() == [Terrain.ROCK, 0, 0]  # outside the map, within radius
    assert vision[4, 4].tolist() == [Terrain.ROCK, 0, 0]  # inner wall


def test_vision_at_night_is_void_beyond_radius_one():
    state, agent = make(x=1, y=1, tick=1000)
    add_fruit(state, 3, 1)
    vision = observe(state, ScentFields(state), agent)["vision"]
    assert vision[3, 5].tolist() == [Terrain.VOID, 0, 0]
    assert vision[3, 4].tolist() == [Terrain.GROUND, 0, 0]
    assert vision[0, 0].tolist() == [Terrain.VOID, 0, 0]


def test_touch_body_and_env():
    state, agent = make(x=6, y=3)
    agent.bumped = True
    add_fruit(state, 6, 3)
    observation = observe(state, ScentFields(state), agent)
    assert observation["touch"].tolist() == [1, 1, 1, 0]
    assert observation["body"].tolist() == [700, 600, 500, 900, 7]
    assert observation["env"].tolist() == [1000]

    state, agent = make(x=4, y=3, tick=50)
    observation = observe(state, ScentFields(state), agent)
    assert observation["touch"].tolist() == [0, 0, 0, 1]
    assert observation["env"].tolist() == [500]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_senses.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.senses'`

- [ ] **Step 3: Create `src/neurogarden/engine/senses.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_senses.py -q`
Expected: `11 passed`

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/senses.py tests/test_senses.py
git commit -m "feat(engine): add cached scent fields and allocentric observations"
```

---

### Task 11: Snapshot, restore and state hash

**Files:**
- Create: `src/neurogarden/engine/snapshot.py`
- Test: `tests/test_snapshot.py`

**Interfaces:**
- Consumes: `get_body`; `RULES_VERSION`, `Config`; `SplitMix64`; `Agent`, `WorldState`.
- Produces: `snapshot(state) -> dict` (JSON-serialisable; layers as base64 of little-endian C-order bytes), `restore(data) -> WorldState` (`ValueError` when `rules_version` differs), `state_hash(state) -> str` (SHA-256 hex over the canonical layout of spec section 6). Derived data (`fruit_version`, `trees`, scent caches) is never stored or hashed.

- [ ] **Step 1: Create `tests/test_snapshot.py`**

```python
import json

import numpy as np
import pytest

from neurogarden.engine.config import Config
from neurogarden.engine.fruit import add_fruit
from neurogarden.engine.snapshot import restore, snapshot, state_hash
from neurogarden.engine.state import Agent, new_state
from neurogarden.engine.tiles import parse_map

MAP = """
#####
#.N~#
#T..#
#####
"""


def make():
    state = new_state(parse_map(MAP), Config(fruit_bites=3), seed=11)
    state.rng.next_u64()
    state.tick = 42
    add_fruit(state, 1, 1)
    state.resource_age[1, 1] = 300
    agent = Agent(1, "fly", 2, 1, 3, 650, 640, 630, 990, age=42, bumped=True)
    state.agents[1] = agent
    state.occupant[1, 2] = 1
    state.next_agent_id = 2
    return state


def test_snapshot_survives_json():
    data = snapshot(make())
    assert json.loads(json.dumps(data)) == data


def test_restore_reproduces_the_state_exactly():
    original = make()
    copy = restore(json.loads(json.dumps(snapshot(original))))
    assert state_hash(copy) == state_hash(original)
    assert copy.config == original.config
    assert copy.rng.state == original.rng.state
    assert copy.agents == original.agents
    assert copy.trees == original.trees
    for name in ("terrain", "resource_kind", "resource_amount", "resource_age", "occupant"):
        restored, source = getattr(copy, name), getattr(original, name)
        assert restored.dtype == source.dtype, name
        assert np.array_equal(restored, source), name
        assert restored.flags.writeable, name


def test_hash_is_64_hex_characters_and_stable():
    assert len(state_hash(make())) == 64
    assert state_hash(make()) == state_hash(make())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: setattr(s, "tick", 43),
        lambda s: s.rng.next_u64(),
        lambda s: s.resource_age.__setitem__((1, 1), 301),
        lambda s: s.resource_amount.__setitem__((1, 1), 2),
        lambda s: setattr(s.agents[1], "satiety", 649),
        lambda s: setattr(s.agents[1], "bumped", False),
        lambda s: setattr(s, "next_agent_id", 3),
    ],
)
def test_hash_covers_every_part_of_the_dynamic_state(mutate):
    state = make()
    before = state_hash(state)
    mutate(state)
    assert state_hash(state) != before


def test_derived_data_is_not_hashed():
    state = make()
    before = state_hash(state)
    state.fruit_version += 5
    assert state_hash(state) == before


def test_restore_rejects_other_rules_versions():
    data = snapshot(make())
    data["rules_version"] += 1
    with pytest.raises(ValueError):
        restore(data)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_snapshot.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.snapshot'`

- [ ] **Step 3: Create `src/neurogarden/engine/snapshot.py`**

```python
"""Snapshot, restore and the canonical state hash (the determinism contract)."""

from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import asdict

import numpy as np

from .body import get_body
from .config import RULES_VERSION, Config
from .rng import SplitMix64
from .state import Agent, WorldState

# layer name -> little-endian dtype, in canonical hash order
LAYERS = (
    ("terrain", "<u1"),
    ("resource_kind", "<u1"),
    ("resource_amount", "<i2"),
    ("resource_age", "<i4"),
    ("occupant", "<i4"),
)


def _layer_bytes(state: WorldState, name: str, dtype: str) -> bytes:
    return np.ascontiguousarray(getattr(state, name)).astype(dtype).tobytes(order="C")


def snapshot(state: WorldState) -> dict:
    """JSON-serialisable copy of the complete dynamic state."""
    return {
        "rules_version": RULES_VERSION,
        "config": state.config.to_dict(),
        "tick": state.tick,
        "rng_state": state.rng.state,
        "width": state.width,
        "height": state.height,
        "layers": {
            name: base64.b64encode(_layer_bytes(state, name, dtype)).decode("ascii")
            for name, dtype in LAYERS
        },
        "agents": [asdict(state.agents[i]) for i in sorted(state.agents)],
        "next_agent_id": state.next_agent_id,
    }


def restore(data: dict) -> WorldState:
    if data["rules_version"] != RULES_VERSION:
        raise ValueError(
            f"snapshot has rules_version {data['rules_version']}, engine is {RULES_VERSION}"
        )
    shape = (data["height"], data["width"])
    layers = {
        name: np.frombuffer(base64.b64decode(data["layers"][name]), dtype=dtype)
        .reshape(shape)
        .astype(np.dtype(dtype).newbyteorder("="))
        for name, dtype in LAYERS
    }
    rng = SplitMix64(0)
    rng.state = data["rng_state"]
    agents = {entry["id"]: Agent(**entry) for entry in data["agents"]}
    return WorldState(
        config=Config.from_dict(data["config"]),
        rng=rng,
        agents=agents,
        next_agent_id=data["next_agent_id"],
        tick=data["tick"],
        **layers,
    )


def state_hash(state: WorldState) -> str:
    digest = hashlib.sha256()
    digest.update(
        struct.pack("<IQQHH", RULES_VERSION, state.tick, state.rng.state, state.width, state.height)
    )
    for name, dtype in LAYERS:
        digest.update(_layer_bytes(state, name, dtype))
    digest.update(struct.pack("<II", state.next_agent_id, len(state.agents)))
    for agent_id in sorted(state.agents):
        a = state.agents[agent_id]
        digest.update(
            struct.pack(
                "<12q",
                a.id,
                get_body(a.body).index,
                a.x,
                a.y,
                a.facing,
                a.satiety,
                a.hydration,
                a.energy,
                a.health,
                a.age,
                int(a.alive),
                int(a.bumped),
            )
        )
    return digest.hexdigest()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_snapshot.py -q`
Expected: `12 passed`

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/snapshot.py tests/test_snapshot.py
git commit -m "feat(engine): add snapshot, restore and the canonical state hash"
```

---

### Task 12: The World facade

**Files:**
- Create: `src/neurogarden/engine/world.py`
- Modify: `src/neurogarden/engine/__init__.py` (replace whole file)
- Test: `tests/test_world.py`

**Interfaces:**
- Consumes: everything from Tasks 2–11.
- Produces: frozen `StepResult(tick, observations: dict[int, dict[str, np.ndarray]], events: list[Event], agent_events: dict[int, list[Event]])`, where `tick` is the world's tick *after* the step and events carry the tick they happened in; `World.from_map(map_text, config=None, seed=0)`, `World.restore(snapshot)`, `World.spawn(body="fly", at=None) -> int`, `World.observe(agent_id)`, `World.step(actions: dict[int, int]) -> StepResult`, `World.snapshot()`, `World.state_hash()`, properties `state`, `config`, `tick`, attribute `map_text`. Unknown agent ids and bad spawns raise `ValueError`. Package exports: `from neurogarden.engine import Action, Config, Event, MapError, Resource, RULES_VERSION, StepResult, Terrain, World, action_names, observation_spec`.

- [ ] **Step 1: Create `tests/test_world.py`**

```python
import numpy as np
import pytest

from neurogarden.engine import Action, Config, World
from neurogarden.engine.rng import SplitMix64

MAP = """
########
#..T...#
#.N...~#
#......#
########
"""

CORRIDOR = """
#####
#...#
#####
"""


def test_from_map_seeds_initial_fruit_and_keeps_the_map_text():
    world = World.from_map(MAP, seed=1)
    assert int((world.state.resource_kind == 1).sum()) == world.config.tree_initial_fruit
    assert world.map_text.splitlines()[0] == "########"
    assert world.tick == 0


def test_spawn_defaults_to_the_nest_facing_south_with_initial_needs():
    world = World.from_map(MAP)
    fly = world.spawn()
    agent = world.state.agents[fly]
    assert fly == 1
    assert (agent.x, agent.y, agent.facing) == (2, 2, 2)
    assert (agent.satiety, agent.hydration, agent.energy, agent.health) == (700, 700, 800, 1000)
    assert world.state.occupant[2, 2] == 1


def test_second_spawn_falls_back_to_the_first_free_walkable_tile():
    world = World.from_map(MAP)
    world.spawn()
    second = world.spawn()
    agent = world.state.agents[second]
    assert second == 2 and (agent.x, agent.y) == (1, 1)


def test_bad_spawns_are_value_errors():
    world = World.from_map(MAP)
    with pytest.raises(ValueError):
        world.spawn(body="dragon")
    with pytest.raises(ValueError):
        world.spawn(at=(0, 0))  # rock
    world.spawn(at=(1, 1))
    with pytest.raises(ValueError):
        world.spawn(at=(1, 1))  # occupied


def test_observe_does_not_advance_time():
    world = World.from_map(MAP)
    fly = world.spawn()
    before = world.state_hash()
    observation = world.observe(fly)
    assert observation["body"].tolist() == [700, 700, 800, 1000, 0]
    assert world.state_hash() == before


def test_step_advances_one_tick_and_returns_observations_and_events():
    world = World.from_map(MAP, Config(fruit_spawn_permille=0))
    fly = world.spawn()
    result = world.step({fly: Action.MOVE_E})
    assert result.tick == world.tick == 1
    # tick 0 is the dark end of dawn, so the move costs the night price of 4
    assert result.observations[fly]["body"].tolist() == [699, 699, 796, 1000, 1]
    assert [e.type for e in result.events] == ["moved"]
    assert result.events[0].tick == 0
    assert result.events[0].data["to"] == (3, 2)
    assert result.agent_events[fly][0].data == {"direction": 1}


def test_missing_action_means_idle():
    world = World.from_map(MAP)
    fly = world.spawn()
    world.step({})
    assert world.state.agents[fly].energy == 799


def test_unknown_agent_id_is_a_value_error():
    world = World.from_map(MAP)
    with pytest.raises(ValueError):
        world.step({7: Action.IDLE})
    with pytest.raises(ValueError):
        world.observe(7)


def test_death_gives_one_final_observation_then_silence():
    world = World.from_map(MAP, Config(initial_satiety=1, initial_health=5))
    fly = world.spawn()
    result = world.step({fly: Action.IDLE})
    assert result.observations[fly]["body"][3] == 0
    assert [e.type for e in result.agent_events[fly]] == ["need_depleted", "damaged", "died"]
    assert world.state.occupant[2, 2] == 0
    after = world.step({fly: Action.MOVE_E})  # actions of the dead are ignored
    assert after.observations == {} and after.agent_events == {}


def test_two_agents_never_share_a_tile_and_the_loser_bumps():
    outcomes = set()
    for seed in range(20):
        world = World.from_map(CORRIDOR, seed=seed)
        left, right = world.spawn(at=(1, 1)), world.spawn(at=(3, 1))
        result = world.step({left: Action.MOVE_E, right: Action.MOVE_W})
        a, b = world.state.agents[left], world.state.agents[right]
        assert (a.x, a.y) != (b.x, b.y)
        assert sorted(e.type for e in result.events) == ["bumped", "moved"]
        assert int((world.state.occupant != 0).sum()) == 2
        outcomes.add("left" if a.x == 2 else "right")
    assert outcomes == {"left", "right"}  # the seeded shuffle decides who goes first


def run(seed, steps, action_seed=99):
    world = World.from_map(MAP, seed=seed)
    fly = world.spawn()
    rng = SplitMix64(action_seed)
    for _ in range(steps):
        world.step({fly: rng.randbelow(7)})
    return world


def test_same_seed_and_actions_give_the_same_hash():
    assert run(5, 300).state_hash() == run(5, 300).state_hash()
    assert run(5, 300).state_hash() != run(6, 300).state_hash()


def test_snapshot_restore_continue_equals_an_uninterrupted_run():
    rng = SplitMix64(3)
    actions = [rng.randbelow(7) for _ in range(400)]
    straight = World.from_map(MAP, seed=8)
    fly = straight.spawn()
    paused = World.from_map(MAP, seed=8)
    paused.spawn()
    for action in actions[:150]:
        straight.step({fly: action})
        paused.step({fly: action})
    resumed = World.restore(paused.snapshot())
    for action in actions[150:]:
        a = straight.step({fly: action})
        b = resumed.step({fly: action})
        if fly in a.observations:
            for name in a.observations[fly]:
                assert np.array_equal(a.observations[fly][name], b.observations[fly][name])
    assert resumed.state_hash() == straight.state_hash()


def test_fuzz_invariants_hold_under_random_actions():
    world = World.from_map(MAP, seed=2)
    flies = [world.spawn(), world.spawn()]
    rng = SplitMix64(17)
    for _ in range(1500):
        world.step({fly: rng.randbelow(9) for fly in flies})  # includes invalid ids 7 and 8
        state = world.state
        for agent in state.agents.values():
            for need in (agent.satiety, agent.hydration, agent.energy, agent.health):
                assert 0 <= need <= 1000
            if agent.alive:
                assert state.occupant[agent.y, agent.x] == agent.id
                assert state.walkable(agent.x, agent.y)
        assert int((state.occupant != 0).sum()) == len(state.living())
        assert (state.resource_amount >= 0).all()
        assert ((state.resource_kind == 0) == (state.resource_amount == 0)).all()
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_world.py -q`
Expected: FAIL with `ImportError: cannot import name 'Action' from 'neurogarden.engine'`

- [ ] **Step 3: Create `src/neurogarden/engine/world.py`**

```python
"""World: the engine's public face. Pure: no I/O, no clock, no reward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import snapshot as _snapshot
from .actions import apply_action
from .body import Action, get_body
from .config import Config
from .events import Event, for_agent
from .fruit import seed_initial_fruit, update_fruit
from .metabolism import metabolise
from .senses import ScentFields, observe
from .state import Agent, WorldState, new_state
from .tiles import Terrain, find_tiles, parse_map

FACING_SOUTH = 2


@dataclass(frozen=True)
class StepResult:
    tick: int  # the world's tick after this step
    observations: dict[int, dict[str, np.ndarray]]
    events: list[Event]  # full events with world coordinates: spectators, storage, stats
    agent_events: dict[int, list[Event]]  # own events, coordinates stripped: brains, rewards


class World:
    def __init__(self, state: WorldState, map_text: str | None = None) -> None:
        self._state = state
        self._scents = ScentFields(state)
        self.map_text = map_text

    @classmethod
    def from_map(cls, map_text: str, config: Config | None = None, seed: int = 0) -> World:
        parsed = parse_map(map_text)
        state = new_state(parsed, config or Config(), seed)
        seed_initial_fruit(state)
        return cls(state, parsed.text)

    @classmethod
    def restore(cls, data: dict) -> World:
        return cls(_snapshot.restore(data))

    @property
    def state(self) -> WorldState:
        return self._state

    @property
    def config(self) -> Config:
        return self._state.config

    @property
    def tick(self) -> int:
        return self._state.tick

    def spawn(self, body: str = "fly", at: tuple[int, int] | None = None) -> int:
        state = self._state
        get_body(body)  # raises ValueError for an unknown body
        if at is None:
            at = self._default_spawn_tile()
        x, y = at
        if not state.walkable(x, y) or state.occupant[y, x] != 0:
            raise ValueError(f"cannot spawn at {at}: tile is not walkable or is occupied")
        cfg = state.config
        agent = Agent(
            id=state.next_agent_id,
            body=body,
            x=x,
            y=y,
            facing=FACING_SOUTH,
            satiety=cfg.initial_satiety,
            hydration=cfg.initial_hydration,
            energy=cfg.initial_energy,
            health=cfg.initial_health,
        )
        state.agents[agent.id] = agent
        state.occupant[y, x] = agent.id
        state.next_agent_id += 1
        return agent.id

    def _default_spawn_tile(self) -> tuple[int, int]:
        state = self._state
        for x, y in find_tiles(state.terrain, Terrain.NEST):
            if state.occupant[y, x] == 0:
                return x, y
        for y in range(state.height):
            for x in range(state.width):
                if state.walkable(x, y) and state.occupant[y, x] == 0:
                    return x, y
        raise ValueError("no free walkable tile to spawn on")

    def observe(self, agent_id: int) -> dict[str, np.ndarray]:
        return observe(self._state, self._scents, self._agent(agent_id))

    def step(self, actions: dict[int, int]) -> StepResult:
        state = self._state
        for agent_id in actions:
            self._agent(agent_id)
        living = state.living()
        order = [agent.id for agent in living]
        state.rng.shuffle(order)

        events: list[Event] = []
        effective: dict[int, Action] = {}
        for agent_id in order:
            action = actions.get(agent_id, Action.IDLE)
            effective[agent_id] = apply_action(state, state.agents[agent_id], action, events)
        for agent in living:
            metabolise(state, agent, effective[agent.id], events)
        update_fruit(state, events)
        state.tick += 1

        return StepResult(
            tick=state.tick,
            observations={a.id: observe(state, self._scents, a) for a in living},
            events=events,
            agent_events={
                a.id: [for_agent(e) for e in events if e.agent_id == a.id] for a in living
            },
        )

    def snapshot(self) -> dict:
        return _snapshot.snapshot(self._state)

    def state_hash(self) -> str:
        return _snapshot.state_hash(self._state)

    def _agent(self, agent_id: int) -> Agent:
        try:
            return self._state.agents[agent_id]
        except KeyError:
            raise ValueError(f"unknown agent id {agent_id!r}") from None
```

- [ ] **Step 4: Replace `src/neurogarden/engine/__init__.py`**

```python
"""The NeuroGarden engine: a pure, deterministic simulation library."""

from .body import Action, action_names, observation_spec
from .config import RULES_VERSION, Config
from .events import Event
from .tiles import MapError, Resource, Terrain
from .world import StepResult, World

__all__ = [
    "RULES_VERSION",
    "Action",
    "Config",
    "Event",
    "MapError",
    "Resource",
    "StepResult",
    "Terrain",
    "World",
    "action_names",
    "observation_spec",
]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_world.py -q`
Expected: `13 passed`. Note the comment in `test_step_advances_one_tick...`: tick 0 is the dark end of dawn, so the first move costs the night price.

- [ ] **Step 6: Run everything so far, lint and commit**

Run: `uv run pytest -q`
Expected: `116 passed`

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/world.py src/neurogarden/engine/__init__.py tests/test_world.py
git commit -m "feat(engine): add the World facade with deterministic multi-agent step"
```

---

### Task 13: The Drosoville map

**Files:**
- Create: `src/neurogarden/engine/maps/__init__.py`, `src/neurogarden/engine/maps/drosoville.txt`
- Test: `tests/test_maps.py`

**Interfaces:**
- Consumes: `parse_map`, `find_tiles`, `Terrain`, `WALKABLE`.
- Produces: `neurogarden.engine.maps.available() -> tuple[str, ...]`, `maps.load(name: str) -> str` (`ValueError` for an unknown name). The map obeys spec 3.3: 32×24, rock border, one nest, at least two ponds, at least four trees, no tree within 6 tiles (Chebyshev) of water, all walkable tiles connected.

- [ ] **Step 1: Create `tests/test_maps.py`**

```python
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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_maps.py -q`
Expected: FAIL with `ImportError: cannot import name 'maps' from 'neurogarden.engine'`

- [ ] **Step 3: Create `src/neurogarden/engine/maps/__init__.py`**

```python
"""Bundled text maps."""

from __future__ import annotations

from importlib import resources


def available() -> tuple[str, ...]:
    files = resources.files(__name__).iterdir()
    return tuple(sorted(f.name[:-4] for f in files if f.name.endswith(".txt")))


def load(name: str) -> str:
    """Text of a bundled map, e.g. load("drosoville")."""
    if name not in available():
        raise ValueError(f"unknown map {name!r}; available: {', '.join(available())}")
    return resources.files(__name__).joinpath(f"{name}.txt").read_text(encoding="utf-8")
```

- [ ] **Step 4: Create `src/neurogarden/engine/maps/drosoville.txt`**

Exactly 24 rows of 32 characters, ending with a newline. Do not reflow it: the golden replay in Task 14 depends on every tile.

```text
################################
#..............................#
#..............................#
#..~~~~.........T..............#
#..~~~~..................T.....#
#..~~~~...............#........#
#...~~................#........#
#..........#................T..#
#..........#....##.............#
#..........#...................#
#..........#........#..........#
#..........#...N....#..........#
#..........#........#..........#
#...................#..........#
#.............####..#..........#
#........##.........#..........#
#...................#....~~....#
#.......................~~~~~..#
#....T..................~~~~~..#
#.......................~~~~~..#
#.......T.....T.........~~~~~..#
#..............................#
#..............................#
################################
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_maps.py -q`
Expected: `6 passed`

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/maps tests/test_maps.py
git commit -m "feat(engine): add the Drosoville map and its loader"
```

---

### Task 14: Replays and the golden file

**Files:**
- Create: `src/neurogarden/engine/replay.py`, `tests/make_golden.py`, `tests/data/golden_replay.json` (generated)
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `World`, `Config`, `RULES_VERSION`, `maps.load`, `SplitMix64`.
- Produces: `record(map_text, seed, actions: list[list[int]], config=None, spawns=None, checkpoint_every=100) -> dict` (one action list per tick, aligned with `spawns`; default spawn `[{"body": "fly", "at": None}]`), `verify(replay: dict) -> None` (raises `ReplayMismatch` with `.tick`, or `ValueError` for another `rules_version`), `ReplayMismatch(AssertionError)`. The engine never touches files: callers read and write the JSON.

- [ ] **Step 1: Create `tests/test_replay.py`**

```python
import json
from pathlib import Path

import pytest

from neurogarden.engine.config import RULES_VERSION, Config
from neurogarden.engine.replay import ReplayMismatch, record, verify
from neurogarden.engine.rng import SplitMix64

GOLDEN = Path(__file__).parent / "data" / "golden_replay.json"

MAP = """
#######
#..T..#
#.N..~#
#######
"""


def small_replay():
    rng = SplitMix64(4)
    actions = [[rng.randbelow(7)] for _ in range(250)]
    return record(MAP, seed=3, actions=actions, config=Config(fruit_bites=2))


def test_record_stores_everything_needed_and_checkpoints():
    replay = small_replay()
    assert replay["rules_version"] == RULES_VERSION
    assert replay["map"] == MAP
    assert replay["config"]["fruit_bites"] == 2
    assert sorted(replay["checkpoints"], key=int) == ["100", "200", "250"]
    assert json.loads(json.dumps(replay)) == replay


def test_a_recorded_replay_verifies():
    verify(small_replay())


def test_a_changed_action_is_detected():
    replay = small_replay()
    # swap in REST (or IDLE for REST): the energy difference survives to the checkpoint
    replay["actions"][50] = [0 if replay["actions"][50][0] == 6 else 6]
    with pytest.raises(ReplayMismatch) as err:
        verify(replay)
    assert err.value.tick == 100


def test_a_changed_seed_is_detected():
    replay = small_replay()
    replay["seed"] += 1
    with pytest.raises(ReplayMismatch):
        verify(replay)


def test_other_rules_versions_are_rejected():
    replay = small_replay()
    replay["rules_version"] += 1
    with pytest.raises(ValueError):
        verify(replay)


def test_golden_replay_still_reproduces():
    """Fails when engine behaviour changed. If the change is intentional, bump
    RULES_VERSION and regenerate with `uv run python tests/make_golden.py`."""
    verify(json.loads(GOLDEN.read_text(encoding="utf-8")))
```

- [ ] **Step 2: Create `tests/make_golden.py`**

```python
"""Regenerate tests/data/golden_replay.json.

Run only after an intentional rule change, together with bumping RULES_VERSION:

    uv run python tests/make_golden.py
"""

import json
from pathlib import Path

from neurogarden.engine import maps
from neurogarden.engine.replay import record
from neurogarden.engine.rng import SplitMix64

GOLDEN = Path(__file__).parent / "data" / "golden_replay.json"
TICKS = 1200
AGENTS = 2


def main() -> None:
    rng = SplitMix64(2026)
    # ids 0..8: valid actions plus a few invalid ones, to pin that path down too
    actions = [[rng.randbelow(9) for _ in range(AGENTS)] for _ in range(TICKS)]
    spawns = [{"body": "fly", "at": None}, {"body": "fly", "at": [14, 11]}]
    replay = record(maps.load("drosoville"), seed=7, actions=actions, spawns=spawns)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(replay, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN} with {len(replay['checkpoints'])} checkpoints")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the tests to make sure they fail**

Run: `uv run pytest tests/test_replay.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.engine.replay'`

- [ ] **Step 4: Create `src/neurogarden/engine/replay.py`**

```python
"""Replays: seed + actions reproduce a world; checkpoints prove it.

The caller does the file I/O; the engine only builds and verifies the data.
"""

from __future__ import annotations

from .config import RULES_VERSION, Config
from .world import World


class ReplayMismatch(AssertionError):
    def __init__(self, tick: int, expected: str, actual: str) -> None:
        super().__init__(f"state hash differs at tick {tick}: expected {expected}, got {actual}")
        self.tick = tick


def _run(replay: dict, on_checkpoint) -> None:
    world = World.from_map(replay["map"], Config.from_dict(replay["config"]), replay["seed"])
    agent_ids = []
    for spawn in replay["spawns"]:
        at = tuple(spawn["at"]) if spawn["at"] is not None else None
        agent_ids.append(world.spawn(body=spawn["body"], at=at))
    every = replay["checkpoint_every"]
    for actions in replay["actions"]:
        world.step(dict(zip(agent_ids, actions, strict=True)))
        if world.tick % every == 0 or world.tick == len(replay["actions"]):
            on_checkpoint(world.tick, world.state_hash())


def record(
    map_text: str,
    seed: int,
    actions: list[list[int]],
    config: Config | None = None,
    spawns: list[dict] | None = None,
    checkpoint_every: int = 100,
) -> dict:
    """Run the actions (one list per tick, aligned with `spawns`) and return the replay."""
    replay = {
        "rules_version": RULES_VERSION,
        "config": (config or Config()).to_dict(),
        "map": map_text,
        "seed": seed,
        "spawns": spawns or [{"body": "fly", "at": None}],
        "actions": [list(tick_actions) for tick_actions in actions],
        "checkpoint_every": checkpoint_every,
        "checkpoints": {},
    }
    _run(replay, lambda tick, digest: replay["checkpoints"].__setitem__(str(tick), digest))
    return replay


def verify(replay: dict) -> None:
    """Re-run the replay; raise ReplayMismatch at the first differing checkpoint."""
    if replay["rules_version"] != RULES_VERSION:
        raise ValueError(
            f"replay has rules_version {replay['rules_version']}, engine is {RULES_VERSION}"
        )
    seen: set[str] = set()

    def check(tick: int, digest: str) -> None:
        expected = replay["checkpoints"].get(str(tick))
        if expected != digest:
            raise ReplayMismatch(tick, str(expected), digest)
        seen.add(str(tick))

    _run(replay, check)
    missing = set(replay["checkpoints"]) - seen
    if missing:
        raise ReplayMismatch(int(min(missing, key=int)), "a checkpoint", "no such tick")
```

- [ ] **Step 5: Run the tests: only the golden one may fail**

Run: `uv run pytest tests/test_replay.py -q`
Expected: `5 passed, 1 failed` — `test_golden_replay_still_reproduces` fails with `FileNotFoundError`.

- [ ] **Step 6: Generate the golden replay**

Run: `uv run python tests/make_golden.py`
Expected: `wrote .../tests/data/golden_replay.json with 12 checkpoints`

Check the last checkpoint. With the code of this plan typed exactly, tick 1200 hashes to
`63a830002d89704662e454495f6beb499af6b0066660caea7349c6c966ac9579`. A different value means some engine file differs from the plan: find the difference before committing, because this file freezes the engine's behaviour.

Run: `uv run python -c "import json; print(json.load(open('tests/data/golden_replay.json'))['checkpoints']['1200'])"`

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_replay.py -q`
Expected: `6 passed`

- [ ] **Step 8: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/engine/replay.py tests/test_replay.py tests/make_golden.py tests/data/golden_replay.json
git commit -m "feat(engine): add replays and freeze behaviour with a golden replay"
```

---

### Task 15: Rewards and episode stats

**Files:**
- Create: `src/neurogarden/dojo/rewards.py`, `src/neurogarden/dojo/stats.py`
- Test: `tests/test_rewards_stats.py`

**Interfaces:**
- Consumes: `NEED_MAX`, `Event`.
- Produces (`rewards.py`): frozen `BodyState(satiety, hydration, energy, health, age)` with `BodyState.from_observation(observation)`; `RewardFn = Callable[[BodyState, BodyState, list[Event], bool], float]` called as `(prev_body, body, agent_events, died)`; `MAX_DRIVE = 3.0`; `drive(body) -> float`; `wellbeing`, `survival`, `make_homeostatic(death_penalty=10.0)`; `REWARDS`; `resolve(name_or_callable) -> RewardFn` (`ValueError` for an unknown name).
- Produces (`stats.py`): dataclass `EpisodeStats(lifespan, days, death_causes, bites, drinks, rest_ticks, bumps, tiles_explored, mean_wellbeing)` with property `score` (= `lifespan`); `StatsTracker(agent_id, start: (x, y), day_length)` with `.stats` and `.update(events, body)` fed with *full* events; `fitness_lifespan(stats) -> float`.

- [ ] **Step 1: Create `tests/test_rewards_stats.py`**

```python
import numpy as np
import pytest

from neurogarden.dojo.rewards import (
    BodyState,
    drive,
    make_homeostatic,
    resolve,
    survival,
    wellbeing,
)
from neurogarden.dojo.stats import StatsTracker, fitness_lifespan
from neurogarden.engine.events import Event

FULL = BodyState(1000, 1000, 1000, 1000, 10)
HALF = BodyState(500, 500, 500, 1000, 10)
EMPTY = BodyState(0, 0, 0, 1, 10)


def test_body_state_reads_the_body_channel():
    observation = {"body": np.array([700, 600, 500, 900, 7], dtype=np.int32)}
    assert BodyState.from_observation(observation) == BodyState(700, 600, 500, 900, 7)


def test_drive_is_zero_when_full_and_three_when_empty_and_ignores_health():
    assert drive(FULL) == 0.0
    assert drive(EMPTY) == 3.0
    assert drive(HALF) == pytest.approx(0.75)
    assert drive(BodyState(500, 500, 500, 1, 10)) == drive(HALF)


def test_wellbeing_is_one_minus_normalised_drive_and_zero_on_death():
    assert wellbeing(FULL, FULL, [], False) == 1.0
    assert wellbeing(FULL, HALF, [], False) == pytest.approx(0.75)
    assert wellbeing(FULL, EMPTY, [], False) == 0.0
    assert wellbeing(FULL, HALF, [], True) == 0.0


def test_survival_is_one_per_living_step():
    assert survival(FULL, HALF, [], False) == 1.0
    assert survival(FULL, HALF, [], True) == 0.0


def test_homeostatic_is_drive_reduction_with_a_death_penalty():
    homeostatic = make_homeostatic(death_penalty=10.0)
    assert homeostatic(HALF, FULL, [], False) == pytest.approx(0.75)
    assert homeostatic(FULL, HALF, [], False) == pytest.approx(-0.75)
    assert homeostatic(FULL, HALF, [], True) == pytest.approx(-10.75)


def test_resolve_takes_a_name_or_a_callable():
    assert resolve("wellbeing") is wellbeing
    assert resolve("survival") is survival
    assert resolve("homeostatic")(HALF, FULL, [], False) == pytest.approx(0.75)

    def custom(prev, body, events, died):
        return 42.0

    assert resolve(custom) is custom
    with pytest.raises(ValueError):
        resolve("glory")


def test_tracker_counts_own_events_and_explored_tiles():
    tracker = StatsTracker(agent_id=1, start=(2, 2), day_length=1200)
    events = [
        Event(0, "moved", 1, {"from": (2, 2), "to": (3, 2), "direction": 1}),
        Event(0, "moved", 2, {"from": (5, 5), "to": (5, 6), "direction": 2}),  # someone else
        Event(0, "ate", 1, {"bites_left": 3}),
        Event(0, "drank", 1),
        Event(0, "rested", 1, {"on_nest": False}),
        Event(0, "bumped", 1, {"direction": 0}),
        Event(0, "fruit_spawned", None, {"x": 1, "y": 1}),
    ]
    tracker.update(events, BodyState(1000, 1000, 1000, 1000, 1))
    tracker.update([Event(1, "moved", 1, {"from": (3, 2), "to": (2, 2), "direction": 3})], FULL)
    stats = tracker.stats
    assert (stats.bites, stats.drinks, stats.rest_ticks, stats.bumps) == (1, 1, 1, 1)
    assert stats.tiles_explored == 2  # revisiting the start adds nothing
    assert stats.lifespan == stats.score == 10
    assert fitness_lifespan(stats) == 10.0


def test_tracker_records_days_death_and_mean_wellbeing():
    tracker = StatsTracker(agent_id=1, start=(0, 0), day_length=4)
    tracker.update([], BodyState(1000, 1000, 1000, 1000, 1))
    tracker.update(
        [Event(1, "died", 1, {"causes": ["starvation"]})], BodyState(500, 500, 500, 0, 2)
    )
    stats = tracker.stats
    assert stats.death_causes == ("starvation",)
    assert stats.days == 0
    assert stats.mean_wellbeing == pytest.approx((1.0 + 0.75) / 2)
    tracker.update([], BodyState(1000, 1000, 1000, 1000, 9))
    assert tracker.stats.days == 2
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_rewards_stats.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.dojo.rewards'`

- [ ] **Step 3: Create `src/neurogarden/dojo/rewards.py`**

```python
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
```

- [ ] **Step 4: Create `src/neurogarden/dojo/stats.py`**

```python
"""Episode statistics, fitness and the public score. World-side: built from full events."""

from __future__ import annotations

from dataclasses import dataclass

from neurogarden.engine.events import Event

from .rewards import MAX_DRIVE, BodyState, drive


@dataclass
class EpisodeStats:
    """Counts only, never coordinates."""

    lifespan: int = 0  # ticks lived; the public score
    days: int = 0  # full days lived
    death_causes: tuple[str, ...] = ()
    bites: int = 0
    drinks: int = 0
    rest_ticks: int = 0
    bumps: int = 0
    tiles_explored: int = 1
    mean_wellbeing: float = 0.0

    @property
    def score(self) -> int:
        return self.lifespan


class StatsTracker:
    _COUNTERS = {"ate": "bites", "drank": "drinks", "rested": "rest_ticks", "bumped": "bumps"}

    def __init__(self, agent_id: int, start: tuple[int, int], day_length: int) -> None:
        self.stats = EpisodeStats()
        self._agent_id = agent_id
        self._day_length = day_length
        self._visited = {start}
        self._wellbeing_sum = 0.0
        self._updates = 0

    def update(self, events: list[Event], body: BodyState) -> None:
        stats = self.stats
        for event in events:
            if event.agent_id != self._agent_id:
                continue
            if event.type in self._COUNTERS:
                name = self._COUNTERS[event.type]
                setattr(stats, name, getattr(stats, name) + 1)
            elif event.type == "moved":
                self._visited.add(tuple(event.data["to"]))
            elif event.type == "died":
                stats.death_causes = tuple(event.data["causes"])
        stats.lifespan = body.age
        stats.days = body.age // self._day_length
        stats.tiles_explored = len(self._visited)
        self._updates += 1
        self._wellbeing_sum += 1.0 - drive(body) / MAX_DRIVE
        stats.mean_wellbeing = self._wellbeing_sum / self._updates


def fitness_lifespan(stats: EpisodeStats) -> float:
    """Default fitness for evolution: how long the fly lived."""
    return float(stats.lifespan)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_rewards_stats.py -q`
Expected: `8 passed`

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/dojo/rewards.py src/neurogarden/dojo/stats.py tests/test_rewards_stats.py
git commit -m "feat(dojo): add learner-side rewards and episode stats"
```

---

### Task 16: Terminal renderer

**Files:**
- Create: `src/neurogarden/dojo/render_ansi.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `World` (`.state`), `day_number`, `is_night`, `light_at`, `Resource`, `Terrain`.
- Produces: `render(world: World, agent_id: int | None = None, ascii: bool = False) -> str` — one glyph per tile (occupant over fruit over terrain), then a status line `Day N | tick T | <sky> <light>`, then four need bars for `agent_id`, then `the fly has died` when it has. This is the spectator view; brains never see it.

- [ ] **Step 1: Create `tests/test_render.py`**

```python
from neurogarden.dojo.render_ansi import render
from neurogarden.engine import Config, World

MAP = """
#####
#FN~#
#T..#
#####
"""


def make(**overrides):
    world = World.from_map(MAP, Config(tree_initial_fruit=0, **overrides))
    return world, world.spawn()


def test_ascii_frame_shows_tiles_fruit_and_fly():
    world, fly = make()
    lines = render(world, fly, ascii=True).splitlines()
    assert lines[:4] == ["#####", "#f@~#", "#T..#", "#####"]
    assert lines[4] == "Day 1 | tick 0 | night 0"
    assert lines[5] == "satiety   #######---  700"
    assert lines[7] == "energy    ########--  800"
    assert lines[8] == "health    ########## 1000"


def test_emoji_frame_uses_one_glyph_per_tile():
    world, fly = make()
    assert render(world, fly).splitlines()[1] == "🪨🍎🪰🟦🪨"


def test_frame_without_an_agent_has_no_need_bars():
    world, _ = make()
    assert len(render(world, None, ascii=True).splitlines()) == 5


def test_dead_fly_is_reported():
    world, fly = make(initial_satiety=1, initial_health=5)
    world.step({})
    assert render(world, fly, ascii=True).splitlines()[-1] == "the fly has died"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_render.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.dojo.render_ansi'`

- [ ] **Step 3: Create `src/neurogarden/dojo/render_ansi.py`**

```python
"""Terminal renderer: emoji by default, plain ASCII as a fallback."""

from __future__ import annotations

from neurogarden.engine.clock import day_number, is_night, light_at
from neurogarden.engine.tiles import Resource, Terrain
from neurogarden.engine.world import World

_EMOJI = {
    Terrain.GROUND: "· ",
    Terrain.ROCK: "🪨",
    Terrain.WATER: "🟦",
    Terrain.TREE: "🌳",
    Terrain.NEST: "🏠",
    "fruit": "🍎",
    "fly": "🪰",
    "bar": ("█", "░"),
    "day": "☀️ ",
    "night": "🌙",
}
_ASCII = {
    Terrain.GROUND: ".",
    Terrain.ROCK: "#",
    Terrain.WATER: "~",
    Terrain.TREE: "T",
    Terrain.NEST: "N",
    "fruit": "f",
    "fly": "@",
    "bar": ("#", "-"),
    "day": "day",
    "night": "night",
}
_BAR_WIDTH = 10
_NEEDS = ("satiety", "hydration", "energy", "health")


def _bar(value: int, glyphs: tuple[str, str]) -> str:
    filled = (value * _BAR_WIDTH + 999) // 1000
    return glyphs[0] * filled + glyphs[1] * (_BAR_WIDTH - filled)


def render(world: World, agent_id: int | None = None, ascii: bool = False) -> str:
    """The whole map plus a HUD for one agent. Spectator view: not for brains."""
    glyphs = _ASCII if ascii else _EMOJI
    state = world.state
    rows = []
    for y in range(state.height):
        cells = []
        for x in range(state.width):
            if state.occupant[y, x] != 0:
                cells.append(glyphs["fly"])
            elif state.resource_kind[y, x] == Resource.FRUIT:
                cells.append(glyphs["fruit"])
            else:
                cells.append(glyphs[Terrain(int(state.terrain[y, x]))])
        rows.append("".join(cells))

    sky = glyphs["night"] if is_night(state.tick, state.config) else glyphs["day"]
    light = light_at(state.tick, state.config)
    rows.append(f"Day {day_number(state.tick, state.config)} | tick {state.tick} | {sky} {light}")
    agent = state.agents.get(agent_id) if agent_id is not None else None
    if agent is not None:
        for need in _NEEDS:
            value = getattr(agent, need)
            rows.append(f"{need:<10}{_bar(value, glyphs['bar'])} {value:>4}")
        if not agent.alive:
            rows.append("the fly has died")
    return "\n".join(rows)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_render.py -q`
Expected: `4 passed`

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/dojo/render_ansi.py tests/test_render.py
git commit -m "feat(dojo): add the emoji and ASCII terminal renderer"
```

---

### Task 17: Gymnasium environment and TinyObservation

**Files:**
- Create: `src/neurogarden/dojo/env.py`, `src/neurogarden/dojo/wrappers.py`
- Modify: `src/neurogarden/dojo/__init__.py` (replace whole file)
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `World`, `Config`, `maps.load`, `observation_spec`, `action_names`, `day_number`, `render`, `BodyState`, `RewardFn`, `resolve`, `StatsTracker`, `VISION_SIZE`, `LIGHT_MAX`, `NEED_MAX`.
- Produces: `NeuroGardenEnv(map="drosoville", config=None, reward="wellbeing", max_steps=6000, render_mode=None)` — `map` is a bundled name, or raw map text when it contains a newline; attributes `world`, `config`, `max_steps`, property `agent_id`; `reset(seed=s)` uses `s` itself as the world seed; `step` returns `terminated` on death and `truncated` at `max_steps`; `info = {"events", "stats", "tick", "day"}`; `render()` returns the ANSI frame in `render_mode="ansi"`. `TinyObservation(env, include_vision=False)` yields a float32 vector in `[0, 1]` of length 25 (or 564). `neurogarden.dojo` registers `NeuroGarden/Drosoville-v0` on import (without `max_episode_steps`) and exports `ENV_ID`, `NeuroGardenEnv`, `TinyObservation`, `make(**kwargs)`.

- [ ] **Step 1: Create `tests/test_env.py`**

```python
import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import neurogarden.dojo as dojo
from neurogarden.dojo import NeuroGardenEnv, TinyObservation
from neurogarden.engine import Action, Config, World, maps

TINY = """
#####
#.N~#
#####
"""


def test_registered_under_the_neurogarden_namespace():
    env = gymnasium.make("NeuroGarden/Drosoville-v0")
    assert isinstance(env.unwrapped, NeuroGardenEnv)
    assert env.spec.max_episode_steps is None  # the env truncates itself
    assert isinstance(dojo.make(max_steps=10).unwrapped, NeuroGardenEnv)


def test_passes_the_gymnasium_checker():
    env = gymnasium.make("NeuroGarden/Drosoville-v0", render_mode="ansi")
    check_env(env.unwrapped)


def test_spaces_follow_the_body_spec():
    env = NeuroGardenEnv()
    assert env.action_space.n == 7
    assert env.observation_space["vision"].shape == (7, 7, 3)
    assert env.observation_space["smell"].dtype == np.int16
    observation, info = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    assert info["tick"] == 0 and info["day"] == 1 and info["events"] == []


def test_same_seed_same_episode():
    def rollout(seed):
        env = NeuroGardenEnv()
        env.reset(seed=seed)
        env.action_space.seed(0)
        for _ in range(200):
            env.step(env.action_space.sample())
        return env.world.state_hash()

    assert rollout(3) == rollout(3)
    assert rollout(3) != rollout(4)


def test_reset_seed_is_the_world_seed():
    env = NeuroGardenEnv()
    env.reset(seed=123)
    twin = World.from_map(maps.load("drosoville"), seed=123)
    twin.spawn()
    assert env.world.state_hash() == twin.state_hash()


def test_death_terminates_and_time_limit_truncates():
    dying = NeuroGardenEnv(map=TINY, config=Config(initial_satiety=2, initial_health=5))
    dying.reset(seed=0)
    _, _, terminated, truncated, _ = dying.step(Action.IDLE)
    assert (terminated, truncated) == (False, False)
    _, reward, terminated, truncated, info = dying.step(Action.IDLE)
    assert (terminated, truncated) == (True, False)
    assert reward == 0.0
    assert info["stats"].death_causes == ("starvation",)
    assert [e.type for e in info["events"]] == ["need_depleted", "damaged", "died"]
    _, reward, terminated, _, _ = dying.step(Action.IDLE)  # finished episode: a no-op
    assert terminated and reward == 0.0

    short = NeuroGardenEnv(map=TINY, max_steps=3)
    short.reset(seed=0)
    flags = [short.step(Action.IDLE)[2:4] for _ in range(3)]
    assert flags == [(False, False), (False, False), (False, True)]


def test_reward_is_pluggable_and_events_carry_no_coordinates():
    seen = []

    def spy(prev, body, events, died):
        seen.append((prev, body, events, died))
        return 2.5

    env = NeuroGardenEnv(map=TINY, reward=spy)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(Action.MOVE_W)
    assert reward == 2.5
    prev, body, events, died = seen[0]
    assert (prev.satiety, body.satiety, died) == (700, 699, False)
    assert events[0].type == "moved" and events[0].data == {"direction": 3}
    assert info["stats"].tiles_explored == 2


def test_default_reward_is_wellbeing():
    env = NeuroGardenEnv(map=TINY)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(Action.IDLE)
    assert 0.0 < reward < 1.0


def test_ansi_render_returns_text():
    env = NeuroGardenEnv(map=TINY, render_mode="ansi")
    env.reset(seed=0)
    frame = env.render()
    assert "Day 1" in frame and "satiety" in frame
    with pytest.raises(ValueError):
        NeuroGardenEnv(render_mode="human")


def test_tiny_observation_is_a_unit_float_vector():
    env = TinyObservation(NeuroGardenEnv(map=TINY))
    observation, _ = env.reset(seed=0)
    assert observation.shape == (25,) and observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    # smell[humidity] own tile, touch.on_nest, body.satiety, env.light
    assert observation[5] == pytest.approx(0.9)
    assert observation[18] == 1.0
    assert observation[19] == pytest.approx(0.7)
    assert observation[24] == 0.0

    with_vision = TinyObservation(NeuroGardenEnv(map=TINY), include_vision=True)
    observation, _ = with_vision.reset(seed=0)
    assert observation.shape == (25 + 7 * 7 * 11,)
    assert with_vision.observation_space.contains(observation)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_env.py -q`
Expected: FAIL with `ImportError: cannot import name 'NeuroGardenEnv' from 'neurogarden.dojo'`

- [ ] **Step 3: Create `src/neurogarden/dojo/env.py`**

```python
"""Gymnasium environment: the engine in lockstep, one fly, learner-side reward."""

from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium import spaces

from neurogarden.engine import maps
from neurogarden.engine.body import action_names, observation_spec
from neurogarden.engine.clock import day_number
from neurogarden.engine.config import Config
from neurogarden.engine.world import World

from .render_ansi import render
from .rewards import BodyState, RewardFn, resolve
from .stats import StatsTracker

BODY = "fly"


class NeuroGardenEnv(gymnasium.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(
        self,
        map: str = "drosoville",
        config: Config | None = None,
        reward: str | RewardFn = "wellbeing",
        max_steps: int = 6000,
        render_mode: str | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._map_text = map if "\n" in map else maps.load(map)
        self.config = config or Config()
        self._reward = resolve(reward)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.observation_space = spaces.Dict(
            {
                name: spaces.Box(
                    low=channel.low,
                    high=channel.high,
                    shape=channel.shape,
                    dtype=np.dtype(channel.dtype).type,
                )
                for name, channel in observation_spec(BODY).items()
            }
        )
        self.action_space = spaces.Discrete(len(action_names(BODY)))
        self.world: World | None = None
        self._fly = 0
        self._steps = 0
        self._done = True
        self._observation: dict[str, np.ndarray] = {}
        self._prev_body: BodyState | None = None
        self._tracker: StatsTracker | None = None

    @property
    def agent_id(self) -> int:
        """Engine id of the fly this environment controls."""
        return self._fly

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        world_seed = seed if seed is not None else int(self.np_random.integers(0, 2**63 - 1))
        self.world = World.from_map(self._map_text, self.config, world_seed)
        self._fly = self.world.spawn(body=BODY)
        agent = self.world.state.agents[self._fly]
        self._tracker = StatsTracker(self._fly, (agent.x, agent.y), self.config.day_length)
        self._steps = 0
        self._done = False
        self._observation = self.world.observe(self._fly)
        self._prev_body = BodyState.from_observation(self._observation)
        return self._observation, self._info([])

    def step(self, action):
        if self.world is None:
            raise RuntimeError("call reset() before step()")
        if self._done:  # stepping a finished episode changes nothing
            return self._observation, 0.0, True, False, self._info([])
        result = self.world.step({self._fly: int(action)})
        self._observation = result.observations[self._fly]
        body = BodyState.from_observation(self._observation)
        died = not self.world.state.agents[self._fly].alive
        events = result.agent_events[self._fly]
        reward = float(self._reward(self._prev_body, body, events, died))
        self._tracker.update(result.events, body)
        self._prev_body = body
        self._steps += 1
        self._done = died
        truncated = not died and self._steps >= self.max_steps
        return self._observation, reward, died, truncated, self._info(events)

    def render(self):
        if self.render_mode == "ansi" and self.world is not None:
            return render(self.world, self._fly)
        return None

    def _info(self, events: list) -> dict:
        return {
            "events": events,
            "stats": self._tracker.stats,
            "tick": self.world.tick,
            "day": day_number(self.world.tick, self.config),
        }
```

- [ ] **Step 4: Create `src/neurogarden/dojo/wrappers.py`**

```python
"""Observation wrappers for small networks."""

from __future__ import annotations

import gymnasium
import numpy as np
from gymnasium import spaces

from neurogarden.engine.body import VISION_SIZE
from neurogarden.engine.config import LIGHT_MAX, NEED_MAX

DEFAULT_AGE_SCALE = 12000
_VISION_CLASSES = (6, 2, 3)  # terrain, resource, occupant classes per vision layer
_BASE_SIZE = 15 + 4 + 5 + 1


class TinyObservation(gymnasium.ObservationWrapper):
    """Flat float32 vector in [0, 1]: smell (15) + touch (4) + body (5) + env (1) = 25.

    include_vision=True appends the 7x7 view one-hot encoded (539 more values).
    """

    def __init__(self, env: gymnasium.Env, include_vision: bool = False) -> None:
        super().__init__(env)
        self._include_vision = include_vision
        self._age_scale = env.unwrapped.config.max_age or DEFAULT_AGE_SCALE
        size = _BASE_SIZE
        if include_vision:
            size += VISION_SIZE * VISION_SIZE * sum(_VISION_CLASSES)
        self.observation_space = spaces.Box(0.0, 1.0, (size,), np.float32)

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        body = observation["body"].astype(np.float32)
        body[:4] /= NEED_MAX
        body[4] = min(body[4] / self._age_scale, 1.0)
        parts = [
            observation["smell"].astype(np.float32).ravel() / NEED_MAX,
            np.minimum(observation["touch"], 1).astype(np.float32),
            body,
            observation["env"].astype(np.float32) / LIGHT_MAX,
        ]
        if self._include_vision:
            for layer, classes in enumerate(_VISION_CLASSES):
                one_hot = np.eye(classes, dtype=np.float32)[observation["vision"][:, :, layer]]
                parts.append(one_hot.ravel())
        return np.concatenate(parts)
```

- [ ] **Step 5: Replace `src/neurogarden/dojo/__init__.py`**

```python
"""The dojo: train against the engine in lockstep. Importing registers the environment."""

import gymnasium

from .env import NeuroGardenEnv
from .wrappers import TinyObservation

ENV_ID = "NeuroGarden/Drosoville-v0"

if ENV_ID not in gymnasium.registry:
    # No max_episode_steps: the env truncates itself, so Gymnasium adds no second time limit.
    gymnasium.register(id=ENV_ID, entry_point="neurogarden.dojo.env:NeuroGardenEnv")


def make(**kwargs) -> gymnasium.Env:
    """gymnasium.make for Drosoville; takes the NeuroGardenEnv constructor arguments."""
    return gymnasium.make(ENV_ID, **kwargs)


__all__ = ["ENV_ID", "NeuroGardenEnv", "TinyObservation", "make"]
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_env.py -q`
Expected: `10 passed` (takes a few seconds: `check_env` runs full episodes)

- [ ] **Step 7: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/dojo/env.py src/neurogarden/dojo/wrappers.py src/neurogarden/dojo/__init__.py tests/test_env.py
git commit -m "feat(dojo): add the Gymnasium environment and TinyObservation"
```

---

### Task 18: Brains

**Files:**
- Create: `src/neurogarden/brains/base.py`, `src/neurogarden/brains/random_brain.py`, `src/neurogarden/brains/scripted.py`
- Modify: `src/neurogarden/brains/__init__.py` (replace whole file)
- Test: `tests/test_brains.py`

**Interfaces:**
- Consumes: `NeuroGardenEnv`, `EpisodeStats`, `Action`, `DIRECTIONS`, `VISION_SIZE`, `NIGHT_LIGHT_THRESHOLD`, `SplitMix64`, `WALKABLE`, `Resource`.
- Produces: `Brain` protocol (`reset(seed: int | None = None) -> None`, `act(observation: dict[str, np.ndarray]) -> int`); `run_episode(env, brain, seed=None) -> EpisodeStats`; `RandomBrain(seed=0)`; `ScriptedBrain(hungry_below=600, thirsty_below=600, urgent_below=250, tired_below=250, rested_above=700, critical_energy=120, keep_heading_permille=800, explore_commit=25, seed=0)`; `BRAINS = {"random": RandomBrain, "scripted": ScriptedBrain}`. Brain randomness is the brain's own `SplitMix64`, never the world's.

Two additions to the spec's rule list, both needed for the brain to survive: `rested_above` gives rule 4 hysteresis (without it the fly dithers at the `tired_below` threshold), and `explore_commit` makes a fly that is stuck behind water or a tree (scent passes, legs do not) walk it off for a while before following the scent again.

- [ ] **Step 1: Create `tests/test_brains.py`**

```python
import numpy as np

from neurogarden.brains import BRAINS, RandomBrain, ScriptedBrain, run_episode
from neurogarden.dojo import NeuroGardenEnv
from neurogarden.engine import Action
from neurogarden.engine.tiles import Terrain


def observation(
    body=(700, 700, 800, 1000, 0),
    touch=(0, 0, 0, 0),
    light=1000,
    smell=None,
    blocked=(),
):
    """A hand-built observation: open ground all around unless `blocked` directions."""
    vision = np.zeros((7, 7, 3), dtype=np.uint8)
    vision[:, :, 0] = Terrain.GROUND
    vision[3, 3, 2] = 1
    for direction in blocked:
        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[direction]
        vision[3 + dy, 3 + dx, 0] = Terrain.ROCK
    return {
        "smell": np.array(smell or [[0] * 5] * 3, dtype=np.int16),
        "vision": vision,
        "touch": np.array(touch, dtype=np.uint8),
        "body": np.array(body, dtype=np.int32),
        "env": np.array([light], dtype=np.int16),
    }


def test_random_brain_is_seeded_and_covers_all_actions():
    a, b = RandomBrain(seed=1), RandomBrain(seed=1)
    picks = [a.act({}) for _ in range(200)]
    assert picks == [b.act({}) for _ in range(200)]
    assert set(picks) == set(range(7))
    a.reset()
    assert [a.act({}) for _ in range(200)] == picks


def test_rule_1_eats_when_hungry_on_fruit_and_drinks_when_thirsty_by_water():
    brain = ScriptedBrain()
    assert brain.act(observation(body=(500, 700, 800, 1000, 0), touch=(0, 1, 0, 0))) == 5
    assert brain.act(observation(body=(700, 500, 800, 1000, 0), touch=(0, 0, 1, 0))) == 5
    assert brain.act(observation(body=(700, 700, 800, 1000, 0), touch=(0, 1, 1, 0))) != 5


def test_rule_2_critical_energy_rests_in_place():
    brain = ScriptedBrain()
    assert brain.act(observation(body=(700, 700, 100, 1000, 0))) == Action.REST


def test_rule_3_urgent_need_forages_even_at_night():
    brain = ScriptedBrain()
    smell = [[500, 0, 600, 0, 0], [0] * 5, [0] * 5]  # fruit is stronger to the east
    night = observation(body=(200, 700, 800, 1000, 0), light=0, smell=smell)
    assert brain.act(night) == Action.MOVE_E


def test_rule_4_night_goes_home_and_rests_on_the_nest():
    brain = ScriptedBrain()
    smell = [[0] * 5, [0] * 5, [500, 0, 0, 0, 600]]  # nest is stronger to the west
    assert brain.act(observation(light=0, smell=smell)) == Action.MOVE_W
    assert brain.act(observation(light=0, touch=(0, 0, 0, 1))) == Action.REST
    assert brain.act(observation(light=0)) == Action.REST  # no nest scent: rest in place


def test_rule_4_tired_keeps_resting_until_rested():
    brain = ScriptedBrain()
    on_nest = (0, 0, 0, 1)
    assert brain.act(observation(body=(700, 700, 200, 1000, 0), touch=on_nest)) == Action.REST
    assert brain.act(observation(body=(700, 700, 500, 1000, 0), touch=on_nest)) == Action.REST
    assert brain.act(observation(body=(700, 700, 750, 1000, 0), touch=on_nest)) != Action.REST


def test_rule_5_serves_the_lower_need_and_only_steps_onto_walkable_tiles():
    brain = ScriptedBrain()
    smell = [[100, 900, 0, 0, 0], [100, 0, 0, 0, 900], [0] * 5]
    thirsty = observation(body=(550, 500, 800, 1000, 0), smell=smell)
    assert brain.act(thirsty) == Action.MOVE_W  # water is west
    hungry = observation(body=(500, 550, 800, 1000, 0), smell=smell)
    assert brain.act(hungry) == Action.MOVE_N  # fruit is north
    walled = observation(body=(500, 550, 800, 1000, 0), smell=smell, blocked=(0,))
    assert brain.act(walled) != Action.MOVE_N


def test_rule_6_explores_without_walking_into_walls():
    brain = ScriptedBrain(seed=3)
    boxed = observation(blocked=(0, 1, 2))
    assert all(brain.act(boxed) == Action.MOVE_W for _ in range(20))
    assert ScriptedBrain().act(observation(blocked=(0, 1, 2, 3))) == Action.IDLE


def test_run_episode_returns_stats_and_is_reproducible():
    def lifespan():
        return run_episode(NeuroGardenEnv(max_steps=300), ScriptedBrain(), seed=5)

    first, second = lifespan(), lifespan()
    assert first.lifespan == 300
    assert (first.bites, first.drinks, first.tiles_explored) == (
        second.bites,
        second.drinks,
        second.tiles_explored,
    )


def test_registry_lists_both_brains():
    assert set(BRAINS) == {"random", "scripted"}
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_brains.py -q`
Expected: FAIL with `ImportError: cannot import name 'BRAINS' from 'neurogarden.brains'`

- [ ] **Step 3: Create `src/neurogarden/brains/base.py`**

```python
"""The brain interface and a helper to run one episode."""

from __future__ import annotations

from typing import Protocol

import gymnasium
import numpy as np

from neurogarden.dojo.stats import EpisodeStats


class Brain(Protocol):
    """Decides. Receives the raw channel dict: the same payload a network client will get."""

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, observation: dict[str, np.ndarray]) -> int: ...


def run_episode(env: gymnasium.Env, brain: Brain, seed: int | None = None) -> EpisodeStats:
    """Run the brain until the fly dies or the episode is truncated."""
    observation, info = env.reset(seed=seed)
    brain.reset(seed)
    while True:
        observation, _, terminated, truncated, info = env.step(brain.act(observation))
        if terminated or truncated:
            return info["stats"]
```

- [ ] **Step 4: Create `src/neurogarden/brains/random_brain.py`**

```python
"""The baseline: uniform random actions."""

from __future__ import annotations

import numpy as np

from neurogarden.engine.body import Action
from neurogarden.engine.rng import SplitMix64


class RandomBrain:
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = SplitMix64(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = SplitMix64(self._seed if seed is None else seed)

    def act(self, observation: dict[str, np.ndarray]) -> int:
        return self._rng.randbelow(len(Action))
```

- [ ] **Step 5: Create `src/neurogarden/brains/scripted.py`**

```python
"""A hand-written survivor: the bar every learned brain has to clear."""

from __future__ import annotations

import numpy as np

from neurogarden.engine.body import DIRECTIONS, VISION_SIZE, Action
from neurogarden.engine.config import NIGHT_LIGHT_THRESHOLD
from neurogarden.engine.rng import SplitMix64
from neurogarden.engine.tiles import WALKABLE, Resource

_FRUIT, _HUMIDITY, _NEST = 0, 1, 2  # rows of the smell channel
_MOVES = (Action.MOVE_N, Action.MOVE_E, Action.MOVE_S, Action.MOVE_W)
_CENTRE = VISION_SIZE // 2


class ScriptedBrain:
    """First matching rule wins:

    1. consume when on fruit and hungry, or next to water and thirsty
    2. energy critical -> rest in place
    3. a need is urgent -> forage for it, even at night
    4. night or tired -> follow the nest scent; rest on the nest
    5. a need is below its threshold -> forage for the lower of satiety / hydration
    6. otherwise explore
    """

    def __init__(
        self,
        hungry_below: int = 600,
        thirsty_below: int = 600,
        urgent_below: int = 250,
        tired_below: int = 250,
        rested_above: int = 700,
        critical_energy: int = 120,
        keep_heading_permille: int = 800,
        explore_commit: int = 25,
        seed: int = 0,
    ) -> None:
        self.hungry_below = hungry_below
        self.thirsty_below = thirsty_below
        self.urgent_below = urgent_below
        self.tired_below = tired_below
        self.rested_above = rested_above
        self.critical_energy = critical_energy
        self.keep_heading_permille = keep_heading_permille
        self.explore_commit = explore_commit
        self._seed = seed
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self._rng = SplitMix64(self._seed if seed is None else seed)
        self._heading = self._rng.randbelow(4)
        self._resting = False
        self._explore_left = 0

    def act(self, observation: dict[str, np.ndarray]) -> int:
        satiety, hydration, energy = (int(v) for v in observation["body"][:3])
        bumped, on_resource, water_adjacent, on_nest = (int(v) for v in observation["touch"])
        night = int(observation["env"][0]) < NIGHT_LIGHT_THRESHOLD
        hungry, thirsty = satiety < self.hungry_below, hydration < self.thirsty_below

        if (on_resource == Resource.FRUIT and hungry) or (water_adjacent and thirsty):
            return Action.CONSUME
        if energy < self.critical_energy:
            self._resting = True
            return Action.REST
        if min(satiety, hydration) < self.urgent_below:
            return self._forage(observation, _FRUIT if satiety <= hydration else _HUMIDITY, bumped)

        if energy < self.tired_below:
            self._resting = True
        elif energy >= self.rested_above:
            self._resting = False
        if night or self._resting:
            if on_nest:
                return Action.REST
            step = self._follow(observation, _NEST)
            return step if step is not None else Action.REST

        if hungry or thirsty:
            return self._forage(observation, _FRUIT if satiety <= hydration else _HUMIDITY, bumped)
        return self._explore(observation, bumped)

    def _walkable(self, observation: dict[str, np.ndarray], direction: int) -> bool:
        dx, dy = DIRECTIONS[direction]
        terrain, _, occupant = observation["vision"][_CENTRE + dy, _CENTRE + dx]
        return terrain in WALKABLE and occupant == 0

    def _follow(self, observation: dict[str, np.ndarray], scent: int) -> Action | None:
        """Step to the walkable neighbour whose scent beats the own tile, if any."""
        smell = observation["smell"][scent]
        best, best_value = None, int(smell[0])
        for direction in range(4):
            value = int(smell[1 + direction])
            if value > best_value and self._walkable(observation, direction):
                best, best_value = direction, value
        if best is None:
            return None
        self._heading = best
        return _MOVES[best]

    def _forage(self, observation: dict[str, np.ndarray], scent: int, bumped: int) -> Action:
        if self._explore_left == 0:
            step = self._follow(observation, scent)
            if step is not None:
                return step
            if int(observation["smell"][scent][0]) > 0:
                # The scent leads through water or a tree: walk it off before trying again.
                self._explore_left = self.explore_commit
        return self._explore(observation, bumped)

    def _explore(self, observation: dict[str, np.ndarray], bumped: int) -> Action:
        """Persistent random walk that avoids turning straight back."""
        if self._explore_left > 0:
            self._explore_left -= 1
        options = [d for d in range(4) if self._walkable(observation, d)]
        if not options:
            return Action.IDLE
        keep = (
            self._heading in options
            and not bumped
            and self._rng.randbelow(1000) < self.keep_heading_permille
        )
        if not keep:
            onward = [d for d in options if d != (self._heading + 2) % 4] or options
            self._heading = onward[self._rng.randbelow(len(onward))]
        return _MOVES[self._heading]
```

- [ ] **Step 6: Replace `src/neurogarden/brains/__init__.py`**

```python
"""Example brains."""

from .base import Brain, run_episode
from .random_brain import RandomBrain
from .scripted import ScriptedBrain

BRAINS = {"random": RandomBrain, "scripted": ScriptedBrain}

__all__ = ["BRAINS", "Brain", "RandomBrain", "ScriptedBrain", "run_episode"]
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_brains.py -q`
Expected: `10 passed`

- [ ] **Step 8: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/brains tests/test_brains.py
git commit -m "feat(brains): add the Brain protocol, RandomBrain and ScriptedBrain"
```

---

### Task 19: Terminal viewer

**Files:**
- Create: `src/neurogarden/dojo/watch.py`
- Test: `tests/test_watch.py`

**Interfaces:**
- Consumes: `BRAINS`, `NeuroGardenEnv` (`world`, `agent_id`), `render`.
- Produces: `main(argv: list[str] | None = None) -> int` and the module entry point `python -m neurogarden.dojo.watch --brain {random,scripted} --seed N --tps 10 --max-steps 12000 [--ascii]`. `--tps 0` runs flat out. Ctrl-C ends the run and still prints the summary.

- [ ] **Step 1: Create `tests/test_watch.py`**

```python
import pytest

from neurogarden.dojo.watch import main


def test_watch_runs_a_short_episode_and_prints_a_summary(capsys):
    assert main(["--brain", "random", "--max-steps", "5", "--tps", "0", "--ascii"]) == 0
    out = capsys.readouterr().out
    assert out.count("Day 1") == 5  # one frame per step
    assert "random: lived 5 ticks (0 days) | still alive" in out


def test_watch_rejects_unknown_brains():
    with pytest.raises(SystemExit):
        main(["--brain", "oracle"])
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_watch.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'neurogarden.dojo.watch'`

- [ ] **Step 3: Create `src/neurogarden/dojo/watch.py`**

```python
"""Watch a brain live in the terminal.

uv run python -m neurogarden.dojo.watch --brain scripted
"""

from __future__ import annotations

import argparse
import sys
import time

from neurogarden.brains import BRAINS

from .env import NeuroGardenEnv
from .render_ansi import render

_HOME_AND_CLEAR = "\x1b[H\x1b[2J"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch a brain live in Drosoville.")
    parser.add_argument("--brain", choices=sorted(BRAINS), default="scripted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tps", type=float, default=10.0, help="ticks per second; 0 = flat out")
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--ascii", action="store_true", help="plain ASCII instead of emoji")
    args = parser.parse_args(argv)

    env = NeuroGardenEnv(max_steps=args.max_steps)
    brain = BRAINS[args.brain](seed=args.seed)
    observation, info = env.reset(seed=args.seed)
    brain.reset(args.seed)
    try:
        while True:
            frame = render(env.world, env.agent_id, ascii=args.ascii)
            sys.stdout.write(f"{_HOME_AND_CLEAR}{frame}\n")
            sys.stdout.flush()
            observation, _, terminated, truncated, info = env.step(brain.act(observation))
            if terminated or truncated:
                break
            if args.tps > 0:
                time.sleep(1.0 / args.tps)
    except KeyboardInterrupt:
        pass
    stats = info["stats"]
    causes = ", ".join(stats.death_causes) or "still alive"
    print(f"\n{args.brain}: lived {stats.lifespan} ticks ({stats.days} days) | {causes}")
    print(f"bites {stats.bites} | drinks {stats.drinks} | explored {stats.tiles_explored} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_watch.py -q`
Expected: `2 passed`

- [ ] **Step 5: Look at it**

Run: `uv run python -m neurogarden.dojo.watch --brain scripted --max-steps 3 --tps 0 --ascii`
Expected: three ASCII frames (24 map rows, a status line, four need bars each) and the summary `scripted: lived 3 ticks (0 days) | still alive`. Keep agent-run episodes this short: every frame lands in your context.

For the human: `uv run python -m neurogarden.dojo.watch --brain scripted` in a real terminal redraws Drosoville in place — 🪰 leaves the 🏠, walks to 🍎 and 🟦, sleeps at night. Add `--ascii` if the emoji do not line up.

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add src/neurogarden/dojo/watch.py tests/test_watch.py
git commit -m "feat(dojo): add the terminal viewer"
```

---

### Task 20: Balance guard and benchmark

**Files:**
- Test: `tests/test_balance.py`
- Modify: `docs/superpowers/specs/2026-09-20-engine-dojo-design.md` (record the result at the end of section 3.8)

**Interfaces:**
- Consumes: `NeuroGardenEnv`, `ScriptedBrain`, `RandomBrain`, `run_episode`.
- Produces: the acceptance check of spec section 9 — over seeds 0–19 with `max_steps=6000`, the `ScriptedBrain` median lifespan is ≥ 3600 ticks and ≥ 3× the `RandomBrain` median — plus a reported (not asserted) steps-per-second figure.

This task adds no production code: the prototype run of this plan passed the guard with the default `Config` and the default `ScriptedBrain` thresholds (scripted median 6000, random median 899, about 4,800 steps per second on an Apple-silicon laptop).

- [ ] **Step 1: Create `tests/test_balance.py`**

```python
"""The balance guard: Drosoville must be neither trivially survivable nor impossible."""

import statistics
import time

import pytest

from neurogarden.brains import RandomBrain, ScriptedBrain, run_episode
from neurogarden.dojo import NeuroGardenEnv

SEEDS = range(20)
MAX_STEPS = 6000


def lifespans(brain_cls):
    env = NeuroGardenEnv(max_steps=MAX_STEPS)
    return [run_episode(env, brain_cls(), seed=seed).lifespan for seed in SEEDS]


@pytest.mark.slow
def test_scripted_brain_outlives_random_brain():
    scripted = statistics.median(lifespans(ScriptedBrain))
    random = statistics.median(lifespans(RandomBrain))
    print(f"\nmedian lifespan: scripted {scripted}, random {random}")
    assert scripted >= 3600, "a sensible brain should live at least three days"
    assert scripted >= 3 * random, "a random brain should die young"


@pytest.mark.slow
def test_benchmark_steps_per_second_is_reported():
    env = NeuroGardenEnv(max_steps=MAX_STEPS)
    brain = ScriptedBrain()
    observation, _ = env.reset(seed=0)
    brain.reset(0)
    steps = 3000
    start = time.perf_counter()
    for _ in range(steps):
        observation, _, terminated, truncated, _ = env.step(brain.act(observation))
        if terminated or truncated:
            observation, _ = env.reset(seed=1)
    rate = steps / (time.perf_counter() - start)
    print(f"\n{rate:,.0f} steps per second (design target: 2,000; reported, not asserted)")
```

- [ ] **Step 2: Run the guard**

Run: `uv run pytest tests/test_balance.py -q -s`
Expected: `2 passed` in roughly half a minute, printing `median lifespan: scripted 6000.0, random 899.0` and a steps-per-second line.

If the guard fails although every earlier task passed, the engine differs from the plan (the golden hash in Task 14 would have shown it). Only if you changed rules on purpose, retune in this order and re-run after each change: `fruit_spawn_permille` (food supply), `smell_range_humidity` (how easily water is found), `ScriptedBrain.explore_commit` (escaping scent traps), then the drains. Any `Config` default change means: bump `RULES_VERSION`, regenerate the golden replay, update spec section 3.8.

- [ ] **Step 3: Record the result in the spec**

In `docs/superpowers/specs/2026-09-20-engine-dojo-design.md`, replace the sentence (wrapped over two lines in the file, last paragraph of section 3.8)

> Values are starting points; implementation tunes them until the balance guard passes and records the final values here.

with

> Balance guard result (sub-project 1): the defaults above passed unchanged — over seeds 0–19 with `max_steps=6000` the `ScriptedBrain` median lifespan is 6000 ticks and the `RandomBrain` median is 899.

(Use the medians your run printed.)

- [ ] **Step 4: Lint and commit**

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`

```bash
git add tests/test_balance.py docs/superpowers/specs/2026-09-20-engine-dojo-design.md
git commit -m "test: add the balance guard and a reported benchmark"
```

---

### Task 21: README, final verification, repository rename

**Files:**
- Modify: `README.md` (replace whole file)

**Interfaces:**
- Consumes: everything.
- Produces: the front page, a fully green run, and — only with the user's explicit yes — the GitHub repository renamed to `neurogarden`.

- [ ] **Step 1: Replace `README.md`**

````markdown
# NeuroGarden

*Small worlds. Strange minds.*

NeuroGarden is a life simulator for small brains: a persistent, 8-bit-RPG-styled
world in which an embodied agent — first a fruit fly — has to stay alive. Bring
your own brain (a script, an evolved net, an RL agent, a connectome-inspired
model, or your own keyboard), connect it to a body, and see how well it lives.

The first world is **Drosoville**.

> Honesty rule: any connectome-based brain running here is a model *inspired by*
> real wiring. Nothing in this project claims to simulate a real fly.

## Status

Sub-project 1 of 5: the simulation **engine** and the training **dojo**.
No server, no web client yet — see `docs/superpowers/specs/`.

## Quick start

```bash
uv sync
uv run python -m neurogarden.dojo.watch --brain scripted
```

Train against it like any Gymnasium environment:

```python
import gymnasium
import neurogarden.dojo  # registers the environment

env = gymnasium.make("NeuroGarden/Drosoville-v0", reward="wellbeing")
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Ideas in one minute

- **The engine has no reward.** The world has consequences (hunger, thirst,
  fatigue, damage, death), not goals. Reward, fitness and score live on the
  learner's side.
- **A fly senses only its surroundings**: smell, a 7×7 view, touch, its own
  body, and the light. It never learns its coordinates.
- **Deterministic**: same seed and same actions give the same world, tick for
  tick. A replay is a seed plus a list of actions.

## Development

```bash
uv run pytest                 # everything, including the slow balance guard
uv run pytest -m "not slow"   # fast loop
uv run ruff check .
```
````

- [ ] **Step 2: Verify everything**

Run: `uv run pytest -q`
Expected: `164 passed`

Run: `uv run pytest -q -m "not slow"`
Expected: `162 passed, 2 deselected`

Run: `uv run ruff check .`
Run: `uv run ruff format --check .`
Expected: clean.

Run: `uv run python -c "from neurogarden.brains import ScriptedBrain, run_episode; from neurogarden.dojo import NeuroGardenEnv; s = run_episode(NeuroGardenEnv(max_steps=2400), ScriptedBrain(), seed=0); print(s.lifespan, s.days, s.death_causes)"`
Expected: `2400 2 ()` — the scripted fly is alive after two days.

- [ ] **Step 3: Check the engine's purity**

Run: `grep -rnE "import random|numpy.random|np.random|import time|from neurogarden.(dojo|brains)" src/neurogarden/engine`
Expected: no output. (`grep` exits with status 1 when nothing matches; here that is success.)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: write the NeuroGarden README"
```

- [ ] **Step 5: Ask the user before renaming anything on GitHub**

Ask: "Rename the GitHub repository `vortom/miniverse` to `neurogarden` now?" Only on an explicit yes:

Run: `gh repo rename neurogarden --repo vortom/miniverse --yes`
Run: `git remote set-url origin git@github.com:vortom/neurogarden.git`
Run: `git remote -v`
Expected: both lines show `git@github.com:vortom/neurogarden.git`.

Do **not** rename the local directory from inside a session: it breaks the session's working directory and orphans Claude's per-project memory. Tell the user they can do it afterwards themselves:

```bash
mv ~/miniverse ~/neurogarden
mv ~/.claude/projects/-Users-tomas-vorel-miniverse ~/.claude/projects/-Users-tomas-vorel-neurogarden
```

- [ ] **Step 6: Hand over**

Use superpowers:finishing-a-development-branch to decide with the user how `feat/engine-dojo` (and `docs/neurogarden-design` beneath it) get merged.

---

## Spec coverage

| Spec section | Task |
|---|---|
| 2 layout, dependencies, engine API | 1, 12 |
| 3.1–3.2 space, tile layers | 4, 6 |
| 3.3 text map, Drosoville constraints | 4, 13 |
| 3.4 time, day and night | 6 |
| 3.5 body, spawn | 6, 12 |
| 3.6 actions | 5, 8 |
| 3.7 tick order | 7, 8, 9, 12 |
| 3.8 config defaults, recorded result | 3, 20 |
| 4 senses | 5, 10 |
| 5 events, agent events | 5, 12 |
| 6 determinism: PRNG, snapshot, hash, rules version, replay | 2, 11, 12, 14 |
| 7 dojo, rewards, stats | 15, 17 |
| 8 brains and viewer | 16, 18, 19 |
| 9 testing, balance guard, benchmark | every task, 20 |
| 10 error handling | 3, 4, 8, 11, 12, 13 |
| 1 step 0: repository rename | 21 (moved last so paths stay stable during the work) |
