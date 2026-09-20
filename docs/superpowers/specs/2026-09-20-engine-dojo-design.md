# Sub-project 1 — Engine + Dojo

Date: 2026-09-20
Status: design approved in brainstorm; awaiting written-spec review.
Parent: `2026-09-20-neurogarden-architecture-design.md` (decisions there are binding).

## 1. Goal

A pure, deterministic simulation library for the world of Drosoville, a
Gymnasium environment around it, two example brains, and a terminal renderer.

**Done when:** `uv run python -m neurogarden.dojo.watch --brain scripted` shows
Drosoville in the terminal with a scripted fly living through several days, all
tests are green, and the balance guard (section 9) passes.

**Out of scope:** networking, protocol schema, server, web client, on-disk
persistence beyond a JSON replay file for tests, learned brains, multi-agent
dojo API, the `egocentric` preset, hazards, predators, flight, life-cycle
stages.

**Step 0 (needs user confirmation before touching GitHub):** rename the GitHub
repository `vortom/miniverse` → `neurogarden`, rename the local directory, and
rewrite `README.md`.

## 2. Package layout

```text
pyproject.toml            Python >= 3.12, src layout, hatchling, managed with uv
src/neurogarden/
  engine/
    rng.py                SplitMix64
    config.py             Config dataclass, RULES_VERSION
    tiles.py              Terrain / Resource enums, text-map parser
    body.py               BodyConfig registry, Action enum, observation spec
    events.py             Event dataclass and event types
    clock.py              light, night and day number from the tick
    state.py              Agent, WorldState (plain integer data)
    fruit.py              fruit placement, rot, spawning
    actions.py            apply one agent's action
    metabolism.py         needs, health, death
    senses.py             smell fields, vision, touch, body, env channels
    snapshot.py           snapshot, restore, state hash
    world.py              World facade: from_map, spawn, observe, step, snapshot, restore, state_hash
    replay.py             record / verify replay data (callers do the file I/O)
    maps/                 drosoville.txt and its loader
  dojo/
    env.py                NeuroGardenEnv (gymnasium.Env), registration, make()
    rewards.py            wellbeing, survival, homeostatic
    stats.py              EpisodeStats, fitness helpers
    wrappers.py           TinyObservation
    render_ansi.py        emoji / ASCII renderer
    watch.py              CLI viewer
  brains/
    base.py               Brain protocol, run_episode()
    random_brain.py
    scripted.py
tests/
```

Runtime dependencies: `numpy`, `gymnasium`. Dev: `pytest`, `ruff`. `uv` pins
the interpreter to the newest Python for which both publish wheels.
`engine/` imports nothing from `dojo/` or `brains/` and performs no I/O except
reading the bundled map via `importlib.resources`.

### Engine API

```python
world = World.from_map(map_text, config=Config(), seed=0)  # or World.restore(snapshot)
fly = world.spawn(body="fly")                # -> agent id
obs = world.observe(fly)                     # observation without advancing time
result = world.step({fly: Action.MOVE_N})    # exactly one tick -> StepResult
result.observations[fly]; result.events; result.agent_events[fly]
world.snapshot(); world.state_hash(); world.tick
```

## 3. World model

### 3.1 Space

Tile grid, width `W`, height `H`. Coordinates `(x, y)`, origin top-left, `x`
grows east, `y` grows south. North is `y − 1`. Neighbourhood is 4-connected.
Out-of-bounds counts as blocking.

Directions are indexed `0=N, 1=E, 2=S, 3=W` everywhere.

### 3.2 Tile layers

Each tile has three independent layers. State is integer numpy arrays of shape
`(H, W)`.

| Layer | Array | Values |
|---|---|---|
| terrain | `terrain: uint8` | `VOID=0` (never in a map; vision-only), `GROUND=1`, `ROCK=2`, `WATER=3`, `TREE=4`, `NEST=5` |
| resource | `resource_kind: uint8`, `resource_amount: int16`, `resource_age: int32` | kind `NONE=0`, `FRUIT=1`; amount = bites left; age = ticks since spawn |
| occupant | `occupant: int32` | agent id, `0` = empty |

Walkable terrain: `GROUND`, `NEST`. At most one occupant per tile. A resource
may share a tile with an occupant (a fly stands on the fruit it eats).

### 3.3 Text map format

One character per tile, rows separated by newlines, all rows equal length.

| Char | Meaning |
|---|---|
| `.` | ground |
| `#` | rock |
| `~` | water |
| `T` | fruit tree |
| `N` | nest |
| `F` | ground with a full fruit at tick 0 |

Parser errors raise `MapError` with line and column.

**Drosoville** (`maps/drosoville.txt`) constraints, enforced by a test: 32×24;
rock border; exactly one nest; at least two separate ponds; at least four fruit
trees; no fruit tree within 6 tiles (Chebyshev) of any water tile, so food and
water pull in different directions; every walkable tile reachable from the nest.

### 3.4 Time, day and night

`tick` starts at 0. One day = `day_length` = 1200 ticks. With phase
`p = tick % day_length`, `light ∈ [0, 1000]`:

| Phase | Light |
|---|---|
| dawn, `0 ≤ p < 100` | `p * 10` |
| day, `100 ≤ p < 700` | `1000` |
| dusk, `700 ≤ p < 800` | `(800 − p) * 10` |
| night, `800 ≤ p < 1200` | `0` |

`is_night = light < 500`. Day number shown to people = `tick // day_length + 1`.

### 3.5 Body

Per agent, all integers: `id` (from 1), `body` (body config name), `x`, `y`,
`facing`, `satiety`, `hydration`, `energy`, `health`, `age`, `alive`, `bumped`.

Needs (`satiety`, `hydration`, `energy`) and `health` live in `[0, 1000]`.
**Health is a consequence, not a need:** needs → health → death.

`World.spawn(body="fly", at=None)`: position `at`; without `at`, the nest tile
if it is free, else the first free walkable tile in row-major order. Raises
`ValueError` if `at` is not walkable or is occupied, or if no tile is free. Initial values come from `Config`; initial
facing is `S`.

### 3.6 Actions (body `fly`, frame `allocentric`)

| Id | Name | Effect |
|---|---|---|
| 0 | `idle` | nothing |
| 1–4 | `move_n`, `move_e`, `move_s`, `move_w` | set `facing`; step if target is in bounds, walkable and unoccupied; otherwise `bumped = 1`. Energy cost is charged either way |
| 5 | `consume` | **eat** if a fruit is on the own tile: `satiety += fruit_bite_satiety`, amount − 1, fruit removed at 0. **Drink** if any 4-neighbour is water: `hydration += drink_hydration`. If both are possible, the lower of `satiety` / `hydration` decides (tie → eat). Neither possible → `consume_failed` |
| 6 | `rest` | `energy += energy_rest`, or `energy_rest_nest` when the own tile is `NEST` |

`facing` is cosmetic in this preset (sprite direction); it is kept in state for
the later `egocentric` preset.

### 3.7 Tick order

`World.step(actions: dict[int, int]) -> StepResult` advances exactly one tick:

1. **Collect.** Every living agent gets its submitted action; missing → `idle`;
   unknown action id → `idle` plus `invalid_action` event. An unknown agent id
   raises `ValueError`; actions for dead agents are ignored.
2. **Order.** Living agent ids ascending, then Fisher–Yates shuffled with the
   world PRNG: for `i` from `n − 1` down to `1`, `j = randbelow(i + 1)`, swap
   `i` and `j`. A single agent draws nothing.
3. **Act.** Apply each agent's action in that order (section 3.6). `bumped` is
   cleared first and set only by a blocked move this tick.
4. **Metabolise**, agents in ascending id:
   - `satiety −= satiety_drain`, `hydration −= hydration_drain`.
   - `energy += ` the delta for the action taken (table below).
   - Clamp needs to `[0, 1000]`. A need reaching 0 this tick emits `need_depleted`.
   - The energy delta is `energy_idle` for `idle` and invalid actions,
     `energy_move` (or `energy_move_night` when `is_night`) for any move
     attempt, `energy_consume` for any consume attempt, and the rest gain for
     `rest`.
   - For each need at 0: `health −= starve_damage` (emits `damaged` with amount
     and causes). If none is at 0 and all needs ≥ `regen_threshold`:
     `health += regen_amount`. Clamp.
   - `age += 1`.
   - `health == 0` → dies with causes = names of the needs at 0
     (`starvation`, `dehydration`, `exhaustion`). `max_age` set and
     `age ≥ max_age` → dies with cause `old_age`. A dead agent leaves the
     occupant layer.
5. **World update.** Fruit: `resource_age += 1`; at `fruit_lifetime` the fruit
   is removed (`fruit_rotted`). Then trees in row-major order. A tree with
   `tree_max_fruit` or more fruits within `tree_radius` (Chebyshev, fruits of
   any origin) draws nothing. Otherwise it draws `randbelow(1000)`; if the
   result is below `fruit_spawn_permille`, candidates are the `GROUND` tiles
   within the radius that hold no resource (occupied tiles allowed), listed in
   row-major order; with `k > 0` candidates a second draw `randbelow(k)` picks
   the tile, which receives a fruit with `fruit_bites` bites and age 0
   (`fruit_spawned`). No candidates → no second draw, no spawn.
6. `tick += 1`.
7. **Observe.** Compute observations for every agent that was alive at the
   start of the tick (an agent that died this tick gets one final observation).

At world creation, trees in row-major order each receive `tree_initial_fruit`
fruits by the same candidate-and-draw placement rule, using the world PRNG.
`F` tiles start with a fruit of `fruit_bites` bites and age 0.

### 3.8 Config defaults

All tunables live in one frozen `Config` dataclass, validated on construction
(`ValueError` on nonsense), stored in snapshots and replay headers.

| Group | Field = default |
|---|---|
| needs | `initial_satiety=700`, `initial_hydration=700`, `initial_energy=800`, `initial_health=1000`, `satiety_drain=1`, `hydration_drain=1` |
| energy delta per action | `energy_idle=-1`, `energy_move=-2`, `energy_move_night=-4`, `energy_consume=-1`, `energy_rest=8`, `energy_rest_nest=12` |
| feeding | `fruit_bites=4`, `fruit_bite_satiety=150`, `drink_hydration=150` |
| health | `starve_damage=5`, `regen_threshold=300`, `regen_amount=1`, `max_age=None` |
| time | `day_length=1200`, `dawn_end=100`, `dusk_start=700`, `night_start=800` (the phase boundaries of 3.4; ramps scale with them) |
| vision | `vision_radius_day=3`, `vision_radius_night=1` |
| smell | `smell_range_fruit=12`, `smell_range_humidity=10`, `smell_range_nest=16` |
| fruit | `tree_radius=2`, `tree_max_fruit=3`, `tree_initial_fruit=2`, `fruit_spawn_permille=8`, `fruit_lifetime=1500` |

`energy_move_night` applies when `is_night`: moving in the dark costs double,
which gives every brain (not only vision users) a reason to rest at night.
Values are starting points; implementation tunes them until the balance guard
passes and records the final values here.

## 4. Senses

Observations are a dict of named integer channels with fixed shapes. All are
centred on the agent and **north-up** (`allocentric` frame). An agent never
receives coordinates.

| Channel | Shape, dtype | Content |
|---|---|---|
| `smell` | `(3, 5)` int16, 0–1000 | rows = scents `[fruit, humidity, nest]`; columns = samples `[own tile, N, E, S, W]` |
| `vision` | `(7, 7, 3)` uint8 | `[row, col, layer]`, row 0 = northmost; layers `[terrain, resource_kind, occupant]` |
| `touch` | `(4,)` uint8 | `[bumped, resource_kind on own tile, water_adjacent, on_nest]` |
| `body` | `(5,)` int32 | `[satiety, hydration, energy, health, age]` |
| `env` | `(1,)` int16 | `[light]` |

**Smell.** Per scent, a field over the grid: multi-source BFS over 4-connected
tiles, blocked only by `ROCK`. With `d` = path distance to the nearest source
and `R` = the scent's range: intensity `= 1000 * (R − d) // R` for `d < R`,
else `0`. Sources: fruit tiles; water tiles; nest tiles. A sample on a `ROCK`
or out-of-bounds neighbour is `0`. Fields are cached and recomputed only when
their sources change (humidity and nest: once; fruit: on spawn, rot, or last
bite). Caches are derived data: never snapshotted, rebuilt on restore.

**Vision.** Radius `r = vision_radius_night + (vision_radius_day −
vision_radius_night) * light // 1000` (Chebyshev). Tiles within `r` show their
true layers; tiles beyond `r` are `[VOID, 0, 0]`; out-of-bounds tiles within
`r` show as `ROCK`. Occupant layer: `0` none, `1` self (centre), `2` other
agent. No line-of-sight occlusion in v1.

`engine.body.observation_spec(body)` and `action_names(body)` describe channels
and actions as data, so the dojo (and later the protocol handshake) are
generated from the same source.

## 5. Events

`Event(tick, type, agent_id | None, data)`. Types: `moved`, `bumped`, `ate`,
`drank`, `consume_failed`, `rested`, `invalid_action`, `need_depleted`,
`damaged`, `died`, `fruit_spawned`, `fruit_rotted`.

`StepResult` carries `tick`, `observations`, `events` (full, with world
coordinates — for spectators, storage and stats) and `agent_events[agent_id]`
(that agent's own events with coordinates stripped — what a brain or a reward
function may see). This keeps the agent stream free of world state.

## 6. Determinism

- **PRNG:** SplitMix64, implemented in `rng.py`, state = one `u64`
  (`seed mod 2^64`). `next_u64`: `state += 0x9E3779B97F4A7C15`; `z = state`;
  `z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9`; `z = (z ^ (z >> 27)) *
  0x94D049BB133111EB`; return `z ^ (z >> 31)`; all mod 2^64. `randbelow(n)`:
  rejection sampling — draw until `r < (2^64 // n) * n`, return `r % n`.
  Reference vectors (verified): seed 0 → `0xe220a8397b1dcdaf`,
  `0x6e789e6aa1b965f4`, `0x06c45d188009454f`; seed 42, `randbelow(1000)` ×5 →
  `413, 291, 858, 764, 250`. No library RNG is used inside the engine, so
  replays survive dependency upgrades and a future port can match bit for bit.
- **Brain randomness lives in the brain**, never in the world PRNG. Replays
  store actions, so brain RNG is irrelevant to reproduction.
- **Snapshot:** `World.snapshot()` returns a JSON-serialisable dict:
  `rules_version`, config, `tick`, `rng_state`, size, the five layer arrays
  (base64 of little-endian C-order bytes), agents, `next_agent_id`.
  `World.restore(snapshot)` reproduces the world exactly.
- **State hash:** `World.state_hash()` = SHA-256 over, in order:
  `u32 rules_version`, `u64 tick`, `u64 rng_state`, `u16 W`, `u16 H`, the layer
  arrays `terrain, resource_kind, resource_amount, resource_age, occupant`
  (little-endian, C order), `u32 next_agent_id`, `u32 agent_count`, then each
  agent in ascending id as little-endian `i64` fields `id, body_index, x, y,
  facing, satiety, hydration, energy, health, age, alive, bumped`.
- **`RULES_VERSION`** (int, starts at 1) is bumped by any change that alters
  state evolution or observations for identical inputs.
- **Replay** (JSON-serialisable dict): `rules_version`, config, map text, seed,
  spawns, per-tick action lists, and `{tick: state_hash}` checkpoints.
  `replay.record(...)` builds one; `replay.verify(replay)` re-runs it and
  raises `ReplayMismatch` at the first differing checkpoint. The engine stays
  free of file I/O: callers read and write the JSON.

## 7. Dojo

```python
import gymnasium, neurogarden.dojo  # import registers the env
env = gymnasium.make("NeuroGarden/Drosoville-v0", reward="wellbeing", max_steps=6000)
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(action)
```

- `NeuroGardenEnv(map="drosoville", config=None, reward="wellbeing",
  max_steps=6000, render_mode=None)`; `neurogarden.dojo.make(...)` is a thin
  helper with the same arguments.
- Single agent. `reset(seed)` builds a fresh world from the seed and spawns one
  fly. Observation space: `Dict` of `Box` spaces matching section 4. Action
  space: `Discrete(7)`.
- `terminated` = the fly died. `truncated` = `max_steps` reached (episode
  length is a learner concern; the engine default is `max_age=None`). The env
  truncates itself; registration sets no `max_episode_steps`, so Gymnasium adds
  no second time limit.
- `info`: `events` (agent events of this step), `stats` (`EpisodeStats` so
  far), `tick`, `day`.
- `render_mode="ansi"` returns the frame as a string.
- **`TinyObservation` wrapper:** float32 vector in `[0, 1]` for small nets —
  smell (15) + touch (4) + body (5, age scaled by `max_age` or 12000 and
  clipped) + env (1) = 25 values; `include_vision=True` appends one-hot vision.

### Rewards — learner side, pluggable

Signature: `(prev_body, body, agent_events, died) -> float`; pass a name or any
callable. Drive `D(b) = Σ ((1000 − n) / 1000)²` over satiety, hydration,
energy, so `D ∈ [0, 3]`. Health is excluded: it is a consequence, not a drive.

| Name | Reward per step |
|---|---|
| `wellbeing` (default) | `1 − D(body) / 3` while alive, `0` on the death step |
| `survival` | `1` while alive, `0` on the death step |
| `homeostatic` | `D(prev) − D(body)`, minus `death_penalty` (default 10.0) on the death step — drive reduction after Keramati & Gutkin |

Why `wellbeing` is the default rather than `homeostatic`: undiscounted
drive-reduction rewards telescope to `D(start) − D(end)`, and their per-step
value is mostly negative, so dying early can look attractive unless the
discount factor and death penalty are tuned. `wellbeing` is dense, bounded,
positive while alive, and aligned with lifespan out of the box.

### Stats, fitness, score

`EpisodeStats`: `lifespan`, `days`, `death_causes`, `bites`, `drinks`,
`rest_ticks`, `bumps`, `tiles_explored`, `mean_wellbeing`. Computed from full
events; exposes counts only, never coordinates. Public score = `lifespan`.
`stats.fitness_lifespan(stats)` is the default fitness; fitness functions are
plain callables over `EpisodeStats`.

## 8. Brains and viewer

```python
class Brain(Protocol):
    def reset(self, seed: int | None = None) -> None: ...
    def act(self, observation: dict[str, np.ndarray]) -> int: ...
```

Brains consume the raw channel dict — the same payload the network SDK will
deliver in sub-project 2.
`brains.base.run_episode(env, brain, seed=None) -> EpisodeStats`.

- `RandomBrain`: uniform over the 7 actions, own seeded SplitMix64.
- `ScriptedBrain`, first matching rule wins:
  1. `consume` when standing on fruit and hungry, or next to water and thirsty.
  2. Energy critical → `rest` in place.
  3. A need is urgent → forage for it, even at night.
  4. Night or tired → follow the nest scent; `rest` on the nest.
  5. A need is below its threshold → forage for the lower of satiety / hydration.
  6. Otherwise explore.

  *Forage* = step to the walkable neighbour (read from `vision`) with the
  strongest matching scent if it beats the own tile; no usable scent →
  *explore* = persistent random walk that keeps its heading with
  `keep_heading_permille` and picks a new one on a bump. Constructor defaults:
  `hungry_below=600`, `thirsty_below=600`, `urgent_below=250`,
  `tired_below=250`, `rested_above=700`, `critical_energy=120`,
  `keep_heading_permille=800`, `explore_commit=25`; tuned together with
  `Config` against the balance guard. `rested_above` gives rule 4 hysteresis
  (a tired fly keeps resting until rested); `explore_commit` makes a fly that
  is stuck behind water or a tree — scent passes, legs do not — explore for
  that many ticks before following the scent again.

`python -m neurogarden.dojo.watch --brain {random,scripted} --seed N --tps 10
--max-steps 12000 [--ascii]`: redraws the frame in place; emoji tiles (🪰 🍎 🌳 🏠 💧 🪨) with an
ASCII fallback; HUD with day, tick, ☀️/🌙 and need bars.

## 9. Testing (test-first)

- **Rules:** each clause of sections 3.6–3.7 — drains, energy deltas incl.
  night cost, eat, drink, the eat/drink tie rule, rest and nest bonus, damage,
  regeneration, every death cause, bump, fruit spawn limits, rot.
- **Senses:** smell falloff, range cut-off, rocks blocking, cache invalidation
  on fruit change; vision radius by light, VOID beyond radius, out-of-bounds as
  rock; touch flags; channel shapes and dtypes match `observation_spec`.
- **Map:** parser errors; Drosoville constraints (3.3).
- **Determinism:** SplitMix64 reference vectors; same seed + actions → same
  `state_hash`; snapshot → restore → continue equals uninterrupted run,
  including observations.
- **Golden replay:** one checked-in replay verified in CI; an intentional rule
  change bumps `RULES_VERSION` and regenerates it.
- **Multi-agent:** two agents never share a tile; a contested tile goes to the
  first in shuffled order, the other bumps.
- **Fuzz:** random actions for many ticks — needs and health stay in
  `[0, 1000]`, occupant layer consistent with agent positions, amounts ≥ 0.
- **Dojo:** `gymnasium.utils.env_checker.check_env` passes; `terminated` /
  `truncated` semantics; reward functions on hand-built bodies.
- **Balance guard:** over 20 seeds with `max_steps=6000`, `ScriptedBrain`
  median lifespan ≥ 3600 ticks and ≥ 3× the `RandomBrain` median. Catches a
  world that is trivially survivable or impossible.
- **Benchmark (reported, not asserted):** single-agent steps per second;
  design target ≥ 2,000.

## 10. Error handling

`MapError` for bad maps; `ValueError` for invalid config, unknown body, unknown
agent id, or bad spawn tile; invalid action ids never raise (they become `idle`
with an event) because action input is untrusted once a network sits in front
of the engine. `World.restore` rejects snapshots whose `rules_version` differs
from the running engine.
