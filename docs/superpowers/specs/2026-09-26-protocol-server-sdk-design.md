# Sub-project 2 — Protocol + Server + SDK

Date: 2026-09-26
Status: approved; implemented on branch feat/protocol-server-sdk (2026-09-26). Section 11 lists what the implementation added beyond this design.
Parent: `2026-09-20-neurogarden-architecture-design.md` (its section 2 decisions
are binding). Builds on sub-project 1 (`2026-09-20-engine-dojo-design.md`).

## 1. Goal

A live Drosoville that several brains inhabit at once over the network: a world
server that ticks on the wall clock, a versioned WebSocket protocol, a Python
SDK through which the existing brains run remotely unchanged, hosted "NPC"
brains so the garden is never empty, and an ASCII spectator.

**Done when:** `neurogarden serve --npc scripted:1` runs in one terminal,
`neurogarden join --brain random --owner alice` and
`neurogarden join --brain scripted --owner bob` in two more, and
`neurogarden watch` in a fourth shows all three flies living in the same world;
a fly whose brain disconnects idles and is reclaimed on reconnect; a dead fly's
owner can rejoin as Fly #2; all tests are green.

**Out of scope:** persistence and replay storage (sub-project 4), the web client
(sub-project 3), TLS, real accounts, admin controls (pause, kick, tick-rate
changes at runtime), multiple worlds per process, bodies other than `fly`,
binary encoding, automatic reconnection inside the SDK.

## 2. Package layout

```text
protocol/v1/neurogarden.schema.json     generated from the pydantic models; drift-tested
src/neurogarden/
  protocol/
    __init__.py
    messages.py       pydantic v2 models: envelope, client → server, server → client
    catalog.py        body catalog built from engine constants (what a channel means)
    schema.py         export_schema() / schema_text(): the drift-tested artifact
    mailbox.py        Mailbox: FIFO with latest-wins for observations and frames
    codec.py          observation payloads <-> numpy channel dicts, via the catalog
  server/
    __init__.py
    names.py          fly names (deterministic, cute) and moods (thought bubbles)
    roster.py         owners, lineage counters, live flies, port ↔ agent id, scores
    chronicle.py      the naturalist's log: sentences from events
    frames.py         builders for every server → client message
    ports.py          Port protocol, RemotePort (agent or spectator), LocalPort (hosted brain)
    runner.py         WorldRunner: queued joins, pending actions, tick(), run(), logs
    gateway.py        WebSocket handler: handshake, validation, dispatch, close codes
    app.py            ServerConfig, Server (async context manager), serve(), NPC hatching
  sdk/
    __init__.py
    client.py         AsyncClient (asyncio), Observation, Spectacle
    session.py        Session / Fly: sync facade on a background loop; run_brain()
  dojo/render_ansi.py   refactor: View + render_view(); render(world, …) builds a View
  cli.py              `neurogarden` entry point: serve / join / watch / schema
tests/
```

New base dependencies: `websockets>=14` and `pydantic>=2.7`. No extras yet
(nothing heavy). `[project.scripts] neurogarden = "neurogarden.cli:main"`.

Engine change (the one rule change of this sub-project): default spawns hatch
on the free walkable tile nearest the nest instead of the first free tile in
row-major order, so a full nest no longer sends newcomers to a corner.
`RULES_VERSION` becomes 2 and the golden replay is regenerated.

## 3. Protocol v1

### 3.1 Envelope

Every WebSocket frame is a JSON text frame `{"v": 1, "type": "<name>",
"payload": {...}}`. Receivers parse the envelope first; a known `type` is
validated against its pydantic model; an unknown `type` is ignored (logged at
debug) so the protocol grows by addition. `v != 1` is a fatal `error`
(`unsupported_version`, close 4001).

Validation failures of a known type are fatal on the server (`malformed`,
close 4000) — the server never guesses what a client meant — and non-fatal on
the client (the SDK raises `ProtocolError`).

### 3.2 Roles and lifecycle

A connection is an **agent** (controls at most one fly at a time) or a
**spectator**, chosen by `hello.role`. `hello` must be the first frame and must
arrive within `hello_timeout` (5 s), else close 4003 (`hello_required`).

```text
agent:      hello → welcome → [join → joined → observation* → died]* → (leave | close)
spectator:  hello → welcome → world → frame*
```

### 3.3 Messages: client → server

| `type` | payload | notes |
|---|---|---|
| `hello` | `protocol: 1`, `token: str`, `owner: str` (1–64 chars, `[A-Za-z0-9_.-]`), `role: "agent" \| "spectator"`, `client: str` (free text, ≤ 128) | wrong token → `unauthorized`, close 4002 |
| `join` | `body: "fly"` | agent role only. Queued; applied at the start of the next tick. If the owner already has a live fly, the connection is attached to it (`reattached: true`) instead of spawning |
| `action` | `tick: int`, `action: int` | accepted only if `tick` equals the tick of the latest observation sent to this fly and the next tick has not fired; latest wins within a tick; anything else is dropped silently |
| `leave` | — | detaches the connection from its fly; the fly stays in the world and idles; the connection may `join` again (reattach) |

### 3.4 Messages: server → client

| `type` | to | payload |
|---|---|---|
| `welcome` | both | `protocol: 1`, `owner`, `role`, `world: {name, width, height, tps, day_length, rules_version}`, `catalog` (3.6) |
| `joined` | agent | `agent_id`, `lineage`, `tick`, `reattached: bool` |
| `observation` | agent | `tick`, `deadline_ms`, `channels: {smell, vision, touch, body, env}` (nested lists of ints in the catalog's shapes), `events: [{type, data}]` (own events, coordinates stripped), `missed` (ticks so far in which the server played `idle` for this fly because no action arrived) |
| `died` | agent | `agent_id`, `lineage`, `tick`, `causes`, `stats` (EpisodeStats fields) |
| `world` | spectator | once after `welcome`: `map_name`, `width`, `height`, `terrain: list[list[int]]` |
| `frame` | spectator | every tick: `tick`, `day`, `light`, `resources: [[x, y, kind, amount], …]`, `agents: [{agent_id, owner, lineage, x, y, facing, satiety, hydration, energy, health, age, alive, connected}]`, `events: [{type, agent_id, data}]` (full events, with coordinates) |
| `error` | both | `code`, `message`, `fatal` |

Close codes: 1001 server shutdown; 4000 `malformed`; 4001
`unsupported_version`; 4002 `unauthorized`; 4003 `hello_required`; 4004
`superseded` (another connection took over this owner).

### 3.5 Tick contract

After the step of tick `t` the server sends `observation(tick = t)` to every
attached fly that was alive at the start of the tick; the observation
describes the world after the step (`t + 1`), its events carry tick `t` — the
engine's convention. `deadline_ms` is the time left until the next tick fires
when the message is built. An action that has not arrived by then means the
server plays `idle` and increments `missed`.

A fly's first observation is the one produced by the tick in which it was
spawned (it necessarily idles during that step).

### 3.6 Body catalog

`welcome.catalog.bodies["fly"]` says what every number means, so a remote brain
needs nothing from the engine package:

```text
frame: "allocentric"
actions: ["idle", "move_n", "move_e", "move_s", "move_w", "consume", "rest"]
channels:
  smell:  {shape: [3, 5], dtype: "int16", low: 0, high: 1000,
           axes: {rows: ["fruit", "humidity", "nest"], cols: ["own", "n", "e", "s", "w"]}}
  vision: {shape: [7, 7, 3], dtype: "uint8", low: 0, high: 255,
           axes: {layers: ["terrain", "resource", "occupant"]},
           tables: {terrain: {void: 0, ground: 1, rock: 2, water: 3, tree: 4, nest: 5},
                    resource: {none: 0, fruit: 1}, occupant: {none: 0, self: 1, other: 2}},
           centre: [3, 3], north_up: true}
  touch:  {shape: [4], dtype: "uint8", low: 0, high: 255,
           axes: {fields: ["bumped", "resource_kind", "water_adjacent", "on_nest"]}}
  body:   {shape: [5], dtype: "int32", low: 0, high: 2147483647,
           axes: {fields: ["satiety", "hydration", "energy", "health", "age"]}}
  env:    {shape: [1], dtype: "int16", low: 0, high: 1000, axes: {fields: ["light"]}}
constants: {need_max: 1000, light_max: 1000, night_light_threshold: 500}
```

Built by `protocol.catalog.build_catalog()` from `engine.body`, `engine.senses`,
`engine.tiles`, `engine.config` constants — the same source the engine uses, so
the two cannot drift.

### 3.7 Schema artifact

`protocol/v1/neurogarden.schema.json` is `protocol.schema.export_schema()`
(JSON Schema 2020-12 with `$defs` for every message and two discriminated
unions, `ClientMessage` and `ServerMessage`). A test regenerates it and fails
on any difference; `neurogarden schema` prints it. Sub-project 3 generates
TypeScript types from this file.

## 4. Server

### 4.1 Process and CLI

```text
neurogarden serve [--map drosoville] [--seed 7] [--tps 5.0] [--host 127.0.0.1]
                  [--port 8765] [--token dev] [--npc scripted:2] [--npc random:1]
                  [--hello-timeout 5]
```

One process = one world, built fresh from map + seed; it ticks whether or not
anyone is connected. `--npc KIND:N` keeps `N` hosted flies of brain `KIND`
alive (owners `npc-<kind>-<i>`); they rejoin automatically after death.
`/healthz` on the same port answers `200 OK` over plain HTTP
(`process_request`). SIGINT/SIGTERM: stop ticking, close every connection with
1001, exit 0. Binding a non-loopback host with the default token `dev` is
refused at start-up.

### 4.2 WorldRunner

State: the `World`, `tps`, the `Roster`, spectators, `queued_joins`,
`pending: {agent_id: (tick, action)}`, `last_sent_tick: {agent_id: int}`,
`missed: {agent_id: int}`, and two in-memory logs that sub-project 4 will
persist: `spawn_log: [(tick, agent_id, owner, lineage)]` and
`action_log: [ {agent_id: action} per tick ]` (the actions actually applied,
`idle` included).

`tick()` — synchronous, the only place the world mutates:

1. Apply queued joins in arrival order: reattach if the owner has a live fly,
   else `world.spawn(body)`, record `spawn_log`, roster update, send `joined`.
2. Collect actions: for every attached, living fly take `pending[agent]` if its
   tick equals `last_sent_tick[agent]`; otherwise `idle` and `missed += 1`.
   Unattached living flies (brain gone) and flies that have not received an
   observation yet (spawned this tick) get `idle` without counting.
3. `result = world.step(actions)`; append `actions` to `action_log`.
4. For every `died` event: send the final `observation`, then `died` (with the
   fly's `EpisodeStats`, tracked per fly with the dojo's `StatsTracker`), free
   the owner's live-fly slot, forget pending/missed state. The dead agent stays
   in the engine state (known gap, sub-project 4).
5. Send `observation(tick)` to every other attached living fly; record
   `last_sent_tick`.
6. Build one `frame` and hand it to every spectator.

`run()` — asyncio task: `next = start + n / tps`, `await asyncio.sleep(next -
now)`, `tick()`, repeat; if a tick overruns, the next fires immediately
(no catch-up bursts: `n` is re-based). Everything the runner touches runs on
the event loop thread; ports are the only asynchronous edge.

### 4.3 Roster

`Roster` keeps `owners: {owner: OwnerState(lineage_counter, live_agent_id |
None, port | None)}`, `agents: {agent_id: owner}`. Rules: one live fly per
owner; `join` while a live fly exists → reattach; a new `hello` for an owner
whose port is still attached **supersedes** it (old port closed 4004, new port
takes the fly, `missed` continues); `leave`/disconnect → `port = None`, fly
stays; death → `live_agent_id = None`, lineage counter kept.

### 4.4 Ports

```python
class AgentPort(Protocol):
    owner: str
    def deliver(self, message: ServerMessage) -> None   # never blocks; latest-wins for observations
    def close(self, code: int, reason: str) -> None
```

- `RemoteAgentPort(connection)`: `deliver` puts into an `Outbox` — a
  one-slot mailbox per message class (`observation`/`frame` replace an
  undelivered predecessor; `joined`, `died`, `error` queue in order) drained by
  a sender task; a slow brain gets the newest observation, never a backlog.
- `LocalAgentPort(brain, runner)`: `deliver(observation)` calls
  `brain.act(decoded_channels)` and `runner.submit_action(...)` immediately;
  `died` triggers `runner.request_join` again (NPC auto-rejoin). Hosted brains
  are ordinary clients that happen to live in-process (architecture §2.2).
- `SpectatorPort(connection)`: same outbox, `frame` latest-wins.

### 4.5 Gateway

`websockets.asyncio.server.serve(handler, host, port, process_request=health)`.
Per connection: await `hello` within `hello_timeout` (else 4003); check `v`,
`protocol` (4001), token (constant-time compare, 4002), owner syntax (4000);
reply `welcome`; agent role → register the port with the roster (supersede if
needed) and loop on `join`/`action`/`leave`; spectator role → send `world`,
register as spectator. Any invalid frame → `error(fatal)` + 4000. On close:
detach from the roster / spectators. All handler → runner calls are plain
method calls on the loop thread (`request_join`, `submit_action`, `detach`,
`add_spectator`, `remove_spectator`).

### 4.6 Logging

Stdlib `logging`, logger `neurogarden.server`: INFO for connections, joins,
deaths, supersedes; DEBUG for per-tick summaries (`tick`, agents, missed).

## 5. SDK

### 5.1 `AsyncClient` (asyncio)

```python
async with AsyncClient(url, owner="alice", token="dev", client="my-brain/0.1") as client:
    welcome = client.welcome  # WelcomePayload
    fly = await client.join()  # -> JoinedPayload
    async for observation in client.observations():  # dict[str, np.ndarray] + .tick/.events/.missed
        await client.act(observation.tick, brain.act(observation.channels))
    print(client.last_died)  # DiedPayload after the loop ends
```

`observations()` ends when `died` arrives; connection loss raises
`ConnectionLost`; a server `error` raises `ServerError(code, message)`.
Observations are decoded by `sdk.codec.decode_channels(payload, catalog)` into
the **same dict of numpy arrays the dojo produces** (shape and dtype from the
catalog), wrapped in an `Observation` dataclass with `tick`, `deadline_ms`,
`events`, `missed`, `channels`.

### 5.2 `Session` / `Fly` (sync facade)

A background thread runs an event loop with the `AsyncClient`;
`Session(url, owner, token)` is a context manager; `session.join() -> Fly`;
`fly.observations()` is a blocking iterator backed by a one-slot latest-wins
mailbox; `fly.act(action)`; `fly.stats` / `fly.died` after the iterator ends;
`session.leave()`.

```python
def run_brain(url, brain, *, owner, token="dev", rejoin=True, max_lives=None) -> list[EpisodeStats]
```

joins, feeds the brain per observation, and on `died` rejoins (Fly #n+1) until
`max_lives` or connection loss. This is what `neurogarden join` runs.

## 6. CLIs

| Command | Does |
|---|---|
| `neurogarden serve …` | section 4.1 |
| `neurogarden join --brain {random,scripted} --owner NAME [--url ws://127.0.0.1:8765] [--token dev] [--seed N] [--lives N]` | `run_brain`; prints one line per life (lineage, lifespan, causes) |
| `neurogarden watch [--url …] [--token dev] [--owner watcher] [--ascii]` | spectator: redraws the frame in place with a roster HUD (owner, lineage, need bars, connected) |
| `neurogarden schema` | prints the protocol JSON Schema |

`python -m neurogarden.dojo.watch` keeps working (local dojo, no server).

Renderer refactor: `render_ansi.View(terrain, resources, agents, tick, day,
light)` with `render_view(view, focus=None, ascii=False)`; `render(world,
agent_id, ascii)` builds a `View` from a `World` (dojo), the spectator builds
one from `world` + `frame`.

## 7. Determinism and the engine boundary

The server adds no randomness and mutates the world only inside `tick()`.
`spawn_log` + `action_log` + map + seed reproduce the server world exactly: a
test replays them through `World` alone and compares `state_hash`. This is
also the shape sub-project 4's storage will persist (tick-stamped spawns, the
known gap of the engine's tick-0 replays).

Every `action` reaching `World.step` is an `int` that passed pydantic
validation (`ge=0`); the engine still treats unknown ids as `idle` +
`invalid_action`.

## 8. Error handling

| Situation | Behaviour |
|---|---|
| frame not JSON / not an envelope / bad payload | `error(malformed, fatal)`, close 4000 |
| `v` or `hello.protocol` ≠ 1 | `error(unsupported_version, fatal)`, close 4001 |
| wrong token | `error(unauthorized, fatal)`, close 4002 |
| no `hello` in time, or another message first | close 4003 |
| `join`/`action`/`leave` from a spectator, or `action` without a fly | `error(<code>, fatal=false)`, ignored |
| superseded by a newer connection of the same owner | close 4004 |
| server shutdown | close 1001 |
| SDK: connection closed by the server | `ConnectionLost(code, reason)` |

## 9. Testing (test-first)

- **Protocol**: every message round-trips through its model; unknown `type`
  is ignored; a bad payload raises; `export_schema()` equals the committed
  file; the catalog matches `engine.body.observation_spec("fly")` shape for
  shape and dtype for dtype.
- **Codec**: decoded channels equal what `World.observe` returns for the same
  state (shape, dtype, values).
- **Runner + roster, no sockets** (fake ports, `tick()` by hand): action
  applied on time; late/stale action → `idle` + `missed`; join queued until the
  next tick and logged; death → final observation + `died`, slot freed,
  lineage increments on rejoin; reattach after detach; supersede closes the
  old port with 4004; unattached fly idles without counting; latest-wins
  outbox drops the older observation.
- **Integration, in-process server** on an ephemeral port, `tps=50`: two SDK
  brains (random, scripted) plus one NPC share the world and all appear in
  spectator frames; a dropped connection idles the fly and a reconnect
  reattaches it; `leave` then `join` reattaches; wrong token → 4002, wrong
  version → 4001, no hello → 4003; a slow spectator receives the latest frame
  after a stall; `/healthz` returns 200.
- **Determinism guard**: `spawn_log` + `action_log` replayed through `World`
  reproduce the server's `state_hash` at the end of an integration run.
- **CLI smoke**: `serve` for ~20 ticks with `join` and `watch` against it;
  `schema` prints valid JSON.
- **Balance**: nothing new — the world rules are unchanged.

## 10. Known limits (v1)

No persistence (a restart is a new world); owner names are trusted on
localhost; a single shared token; no TLS; one body; no admin controls; the SDK
does not reconnect by itself; frames are full state every tick (fine at
32×24 and 5 tps, a diff format can be added additively); dead agents
accumulate in the engine state until sub-project 4 adds `despawn`.

## 11. What the implementation added (and where it deviates)

Additions, all additive to the protocol above:

- **Fly names.** Every life gets a deterministic two-word name from
  `(owner, lineage)` — `joined.name`, `died.name`, `frame.agents[].name`.
- **Moods.** `frame.agents[].mood` is one of `content, hungry, thirsty,
  sleepy, desperate, dying, dead`, derived from the visible needs (never a
  rule); the spectator draws it as a thought bubble (✨ 🍎 💧 💤 ❗ ☠️).
- **`say`.** A client → server message (`text`, ≤ 40 chars) that shows as the
  fly's speech bubble in frames — for brains that want to talk (LLM brains
  later).
- **Chronicle.** A `chronicle` server → spectator message: the naturalist's
  log, plain sentences written by the server from events (hatchings, meals,
  depletions, deaths, owners leaving and returning), with a compass sector
  ("in the north-east"); the last 30 lines are replayed to a new spectator.
- **Scores.** `frame.scores`: per owner, lives, best lifespan, alive — the
  spectator's leaderboard.
- **`welcome.motd`.**
- **`neurogarden watch --follow OWNER`** highlights one owner's fly.
- **`neurogarden serve` defaults to `--npc scripted:1`** (a resident fly), so
  a fresh server is never an empty garden; `--npc none` for silence.

Deviations from sections 2–6:

- `codec` and the mailbox live in `protocol/`, not `sdk/` — the server's
  hosted brains need them too.
- One `RemotePort` class serves both roles; `LocalPort` is the hosted brain.
- A `join` after `leave` re-attaches the same connection (no reconnect
  needed).
- The engine changed after all: the nest-adjacent spawn rule
  (`RULES_VERSION` 2), because three NPC flies hatching in the map's corner
  looked wrong on the first live run.
- The implementation was written and tested as a whole (261 tests) and
  committed in five reviewable steps rather than transcribed from a
  code-carrying plan.
