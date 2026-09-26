# NeuroGarden — Architecture

Date: 2026-09-20
Status: approved in brainstorm; governs all sub-project specs.

> **NeuroGarden** — *Small worlds. Strange minds.*
> First world: **Drosoville**.

## 1. What this is

NeuroGarden is a life simulator for small brains: a persistent, 8-bit-RPG-styled
pixel world in which embodied agents — first a fruit fly — have to stay alive.
People connect their own brain (a scripted policy, an evolved net, an RL agent,
a connectome-based model, or a human at a keyboard) to a body in the world and
see how well it lives.

Positioning: by September 2026 there are dozens of projects that wire a fly
connectome into a one-off game (see `awesome-fly`). Each builds a brain *and* a
throwaway world. NeuroGarden is the missing common habitat: one cute, watchable
world with a stable interface, where different brains can live side by side and
be compared.

Honesty rule: any connectome-based brain running here is a *model inspired by
real wiring*. Connections may come from real data; neuron equations, sensory
encodings and movement rules are modelling choices. The project never claims to
simulate a real fly.

### Audiences, in order

| Stage | Who connects a fly | Status |
|---|---|---|
| C | Only the author, on localhost | v1 target |
| A | Programmers and ML tinkerers bringing their own brain over a public API | architecture must allow without rewrite |
| B | Non-programmers who raise and teach a fly in the browser | possible later; must not be blocked |

## 2. Architectural decisions that are expensive to reverse

These are fixed now. Everything else is deliberately deferred (section 7).

1. **The engine is a pure library.** No I/O, no wall clock, no networking, no
   reward. `step(state, actions) → state, observations, events`. Runners wrap
   it: the *dojo* runs it in lockstep at full speed for training; the *world
   server* runs it on wall-clock ticks for living.
2. **Everything is a client.** A human in a browser, a remote brain on
   someone's GPU box, and a first-party hosted brain all talk to the world
   server over the same versioned protocol. The engine has no special cases
   for any of them. The server never executes user code.
3. **Clients dial out over WebSocket.** Brains run behind home NAT; they open
   an outbound connection. No webhooks, no public endpoint required.
4. **Agent stream ≠ spectator stream.** A brain receives only what its body
   senses. Spectators receive the rendered world. World state never leaks into
   agent observations.
5. **Identity is in the handshake from day one.** Token, owner id, requested
   body. On localhost the token is a static dev token; real auth later swaps
   the token issuer, not the protocol.
6. **One world = one process; state is plain integer data.** A snapshot plus
   the per-tick action log reproduces a world exactly (replay, crash recovery,
   redeploy). Scaling means running more worlds, never distributing one.
7. **Python engine, server and SDK; thin TypeScript browser renderer.**
   `pip install neurogarden` is the front door for the ML audience. Insurance
   for a later Rust/WASM core: narrow engine API, integer-only state, a fully
   specified PRNG, and golden replay tests that any port must reproduce
   tick for tick.

Protocol envelope (details belong to sub-project 2): JSON messages
`{v, type, payload}`; unknown `type` values are ignored so the protocol grows
additively; a JSON Schema is the single source of truth from which Python and
TypeScript types are generated.

## 3. Shape

```text
                 ┌──────────── world server (1 process = 1 world) ────────────┐
 browser ──WS──► │ gateway: token · protocol version · rate limit             │
 (human+spectate)│      ▼                                                     │
 brain ────WS──► │ runner: wall-clock ticks · action deadline · default action│
 (remote client) │      ▼                                                     │
                 │ ENGINE (pure lib): step(actions) → observations, events    │
                 │      ▼                                                     │
                 │ storage: snapshots + action log                            │
                 └────────────────────────────────────────────────────────────┘

 dojo (local): same ENGINE · lockstep runner · Gymnasium wrapper · no network
```

Live-world timing: 5 ticks per second by default (configurable); one action per
agent per tick; an agent that misses the tick deadline gets `idle`; a brain
that disconnects leaves its fly idle (later: walks home and sleeps).

## 4. Four concepts that stay separate

| Concept | What it is | Where it lives |
|---|---|---|
| **World consequences** | What actually happens: hunger, thirst, fatigue, damage, death | Engine (body state + events) |
| **Reward** | Per-step signal for a learning algorithm | Learner side (dojo wrapper, user code) |
| **Fitness** | Per-episode number for evolution | Learner side, computed from episode stats |
| **Public score** | Comparable number shown to people | Stats computed from events; primary = lifespan |

The engine has no reward. The world has consequences, not goals. Reward,
fitness and score are pure functions over body state, events and episode
stats, so each can change without touching the world.

## 5. Bodies, frames and presets

An agent is a **body** (what it can sense and do) plus a **brain** (how it
decides). The engine knows only bodies. A body is configuration: its sense
channels, its action set, and its **frame**.

Rule: the observation frame must match the action frame.

| Preset | Observation | Actions | Status |
|---|---|---|---|
| `allocentric` | Centred on the agent, north-up, not rotated | `move_n/e/s/w` | v1 |
| `egocentric` | Centred, rotated to heading, left/right antennae | `turn_left/right`, `forward` | later experiment |

Both presets are partial and local: an agent never learns its coordinates.
Sense channels and the action set are declared per body (and later sent in the
protocol handshake), so new senses, stages (egg → larva → pupa → adult) and
species are new configuration, not engine rewrites.

## 6. Sub-projects

Each gets its own spec → plan → implementation cycle.

| # | Sub-project | Delivers |
|---|---|---|
| 1 | **Engine + dojo** | Pure simulation library, Drosoville text map, Gymnasium environment, random and scripted brains, terminal renderer |
| 2 | Protocol + server + SDK client | JSON Schema protocol, world server with wall-clock runner, Python client; a brain connects over WebSocket to a live world |
| 3 | Web client | TypeScript pixel renderer, spectating, human keyboard play |
| 4 | Storage + replay | Persistent snapshots, action log, replay viewer |
| 5 | Learned brain example | A tiny evolved net trained in the dojo and released into the garden — the first "wow" |

Later, unordered: cloud deployment and real auth; many worlds and a lobby;
leaderboards; multiple flies, competition, predators and hazards; flight;
life cycle; egocentric preset; brain telemetry ("brain scope") for spectators;
hosted native brains and in-browser teaching (audience B); connectome-based
brains.

## 7. Deliberately deferred

Auth provider and accounts; SQLite vs Postgres; hosting vendor and containers;
lobby and matchmaking; native-brain format and breeding; binary message
encoding; Canvas vs PixiJS; trajectory export format; brain telemetry message.
All are additive or swappable behind the decisions in section 2.

## 8. Repository layout (target)

```text
src/neurogarden/
  engine/    pure simulation library          (sub-project 1)
  dojo/      Gymnasium wrapper, rewards, stats (sub-project 1)
  brains/    example brains                   (sub-projects 1, 5)
  server/    gateway, runner, storage         (sub-projects 2, 4)
  sdk/       network client                   (sub-project 2)
protocol/    JSON Schema, generated types     (sub-project 2)
web/         TypeScript renderer              (sub-project 3)
tests/
docs/superpowers/specs/
```

One Python distribution (`neurogarden`) with optional extras for heavier
dependencies (`neurogarden[server]`), managed with `uv`.
