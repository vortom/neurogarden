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

Six sub-projects so far: the simulation **engine**, the training **dojo**, the
**live world** — a server, a wire protocol and a Python SDK, so several brains
can live in one Drosoville at the same time — the **browser client**, a
pixel-art view of the garden you can also play in, the **archive**: a world
that survives its process, and lives that can be watched again as ghosts or
exported as datasets, the **first learned brain**: a tiny network evolved
in the dojo, living in the garden with its neurons showing, and the
**connectome fly**: a brain that runs on the wiring of the MaleCNS connectome,
with a random-graph control beside it. See `docs/superpowers/specs/`.

Where it is going: a **life and evolution simulator for small minds** — real
fly wiring (MaleCNS), learned brains and humans in one world, every life
recorded, reproduction and heritable bodies, selection pressures as world
packs, the record used to train and compare strategies, and a public garden
in the cloud anyone can connect a fly to. The plan, the prior art and the
honesty rules are in
`docs/superpowers/specs/2026-10-03-evolution-simulator-roadmap.md`.

## Quick start: a live garden

```bash
uv sync
uv run neurogarden serve                                   # a world with one resident fly
uv run neurogarden join --owner alice --brain scripted     # in another terminal
uv run neurogarden join --owner bob --brain random         # and another
uv run neurogarden watch --follow alice                    # and watch them all
```

Then open **http://127.0.0.1:8765/**: the server serves a small pixel-art page —
the garden, every fly with its mood bubble, the naturalist's log, a leaderboard
and day/night. Type a name and *Hatch a fly* to play one yourself: arrows move,
space eats or drinks, R rests; your fly speaks the same protocol as any brain.
When it dies you get an obituary and can hatch again.

The spectator shows the map, every fly's needs and mood (🍎 hungry, 💧 thirsty,
💤 sleepy, ❗ desperate, ☠️ dying, ✨ content), who is connected, a leaderboard,
and the naturalist's log the server writes as things happen:

```text
Day 2, dusk: Dusty Wing (alice) finds fruit in the north-east.
Day 3, night: Amber Zip (bob) dies of dehydration in the south.
```

Every fly gets a name; death is final for that fly, and the owner rejoins as life
#2. A brain that disconnects leaves its fly idling until the owner returns.

## The archive: a world that survives its process

`neurogarden serve` writes the world to `neurogarden.db` (SQLite, one file per
world; `--archive` picks another path, `--archive :memory:` none at all). Stop
the server and start it again and it is the same world: same tick, same fruit,
same flies waiting for their owners, same lineage counters, scores and
naturalist's log. Everything the engine was ever told is written down, so any
life can be lived again:

```bash
uv run neurogarden lives                                   # the hall of flies
uv run neurogarden replay --owner alice --life 2           # watch it again in the terminal
uv run neurogarden export --owner alice --life 2 --out alice-2.npz   # (observation, action) pairs
uv run neurogarden verify                                  # re-run the whole history, check every hash
```

In the browser, every row of the hall of flies has a 👻: it opens
`?ghost=alice/2`, the same page replaying that life at 4× (`&speed=16` for
faster) with the ghost highlighted. Ghost frames are regenerated through the
engine from the archived actions — nothing but actions is stored, which is also
why `export` gives you exactly what the brain saw.

Connect your own brain with the SDK — the observation is the same dict of numpy
arrays the dojo produces, so a brain trained offline runs live unchanged:

```python
from neurogarden.sdk import Session

with Session("ws://127.0.0.1:8765", owner="alice") as session:
    fly = session.join()
    for observation in fly.observations():  # ends when the fly dies
        fly.act(observation.tick, my_brain.act(observation.channels))
    print(fly.name, fly.stats)
```

Other languages: `neurogarden schema` prints the protocol's JSON Schema
(`protocol/v1/neurogarden.schema.json`); `welcome` carries a catalog that says
what every number in an observation means and what every event carries. The
protocol grows by addition: a client ignores fields it does not know, so a
newer server never breaks an older brain.

## The evolved brain

Nobody wrote this one. `EvolvedBrain` is a 25-in, 16-hidden, 7-out network —
535 numbers — whose weights were found by evolution in the dojo: perturb,
live a few lives, keep what lived better (`neurogarden.dojo.evolve`, a plain
evolution strategy in numpy). It ships with the package, so it can live in
your garden right away, beside the hand-written survivor and the random
baseline:

```bash
uv run neurogarden serve --npc evolved:1 --npc scripted:1 --npc random:1
uv run neurogarden join --owner you --brain evolved      # or fly it yourself
```

How good is it? The balance guard (`uv run pytest -m slow`) runs 20 lives of
up to 6000 ticks per brain: median lifespan **5865** for the evolved brain,
6000 for the hand-written survivor, 899 for the random baseline.

Its speech bubble is its hidden layer, one glyph per neuron (`▁▂▃▄▅▆▇█`): a
brain scope you can watch fire as it smells fruit. Any brain with a
`thought()` method gets the same bubble, hosted or over the SDK.

Breed your own, and carry on from the shipped weights or from scratch:

```bash
uv run neurogarden evolve --out mine.npz --generations 40          # minutes on a laptop
uv run neurogarden evolve --out mine.npz --start src/neurogarden/brains/weights/evolved-v1.npz --max-steps 6000
uv run neurogarden join --owner you --weights mine.npz               # --weights means the evolved brain
```

Fitness is `forager` by default: `lifespan × (1 + mean wellbeing)` plus a
bounty per bite (`--fitness wellbeing` without the bounty, `--fitness
lifespan` for the bare public score). Plain lifespan has no slope until a fly
eats; wellbeing alone breeds a fly that drinks, rests and starves at tick 899
— the bounty is what pulled evolution past that wall. The brain acts by
drawing from the softmax of its scores at the temperature it was evolved at (a
policy that always does the same thing gives evolution nothing to rank), with
its own seeded generator, so a life is reproducible.

## The connectome fly

A brain built on real wiring. `ConnectomeBrain` runs a leaky rate network on
the central brain of the MaleCNS connectome — 50,668 neurons and 2,425,802
connections of at least five synapses, out of the 165,122 traced neurons and
25,563,197 connections of the whole dataset (Janelia FlyEM, MaleCNS v1.0,
CC-BY 4.0). Smell, taste and touch drive fixed groups of sensory neurons, the
body's needs drive dopamine and endocrine neurons, light drives visual
projection neurons; the activity of the 1,314 descending neurons, pooled, is
read out into the seven actions.

What this is, and is not: the *connections* are real. The neuron equation, the
signs given to transmitters, which neurons "smell fruit" and the readout are
this project's modelling choices. What is learned — 25 input gains, one network
gain and a 455-number readout, 481 numbers in all — is found by the same
evolution that bred the small network; the wiring is never trained. It is a
model inspired by real wiring, not a simulated fly. To keep that honest, every
connectome brain ships with a **control**: the same brain bred the same way on
a graph where each neuron receives another neuron's inputs
(`connectome-random`) — the senses still reach the descending neurons there,
by pathways no fly ever had. If the control lives as long, the wiring is not
what is doing the work — the balance guard prints both.

```bash
uv sync --extra connectome                     # scipy + pyarrow
uv run neurogarden connectome fetch            # three files, ~1.1 GB, into ~/.cache/neurogarden/malecns
uv run neurogarden connectome build            # the central graph; a minute or two
uv run neurogarden serve                                    # a garden…
uv run neurogarden flock --brain connectome --count 10      # …and ten connectome flies in it
uv run neurogarden flock --brain connectome-random --count 3   # and three of the control
```

`flock` runs N brains in one process, each its own owner (`cns-1` … `cns-10`)
with its own lives and place in the hall of flies; connectome brains share one
wiring and are stepped together. Unlike the small network, this brain has a
memory: its network state persists from tick to tick within a life. Its speech
bubble shows sixteen of its pooled descending features.

Breed your own readout (`--control SEED` breeds the control instead). The
weights are written after every generation, so a long run can be stopped:

```bash
uv run neurogarden evolve --brain connectome --out mine.npz --generations 30 --max-steps 1500
uv run neurogarden flock --weights mine.npz --count 5
```

On a CPU one update of the central graph costs a few milliseconds per fly, so
ten flies fit a 5 ticks/s world. The full graph is a GPU job: the same model
runs on torch when it is installed (`backend="torch"`); see the roadmap.

## Quick start: the dojo

```bash
uv run python -m neurogarden.dojo.watch --brain scripted     # or evolved, or random
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
  tick. A replay is a seed plus a list of inputs — the live server's archive
  replays through the engine alone to the same state hash, which is what
  `neurogarden verify` checks and what ghosts and datasets are made of.
- **Everything is a client.** A brain on your GPU box, a hosted "NPC" brain in
  the server process, and a human in a browser all speak the same protocol; the
  engine never knows which is which.

## Notes

- `RULES_VERSION` and `tests/make_golden.py`: any change to state evolution
  or observations must bump `RULES_VERSION` and regenerate the golden replay
  with `uv run python tests/make_golden.py`.
- Pass `--ascii` to `neurogarden watch` or `neurogarden.dojo.watch` on terminals
  where the emoji tiles misalign.
- The server refuses to bind a non-loopback host with the default token; pass
  `--token` to expose a world beyond your machine.
- An archive holds one world: `serve` takes the map and seed from it unless you
  pass `--map`/`--seed`, and refuses to run a different world (or a different
  `RULES_VERSION`) on top of it — start another file instead. A clean stop
  writes a snapshot; after a crash a resume replays at most 600 ticks from the
  last one. Not kept across a restart: missed-tick counters and speech bubbles.
  `*.db` files are git-ignored.
- Browsers may only open a socket to a world from the page that world served:
  the handshake checks `Origin` against the host and port `serve` was given, so
  set `--host` to the address people will actually browse to. Clients that send
  no `Origin` — the SDK, `neurogarden join`, `neurogarden watch` — are never
  affected. The page reads a non-default token from `?token=…` and wipes it from
  the address bar; that is a convenience for a world on your own machine, not a
  way to hand out access.

## Development

```bash
uv run pytest                 # everything, including the slow balance guard
uv run pytest -m "not slow"   # fast loop
uv run ruff check .
```

The browser client lives in `web/` (Vite + TypeScript, no framework):

```bash
npm --prefix web install
npm --prefix web test     # vitest
npm --prefix web run types  # regenerate src/wire.d.ts from protocol/v1/neurogarden.schema.json
npm --prefix web run build  # writes the bundle into src/neurogarden/server/static/ (committed)
npm --prefix web run check  # rebuild and fail if the committed bundle is out of date
```

`uv run pytest -m slow` runs the same bundle check from the Python side, and
skips it when npm is not installed.
