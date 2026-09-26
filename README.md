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

Sub-projects 1 and 2 of 5: the simulation **engine**, the training **dojo**, and
the **live world** — a server, a wire protocol and a Python SDK, so several brains
can live in one Drosoville at the same time. No web client yet (sub-project 3);
see `docs/superpowers/specs/`.

## Quick start: a live garden

```bash
uv sync
uv run neurogarden serve                                   # a world with one resident fly
uv run neurogarden join --owner alice --brain scripted     # in another terminal
uv run neurogarden join --owner bob --brain random         # and another
uv run neurogarden watch --follow alice                    # and watch them all
```

The spectator shows the map, every fly's needs and mood (🍎 hungry, 💧 thirsty,
💤 sleepy, ❗ desperate, ☠️ dying, ✨ content), who is connected, a leaderboard,
and the naturalist's log the server writes as things happen:

```text
Day 2, dusk: Dusty Wing (alice) finds fruit in the north-east.
Day 3, night: Amber Zip (bob) dies of dehydration in the south.
```

Every fly gets a name; death is final for that fly, and the owner rejoins as life
#2. A brain that disconnects leaves its fly idling until the owner returns.

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
what every number in an observation means.

## Quick start: the dojo

```bash
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
  tick. A replay is a seed plus a list of actions — the live server's spawn and
  action logs replay through the engine alone to the same state hash.
- **Everything is a client.** A brain on your GPU box, a hosted "NPC" brain in
  the server process, and (soon) a human in a browser all speak the same
  protocol; the engine never knows which is which.

## Notes

- `RULES_VERSION` and `tests/make_golden.py`: any change to state evolution
  or observations must bump `RULES_VERSION` and regenerate the golden replay
  with `uv run python tests/make_golden.py`.
- Pass `--ascii` to `neurogarden watch` or `neurogarden.dojo.watch` on terminals
  where the emoji tiles misalign.
- The server refuses to bind a non-loopback host with the default token; pass
  `--token` to expose a world beyond your machine.

## Development

```bash
uv run pytest                 # everything, including the slow balance guard
uv run pytest -m "not slow"   # fast loop
uv run ruff check .
```
