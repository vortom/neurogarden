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

## Notes

- `RULES_VERSION` and `tests/make_golden.py`: any change to state evolution
  or observations must bump `RULES_VERSION` and regenerate the golden replay
  with `uv run python tests/make_golden.py`.
- Pass `--ascii` to `neurogarden.dojo.watch` on terminals where the emoji
  tiles misalign.

## Development

```bash
uv run pytest                 # everything, including the slow balance guard
uv run pytest -m "not slow"   # fast loop
uv run ruff check .
```
