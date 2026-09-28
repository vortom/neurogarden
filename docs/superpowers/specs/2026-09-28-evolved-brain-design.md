# NeuroGarden sub-project 5: the first learned brain

Date: 2026-09-28. Builds on the engine/dojo spec (2026-09-20) and the live-world
spec (2026-09-26). Section 7 records what the implementation added or changed.

## 1. Goal

A brain nobody wrote. A tiny network is evolved in the dojo by living many short
lives, then released into the garden as an ordinary client — beside the scripted
survivor, the random baseline and the humans — where the hall of flies decides
how good it is. The point is the "wow": a few hundred numbers found by lifespans
alone, foraging in Drosoville, with its neurons visible over its head.

Not the point (later, if ever): gradient methods, connectome-inspired wiring,
imitation of exported human lives (the archive's `export` makes that possible;
this sub-project does not do it).

## 2. Package layout

```text
src/neurogarden/
  dojo/features.py        the tiny feature vector, shared by wrapper and brain   (new)
  dojo/evolve.py          the evolution strategy                                (new)
  dojo/stats.py           + fitness_wellbeing, FITNESSES
  brains/evolved.py       Genome, EvolvedBrain                                  (new)
  brains/weights/         evolved-v1.npz, the shipped weights                   (new)
  brains/base.py          + think(): a brain's optional thought(), bounded
  server/ports.py         hosted brains show their thoughts as speech bubbles
  sdk/session.py          run_brain shows them too
  cli.py                  evolve; join --weights
```

## 3. The brain

`tiny_features(observation)`: smell (3 scents × own tile + N/E/S/W = 15), touch
(4), body (5: needs and age, scaled to [0, 1]), light (1) — 25 numbers. No vision:
the scripted brain uses it only to avoid walls, and bumping is a cost evolution
can feel. `TinyObservation` now wraps this same function, so what a network sees
in training is byte-for-byte what it sees in the garden.

`Genome`: `w1 (25×H)`, `b1 (H)`, `w2 (H×7)`, `b2 (7)`, H = 16 by default (551
weights). `hidden = tanh(x·w1 + b1)`, `scores = hidden·w2 + b2`. Saved as an
`.npz` with the four arrays and a `meta` JSON string (features version, hidden
size, how it was evolved).

`EvolvedBrain(seed, path=None, genome=None, temperature=None)`: with a
temperature the action is drawn from `softmax(scores / temperature)` using the
brain's own `SplitMix64` (seeded by `reset(seed)`, so a life is reproducible);
with temperature 0 the highest score wins. The shipped brain acts at the
temperature it was evolved at (recorded in `meta`): a policy that always does
the same thing gives evolution nothing to rank, and a brain deployed colder than
it was bred behaves differently from what its lifespans promised.

`thought()`: the hidden layer as a sparkline (`▁▂▃▄▅▆▇█`, one glyph per neuron)
— a brain scope in a speech bubble. `brains.base.think(brain)` is the runners'
side: an optional `thought()`, cut to `SAY_MAX`, exceptions swallowed. Hosted
brains (`LocalPort`) and `run_brain` say it every 5 ticks.

## 4. The strategy (`dojo/evolve.py`)

A plain evolution strategy (Salimans et al. 2017), numpy only:

- start from random weights scaled by 1/√fan-in (all zeros would idle every fly
  to death);
- each generation: fresh world seeds shared by every candidate, `population/2`
  Gaussian perturbations mirrored (`+ε`, `−ε`), one life per seed per
  candidate, rank-normalised fitness in [−0.5, 0.5], step
  `θ += lr / (population·σ) · Σ rank·ε`; a generation where everyone scored the
  same steps nowhere;
- fitness `wellbeing` by default: `lifespan × (1 + mean_wellbeing)` — a slope
  to climb before the first extra tick of life is won; `lifespan` is available
  and is what the hall of flies ranks by;
- evaluation in a `ProcessPoolExecutor` (one env per worker; the engine is
  pure) or in-process for tests; the whole run is deterministic from `seed`
  given a fixed population and worker-independent seeds.

`EvolveConfig(generations, population, sigma, learning_rate, hidden, episodes,
max_steps, seed, workers, map, fitness, init_scale, temperature)`;
`evolve(config, start=None, on_generation=None) -> Evolved(genome, history,
config)`; `Evolved.meta` is what goes into the file.

## 5. CLI

```text
neurogarden evolve --out brain.npz [--generations N] [--population N] [--episodes N]
                   [--max-steps N] [--sigma X] [--learning-rate X] [--hidden N]
                   [--fitness lifespan|wellbeing] [--seed N] [--workers N] [--start FILE.npz]
neurogarden join --brain evolved [--weights FILE.npz]
neurogarden serve --npc evolved:1 --npc scripted:1 --npc random:1
```

`evolve` prints one line per generation (best, mean, centre fitness, seconds)
and writes the weights at the end; Ctrl-C writes nothing.

## 6. Testing

- `tiny_features` equals the wrapper's output; genome ↔ vector ↔ file round
  trips; a foreign features version is refused.
- The brain acts by the highest score at temperature 0 and draws
  reproducibly above it; sparklines are ≤ `SAY_MAX` and free of control
  characters; `think` is optional and never raises.
- A hosted brain's thoughts show as `say` in frames.
- `evolve` is deterministic (two runs, same weights), resumes from `--start`,
  validates its config; ranks are centred and silent on a tie.
- The shipped weights load, carry their meta, act.
- Balance (slow): the shipped brain's median lifespan over 20 seeds beats the
  random brain's by a wide margin; the number is printed beside the scripted
  brain's.
- CLI: `evolve` with a tiny config writes a loadable file; `join --weights`
  flies it.

## 7. What the implementation added (and where it deviates)

Filled in at the end of the sub-project.
