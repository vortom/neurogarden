# NeuroGarden sub-project 6: the connectome fly

Date: 2026-10-03. Implements row 6 of the roadmap
(`2026-10-03-evolution-simulator-roadmap.md`). Section 9 records what the
implementation added or changed.

## 1. Goal

A fly whose brain is the wiring of the MaleCNS connectome: real connections,
modelled dynamics, our own sensory encoding and a learned readout — and, beside
it from the first day, the same fly on a random graph of the same size, so the
question "does the wiring matter?" has a number. Several of them live in one
garden at once, cheaply.

Honesty rule (README): this is a model *inspired by* real wiring. The data
gives who connects to whom, how many synapses, and a predicted transmitter.
The neuron equation, the signs, the scaling, what counts as "smelling fruit"
and how activity becomes an action are our choices, listed in section 4.

## 2. What the spike measured (2026-10-03, this Mac, scipy)

Scripts kept in `docs/superpowers/spikes/`. Source files: MaleCNS v1.0 flat
connectome (`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`,
CC-BY 4.0): annotations (211,577 bodies), neurotransmitters (1,835,518 rows),
weights (151,856,684 segment pairs).

- Neurons with `status == "Traced"`: **165,122**. Connections between them:
  **25,563,197**, carrying 124,025,048 synapses (median 2 per connection, max
  2,591); 6,235,682 connections have ≥ 5 synapses.
- Consensus transmitter of those neurons: acetylcholine 103,718, glutamate
  29,296, GABA 22,055, histamine 5,910, unclear 3,100, missing 502, dopamine
  392, octopamine 101, serotonin 48.
- Populations (annotation `class` / `superclass`): olfactory 2,639, gustatory
  1,428, hygrosensory 66, thermosensory 25, visual 4,107, Kenyon cells 4,064,
  dopamine neurons 340, mushroom-body output neurons 97; descending neurons
  1,314.
- Building the graph from the files: 85 s (the weights are read in batches).

| graph | neurons | connections | one update, 1 fly | 10 flies |
|---|---|---|---|---|
| full | 165,122 | 25,563,197 | 61.7 ms | 884.8 ms |
| full, ≥ 5 synapses | 165,122 | 6,235,682 | 15.4 ms | 139.8 ms |
| central brain (no optic lobes, no nerve cord) | 50,668 | 10,074,936 | 19.1 ms | 176.2 ms |
| **central, ≥ 5 synapses** | 50,668 | 2,425,802 | **5.2 ms** | **28.2 ms** |

On a CPU the cost is the number of connections times the number of flies;
batching does not beat memory bandwidth. A 5 ticks/s world gives 200 ms per
tick, so on a CPU the pruned central brain carries ten flies with a few
updates per tick, and the full graph is a GPU job (roadmap §6).

Rate dynamics on the pruned central brain (`r ← (1−leak)·r + leak·tanh(gain·W·r + input)`,
inputs normalised per neuron): a smell on half the olfactory neurons reaches
the descending neurons in 3–5 updates; at gain 1 it fades (mean |activity|
0.001), at gain 4 with leak 1 it carries (0.31 after 12 updates, 1,039 of 1,314
descending neurons active, and moving the smell changes them by 0.32). So the
gain is a parameter to learn, and the network has memory across ticks — which
the small evolved network does not.

## 3. Package layout

```text
src/neurogarden/
  connectome/data.py      fetch the files, build and cache a graph, load it, random control   (new)
  connectome/model.py     Wiring, the rate network (numpy/scipy; torch optional), encoding, readout (new)
  brains/connectome.py    ConnectomeGenome, ConnectomeBrain                                  (new)
  brains/weights/         connectome-v1.npz, connectome-random-v1.npz (the control)
  dojo/evolve.py          evolves any trainable brain, not only the small network
  sdk/flock.py            N brains in one process, each its own owner                         (new)
  cli.py                  connectome fetch|build|info; evolve --brain; flock
docs/superpowers/spikes/  the two measurement scripts
```

Optional extras: `neurogarden[connectome]` = scipy + pyarrow (pyarrow only to
build). `torch` is not a dependency: the CUDA backend is used when torch is
importable and asked for. The core stays numpy-only; without the extra,
`--brain connectome` is refused with the command that installs it.

## 4. The model — every choice ours

**Graph variants.** `central` (the default): neurons whose superclass is one
of cb_intrinsic, cb_sensory, descending_neuron, ascending_neuron, cb_motor,
visual_projection, visual_centrifugal, cb_endocrine, sensory_ascending.
`full`: every traced neuron. `min_synapses` (default 5) drops weaker
connections. A built graph is cached as
`~/.cache/neurogarden/malecns/graph-<variant>-ms<k>.npz` with its neuron table
and a content hash; `NEUROGARDEN_CACHE` moves the cache.

**Weights.** `W[post, pre] = sign(pre) · synapses`, then each row divided by
its total |input| so a neuron's inputs sum to at most 1. Sign: acetylcholine
+1; GABA, glutamate, histamine −1; everything else (dopamine, serotonin,
octopamine, unclear, missing) +1.

**Dynamics.** A leaky rate network, `substeps` (default 3) updates per world
tick: `r ← (1−leak)·r + leak·tanh(gain·W·r + input)`, leak 0.5. The state `r`
persists across ticks within a life and is zeroed at birth. A spiking model
would need hundreds of updates per tick; the rate model is what the tick
budget allows, and prior art uses the same simplification.

**Senses in.** The 25 tiny features (the same vector the evolved brain sees)
each drive their own fixed group of neurons, chosen by class and split by a
seeded shuffle:

| features | neurons driven |
|---|---|
| fruit smell: own tile, N, E, S, W | olfactory, five groups |
| humidity smell ×5 | hygrosensory, five groups |
| nest smell ×5 | thermosensory + other chemosensory, five groups |
| touch: bumped / on fruit / by water / on nest | mechanosensory, gustatory (two groups), mechanosensory |
| body: satiety, hydration, energy, health, age | endocrine and dopamine neurons, five groups |
| light | visual-projection neurons |

Each feature has a learned gain (25 numbers).

**Actions out.** The 1,314 descending neurons are pooled by a fixed seeded
sparse projection into 64 features; a learned linear readout maps them to the
7 actions (455 numbers). The action is drawn from the softmax at the brain's
temperature, as for the evolved brain.

**What is learned**: 25 input gains, the network gain, the readout — 481
numbers, evolved by the existing strategy. **What is never trained**: the
wiring.

**The control.** `random`: the same neurons, the same number of connections,
the same weights and signs, with every connection's target shuffled (seeded).
Trained with the same budget and shipped beside the real one.

## 5. Brains, files and commands

`ConnectomeGenome` (`.npz`): the 481 numbers plus meta — graph variant,
`min_synapses`, control seed or none, graph hash, substeps, leak, encoding and
projection seeds, temperature, ancestry. The graph is not in the file; the
brain refuses to load against a graph with another hash.

`ConnectomeBrain(seed, path=None, genome=None, wiring=None, temperature=None)`
follows the `Brain` protocol; `thought()` is a sparkline of 16 of the pooled
descending features. `BRAINS["connectome"]` loads the shipped weights and the
cached graph; without the graph it says how to get it.

```text
neurogarden connectome fetch                      # download the three files (~1.1 GB) into the cache
neurogarden connectome build [--graph central|full] [--min-synapses 5]
neurogarden connectome info                       # what is cached, sizes, hashes
neurogarden evolve --brain connectome [--graph central] [--control random] --out mine.npz
neurogarden join --brain connectome [--weights mine.npz]
neurogarden flock --brain connectome --count 10 [--owner-prefix cns] [--weights …]
neurogarden serve --npc connectome:2
```

`flock`: one process, N sessions, each fly its own owner (`cns-1` … `cns-N`),
all sharing one wiring in memory. On a CPU each fly is updated in turn (no
batching win, §2); the torch backend batches them in one multiply.

## 6. Evolution of any brain

`dojo.evolve` takes a *trainable*: a small picklable description that every
worker turns into `size`, `initial(rng)` and `build(vector) → Brain`. The
small network is one trainable (unchanged behaviour and files); the
connectome brain is another (each worker loads the graph once). Fitness,
ranking, seeds and the file's ancestry meta are shared.

## 7. Testing

No test downloads anything. A synthetic wiring (a few dozen neurons with the
needed classes) and tiny feather fixtures written by the tests cover: the
build (filtering to traced neurons, signs, pruning, the central variant, the
hash), row normalisation, the random control (same counts, different edges,
seeded), the rate step against a hand computation, encoding groups (disjoint,
seeded), the projection, genome round trips and the hash refusal, the brain's
reproducibility and state reset, `thought()`, evolve with the connectome
trainable (deterministic, tiny), the flock against a real server, and the CLI
(`info`, refusals without the extra or the cache). The torch backend is
tested against the numpy one when torch is importable. A slow guard, skipped
without the cache, runs the shipped connectome and control brains and prints
their median lifespans beside the others.

## 8. Not in this sub-project

Neural telemetry in the server's archive (a protocol addition; the flock can
write it locally with `--telemetry FILE.npz`), plasticity, the full graph on
CUDA as a tested path (the backend is written and unit-tested on CPU torch
when available; real GPU runs happen on the owner's machine), a spiking model.

## 9. What the implementation added (and where it deviates)

**The largest deviation: the shipped brain was taught, not bred** (§4 said the
481 numbers are "evolved by the existing strategy").

- *Evolution from nothing stalls.* On the real graph (30 generations planned,
  32 candidates, two lives of up to 1500 ticks each, about two minutes a
  generation on this Mac) the centre reached the drink-rest-and-starve policy
  by generation 3 (fitness 1315: a death at tick 899, the random brain's
  lifespan) and by generation 10 every one of the 32 candidates scored exactly
  1315. Rank-normalised scores that all tie give no step; the run was stopped.
  The small network met the same wall and needed 120 generations to pass it;
  at a hundred steps a second per core that is days here.
- *So the readout is taught* (`neurogarden distil`, `dojo/distil.py`). The
  shipped evolved brain senses the same 25 features, so it can say for any
  moment how likely it would be to take each action. Round 1: it flies, the
  connectome network senses the same moments, and the readout is fitted to its
  answers by softmax regression (the readout is linear, so this is the whole
  problem). Later rounds: the student flies, the teacher labels what the
  student met, and the fit is redone on everything so far (DAgger). The input
  side (25 gains, the network gain) keeps its starting values; `evolve
  --start` can breed it afterwards. A lesson of five rounds of four lives
  takes about five minutes.
- *What the lesson showed about the model.* The teacher is almost linear in
  its features: a readout fitted straight on the 25 features agrees with it on
  98% of moments. Through the real wiring the same fit agrees on 76%. Measured
  on a held-out life (`spikes/malecns_throughput_spike.py`, R² of a ridge fit
  from the network back to each group of senses): the 64 pooled features
  recover fruit smell at 0.24, humidity 0.40, nest smell 0.11, touch 0.11,
  body 0.09, light 0.58; all 1,314 descending neurons recover 0.66, 0.78,
  0.14, 0.47, 0.32, 0.97. So the pooling loses most of what reaches the
  descending neurons, and not everything reaches them. A wider readout (256
  features) fitted better and did not live longer; leak 1.0, six updates a
  tick and other gains were no better than the defaults (gain 2 with doubled
  input fits slightly better).

**Deviations from the sections above**

- **The control permutes rows** (§4 said "every connection's target shuffled").
  `Wiring.randomised(seed)` gives every neuron another neuron's whole set of
  inputs. Same neurons, connection count, weights, signs and — unlike a
  per-connection shuffle — exactly the same balance of input per neuron, so the
  row normalisation is untouched and the only thing lost is *which* neurons a
  neuron listens to. The permutation comes from the project's own generator,
  so the control is the same graph on any numpy. On the command line it is
  `--control SEED`, not `--control random`; the shipped control uses seed 1.
  The control is not a graph without pathways: from the 1,780 driven neurons
  the real graph reaches 441, 1,292 and 1,312 of the 1,314 descending neurons
  within one, two and three connections; control 1 reaches 282, 1,200 and
  1,275. What it lacks is the fly's pathways, not pathways.
- **Senses** (§4 table). "On nest" drives proprioceptive and unclassified
  sensory neurons, not a second mechanosensory group; a group holds at most
  128 neurons; a class the graph lacks falls back to a pool of other sensory
  neurons instead of failing (a graph with no sensory neurons at all is
  refused). On the central graph the groups are uneven, because the classes
  are: 128 neurons per fruit-smell direction, 13 per humidity direction, 5 per
  nest-smell direction, 128 for each touch, 82 per body need, 128 for light.
  The learned input gains are what evens that out.
- **Actions out.** The projection is not sparse: every descending neuron feeds
  exactly one of the 64 buckets with a seeded sign, scaled by one over the
  square root of the bucket's size. The pooled features are multiplied by a
  fixed 10 before the readout, and the network gain is stored as its logarithm
  and starts at 3 — both so that evolution's one step size suits all 481
  numbers.
- **Trainable** (§6). The interface is `size`, `initial(rng, scale)`,
  `genome(vector)`, `brain(vector)`, `check_start(genome)` and `describe()`
  (what goes into the file's meta), not `build(vector)`.
- **Flock on a CPU** (§5). The flies of a flock are stepped together in one
  multiply on either backend; it costs what stepping them in turn would (§2),
  and keeps one code path.
- **Not built: `flock --telemetry`** (§8). A connectome fly's activity is shown
  as its thought bubble only. Recording it belongs with the other datasets in
  sub-project 9.
- **Torch backend: written, not run.** There is no torch wheel for this Mac
  (x86-64, Python 3.14), so the test that compares it with numpy is skipped
  here. It is unproven until it runs on the GPU machine.

**Additions**

- `neurogarden evolve` writes the weights after every generation (to a
  `.part` file, then renamed), and says which generation the file holds when it
  is stopped: a connectome run takes an hour and a crash used to leave nothing.
- `--weights FILE` alone picks the brain: the file says whether it is a small
  network or a connectome readout. `evolve --start FILE` inherits the brain,
  graph, pruning, control and substeps from the file.
- `brains.evolved` grew `choose` (the seeded softmax draw) and `sparkline`,
  shared by both learned brains.
- **What a brain file is tied to.** The graph's hash covers the connections
  (in fixed number types, so it does not depend on how scipy stores them) and
  every neuron's class and superclass, because the labels decide which neurons
  a sense drives. The file also carries `model_version` (1): the encoding
  table, the pooling and the update rule as code. Another hash or version is
  refused on load — and by `evolve --start`, which will not carry weights from
  one wiring onto another (the real graph onto its control, say).
- **The cache is whole or refused.** The graph and its neuron table are
  written beside their names and renamed; the table carries the hash, and
  `load` refuses a pair that does not match (half a rebuild, a truncated file)
  with the command that fixes it.
- `evolve` refuses `--graph`, `--min-synapses`, `--control` and `--substeps`
  for the small network instead of ignoring them. A run that carries on from
  `--start` draws on from where its ancestors stopped (the seed is mixed with
  the generations behind it) instead of breeding on the same worlds and noise
  again — which matters now that stopping and resuming is ordinary.
- A server builds its hosted brains before it creates the world, so
  `serve --npc connectome:1` without a graph is refused with nothing left
  behind; `dojo.watch` refuses the same way.
- **The flock.** Every fly has its own stream of chance (`--seed` plus its
  place in the flock, mixed with its life number): flies that share weights
  must not share their draws, or ten flies are one fly ten times. The brains
  think off the event loop, so the sockets are read while a big graph is
  multiplied. Owners are `cns-N`, `rnd-N` for the control, and the file's name
  for `--weights`, so a second flock does not take over the first one's flies.
- **One layout, one network.** A `Wiring` keeps every row's columns ascending
  (torch's CSR requires it, and the sums then add up in the same order
  wherever the graph came from), and holds the network built on it, so on
  torch the graph is put on the device once, not once per fly.
- `connectome fetch` checks the bytes received against the promised length
  and times out on a stalled connection; `connectome info` names a neuron
  table it cannot read instead of dying on it.
- The synthetic wiring in the tests has 400 neurons, not a few dozen: the
  projection needs 64 descending neurons to pool.
- Building the central graph from the cached files takes 78 s through the
  package (85 s in the spike).
