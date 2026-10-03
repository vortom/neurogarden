# NeuroGarden: the life and evolution simulator — roadmap

Date: 2026-10-03. Sits above the architecture spec (2026-09-20) and the five
sub-project specs. Sub-projects 1–5 are built and merged; this document is the
plan for what comes next and the reasoning behind it. Each sub-project below
still gets its own spec, build, two reviews, fix wave and PR, as before.

## 1. What we are building

A **life and evolution simulator for small minds**: one persistent,
watchable pixel world in which brains of different kinds — real fly wiring
(MaleCNS), learned networks, hand-written scripts, and humans at a keyboard —
live side by side under the same rules, are ranked by the same number, leave a
complete record of every life, reproduce when they thrive, and evolve under
pressures we choose. The record is also a dataset: every life, human or not,
feeds reinforcement learning, imitation and evaluation, and strategies are
compared across worlds. The world server runs in the cloud as a **public
garden**: anyone connects a fly — a MaleCNS brain on their own GPU, a learned
brain, or themselves at the keyboard — and every life they live lands in the
same archives, so the dataset grows with the community.

Not a simulation of a real fly. The world is a tile garden, the body has seven
actions, and a "MaleCNS fly" is real wiring with modelled senses, a learned
readout and modelled learning. The README's honesty rule stands: *a model
inspired by real wiring*. Where the project touches science it does so by
building the controls in — a random graph of the same size next to the real
one, lesions, held-out worlds, seeded lives — so a claim can be checked, not
by claiming to have uploaded a fly.

## 2. Where it stands among existing projects (researched 2026-10-03)

Fly-connectome projects: the Fly Brain Hub directory lists **410** fruit-fly
brain projects (https://fly-brain-hub.vercel.app/). Asked for projects with
several connectome-driven flies in one world, evolution or reproduction of
connectome agents, or persistent worlds with recorded data for RL, it has
**none**; two have a human playing against a connectome agent (Fly Brain
Dungeon, fly-muaythai). The viral games (DOOMFLY, NeuroCraft Fly, Fly64, …)
each wire one MaleCNS/FlyWire simulation to one game with hand-chosen inputs
and a readout. Patrick Mineault's critique of them
(https://www.neuroai.science/p/are-flies-playing-beat-saber) is the bar to
clear: "A randomly activated recurrent neural network is used to button mash
on an N64 emulator. That's it!" and, with trained decoders on top, "the
connectome is beside the point". His prescription — genuine sensory-motor
loops, constraints, measurable benchmarks, controls — is what sections 4–5
build in.

Artificial-life evolution simulators are a mature genre and none of them uses
real wiring, mixes humans and RL agents in one population, or keeps a
replayable archive as a dataset: The Bibites (neural-network creatures,
pheromones, "speciation and lineage trees", https://thebibites.itch.io/the-bibites),
Petriarch (browser god game, "behavior and bodies are genome-driven, with no
authored fitness function", https://github.com/brac/petriarch), ERAIASON
(animats that eat, mate and fight, https://ccrock4t.itch.io/eraiason), Aeon
(browser civilisation sim of evolving networks), Polyworld (the classic), ALIEN
(CUDA particle life, https://alien-project.org/) and Microcosmos (GPU,
differentiable, https://www.alphaxiv.org/abs/2607.02954). These are the
reference points for reproduction, speciation and population tooling.

RL platforms — Neural MMO (massively multi-agent survival,
https://arxiv.org/pdf/1903.00784) and Multi-Agent Craftax
(https://arxiv.org/pdf/2511.04904) — are benchmarks for training agents, not
watchable worlds with real wiring or human play data.

Scientific precedent for "evolution reveals strategies": *Flies as Ship
Captains? Digital Evolution Unravels Selective Pressures to Avoid Collision in
Drosophila* (https://arxiv.org/abs/1603.00802) evolved collision avoidance in
silico with "collision avoidance [as] the sole penalty" and found the strategy
evolves only "in a narrow range of cost/benefit ratios" — the kind of question
NeuroGarden's world packs and breeding can ask in a playful setting.

**Unique combination**: (1) one persistent world shared by real-wiring brains,
learned brains and humans; (2) every life archived and replayable, so the
garden is a dataset; (3) reproduction, heritable bodies and chosen selection
pressures; (4) controls built in. Scientific value is modest and real: does
the wiring matter (vs. a random graph), does mushroom-body plasticity learn an
aversion, which pressures evolve which strategies, how do human strategies
compare with evolved ones.

## 3. The loop

```text
brain files ──join──► LIVE GARDEN ──records──► ARCHIVE ──replay, rank──► EVALUATE
     ▲          (you, scripted, evolved,        (every tick, life,          (hall of flies,
     │           MaleCNS ×N, random graph)       log, telemetry)             bench: brains × worlds)
     │                                               │                             │
     └──────── new weights ◄── LEARN (dojo) ◄── export (obs, action) ◄── scores → fitness
                               evolution · RL · imitation      ◄── WORLD PACKS (train / held-out)
```

A fly's brain: senses → *learned* encoding → fixed graph (MaleCNS wiring,
random control, or a small network) → *learned* readout → action. Only the
encoding and readout learn, unless a plasticity rule is added inside the graph
on purpose (section 5, last item).

The split the public garden needs already exists: the **backend** is the world
server (engine, runner, archive, WebSocket API; one process and one SQLite
file per world; it also serves the static page); the **frontend** is a static
bundle that speaks only the protocol and can live on any host; **clients** are
the SDK, `join` and `flock` — the brains run where their owners run them,
never on the server, which is why a world stays cheap however heavy the
brains are.

## 4. Sub-projects

| # | Name | Delivers | Needs the engine changed? |
|---|---|---|---|
| 6 | The connectome fly | `neurogarden connectome fetch` (MaleCNS v1.0 flat files, ~1.2 GB, cached outside the repo); `ConnectomeBrain` (LIF over the wiring; CPU/scipy backend for subgraphs, CUDA backend for the full graph); the random-graph control; `neurogarden flock --brain connectome --count N` (N flies, one batched update, each its own owner); neural telemetry table in the archive; evolved readout; balance guard | no |
| 7 | World packs and the bench | worlds as map + config: scarce water, poisoned fruit (a resource kind with its own scent and a `damaged` consequence), predators (a hunting body, a new occupant class), seasons, bigger maps; per-world halls of flies; `neurogarden bench --brain … --world … --lives 20` (seeded, median lifespan and stats per cell, JSON/CSV); `evolve --map a --map b` for generalists; held-out worlds | yes, additive: new resource kind, body, events; `RULES_VERSION` bump; senses keep their shapes |
| 8 | The public garden | the deferred cloud step: a Docker image of the server; TLS behind a reverse proxy, the page on the same host; accounts and per-owner tokens with invite links; persistent volume, backups from the archive's own snapshots; rate limits and `say` moderation; a public Drosoville (frozen rules) first, worlds per group later with a lobby; dataset releases (archive dumps + `export` bundles, CC-BY, with a consent line for human play: *your lives are recorded and published*) | no |
| 9 | Learning from the record | `export --format jsonl|npz|parquet`; `MimicBrain` (behaviour cloning from archived human lives, numpy); PPO on readouts and small brains in the dojo (optional extra, torch); comparison in the bench | no |
| 10 | Breeding | server-side reproduction: a fly that thrives for N days spawns a child beside it with the parent's brain file mutated by a seeded RNG, archived as an input; population cap and food as carrying capacity; lineage trees from the archive | no |
| 11 | Bodies that evolve | a few per-agent body genes (metabolism rates, scent ranges, move cost, vision radius within the fixed window) inherited and mutated; eggs as world objects with a hatch time and an energy cost; predators eat eggs | yes: per-agent body parameters, `lay`, eggs; `RULES_VERSION` bump; golden replay regenerated |
| 12 | Watching evolution | genomes per life in the archive; trait histograms over time; species by brain distance; a phylogeny and population view in the browser; `bench` over generations | no |
| — | Plasticity inside the wiring (research) | dopamine-gated synapse changes in the mushroom body: poisoned fruit → learned aversion; lesions and ablations as experiments | no (brain-side) |

Still deferred: many worlds per process, a marketplace of brains, anything that needs the engine to know who is a human.

## 5. The data we keep, and what it is for

| Data | Where | Used for |
|---|---|---|
| every tick's inputs (spawns with tiles, despawns, actions) | archive `ticks` | exact replay of any stretch; ghosts; `verify`; `export` |
| every life: owner, lineage, name, body, born/died ticks, causes, stats | archive `lives` | halls of flies, bench, lineage trees |
| brain genome per life (file reference + weights hash; mutations as inputs) | archive, SP9 | heredity, phylogeny, reproducible breeding |
| body genes per life | archive, SP10 | trait histograms, body evolution |
| neural telemetry (descending-neuron activity per tick, subsampled) | archive, SP6 | "which neurons fired before it turned"; brain-scope views |
| naturalist's log | archive `chronicle` | the story; ghosts |
| snapshots and checkpoints | archive | resume; integrity |
| `(observation, action)` per life, regenerated through the engine | `export` (npz now; jsonl/parquet in SP8) | imitation, RL, analysis |
| bench results (brain × world × seeds) | files, SP7 | strategy comparison, specialist vs generalist |

The engine never stores a reward. Reward, fitness and score stay on the
learner's side, as the architecture spec decided.

Two rules keep the dataset worth having: every record carries its
`RULES_VERSION` and world identity (it does), and the public Drosoville's rules
are frozen — experiments live in other worlds. Human lives are published only
under a consent line shown before the first hatch.

## 6. Hardware

Everything but the full-graph connectome runs on a laptop CPU (the engine
does ~5k steps/s per core; evolution and PPO on small brains use all cores).
The full MaleCNS graph (~165k neurons, ~25.6M synapses) wants a GPU: a 16 GB
RTX 5060 Ti holds the sparse matrix many times over and does a substep in
milliseconds, so N flies or a population of candidates are one sparse×dense
multiply. The connectome brain is written against a backend interface —
numpy/scipy on CPU for subgraphs and tests, torch-CUDA for the full graph — so
the same brain file gives the same behaviour on both, up to float rounding.
WSL2 is the recommended host on Windows (same environment as development, the
one-writer archive lock works, CUDA works inside it).

## 7. Decisions taken

- Reproduction server-side first (SP10), eggs in the engine after (SP11): the
  first needs no engine change and gives live evolution sooner.
- Which MaleCNS cell types carry each sense in and which descending neurons
  read actions out is settled in the SP6 spike from the annotation files,
  before any training, and recorded in the SP6 spec.
- Mutation rates, population caps, regrowth are tuned by experiment; a run
  that ends in extinction is a result.
- The public garden ships small and early (one VM, per-owner tokens, TLS,
  backups, dumps) rather than complete and late: the dataset starts growing
  the day the first friend connects a fly.
- Cloud deployment was deferred by the owner on 2026-09-28 and placed as SP8
  on 2026-10-03, after there is something worth connecting to and comparing.
