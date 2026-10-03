"""Evolution in the dojo: find a brain's weights by living many short lives.

A plain evolution strategy (Salimans et al. 2017): perturb the weights in mirrored pairs,
live one life per perturbation on the generation's seeds, rank the fitnesses, and step
the weights towards the perturbations that lived better. The default fitness is
`dojo.stats.fitness_forager`: lifespan weighted by wellbeing, plus a bounty per bite — a
slope to climb before the first extra tick of life is won, and a pull past the wall where
a fly drinks and rests but never eats. The hall of flies still ranks by lifespan alone.

What is bred is a *trainable*: the small network (`brains.evolved.MlpTrainable`) or the
learned parts around a connectome (`brains.connectome.ConnectomeTrainable`). The strategy
only sees a vector of numbers and a way to turn one into a brain.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from neurogarden.brains.base import brain_seed
from neurogarden.brains.connectome import ConnectomeTrainable
from neurogarden.brains.evolved import DEFAULT_HIDDEN, EvolvedBrain, Genome, MlpTrainable
from neurogarden.connectome.data import DEFAULT_MIN_SYNAPSES, VARIANTS
from neurogarden.connectome.model import DEFAULT_SUBSTEPS, POOLED
from neurogarden.dojo.env import NeuroGardenEnv
from neurogarden.dojo.stats import FITNESSES, EpisodeStats, fitness_lifespan
from neurogarden.engine import maps
from neurogarden.engine.config import Config
from neurogarden.engine.tiles import parse_map

Fitness = Callable[[EpisodeStats], float]
BRAINS = ("evolved", "connectome")  # what can be bred
# Carrying on from a taught brain (`dojo.distil`) takes small steps: at the usual sigma every
# perturbation wrecks the taught readout and one generation undoes the lesson (measured on the
# real graph: the centre's fitness fell from 4245 to 1575). The learning rate goes with the
# square of sigma, so a step stays about a third of a perturbation.
FINE_SIGMA = 0.01
FINE_LEARNING_RATE = 0.0005


@dataclass(frozen=True)
class EvolveConfig:
    generations: int = 40
    population: int = 32  # perturbations per generation, in mirrored pairs (even)
    sigma: float = 0.1  # how far a perturbation reaches
    learning_rate: float = 0.05
    hidden: int = DEFAULT_HIDDEN
    episodes: int = 2  # lives per candidate, on the generation's seeds
    max_steps: int = 2400  # two days: long enough to tell foragers from wanderers
    seed: int = 0
    workers: int | None = None  # processes for the evaluation; 0 or 1 = in this process
    map: str = "drosoville"
    fitness: str = "forager"  # see dojo.stats.FITNESSES; lifespan is what the garden ranks by
    init_scale: float = 1.0  # spread of the starting weights, relative to 1/sqrt(fan-in)
    temperature: float = 0.5  # softmax temperature the brains act at, evolving and after
    brain: str = "evolved"  # what is bred: the small network, or the parts around a connectome
    graph: str = "central"  # connectome only: which cached graph
    min_synapses: int = DEFAULT_MIN_SYNAPSES  # connectome only: connections weaker are dropped
    control: int | None = None  # connectome only: breed on the row-shuffled graph of this seed
    substeps: int = DEFAULT_SUBSTEPS  # connectome only: network updates per world tick
    pooled: int = POOLED  # connectome only: features the readout sees

    def __post_init__(self) -> None:
        if self.population < 2 or self.population % 2:
            raise ValueError("population must be an even number of at least 2")
        if min(self.generations, self.episodes, self.max_steps, self.hidden) < 1:
            raise ValueError("generations, episodes, max_steps and hidden must be positive")
        if self.sigma <= 0 or self.learning_rate <= 0 or self.init_scale <= 0:
            raise ValueError("sigma, learning_rate and init_scale must be positive")
        if self.workers is not None and self.workers < 0:
            raise ValueError("workers must be None (all cores), 0 or 1 (in-process), or more")
        if self.map not in maps.available():
            parse_map(
                self.map
            )  # raw map text is fine; anything else is refused here, not in a worker
        if self.fitness not in FITNESSES:
            raise ValueError(f"unknown fitness {self.fitness!r}; choose from {sorted(FITNESSES)}")
        if self.temperature < 0:
            raise ValueError("temperature must be 0 (always the highest score) or positive")
        if self.brain not in BRAINS:
            raise ValueError(f"unknown brain {self.brain!r}; choose from {list(BRAINS)}")
        if self.graph not in VARIANTS or min(self.min_synapses, self.substeps, self.pooled) < 1:
            raise ValueError(
                f"graph must be one of {list(VARIANTS)}; min_synapses, substeps and pooled "
                "at least 1"
            )

    def trainable(self):
        """The thing this run breeds: how big its vector is and how a vector becomes a brain."""
        if self.brain == "connectome":
            return ConnectomeTrainable(
                graph=self.graph,
                min_synapses=self.min_synapses,
                control=self.control,
                substeps=self.substeps,
                temperature=self.temperature,
                pooled=self.pooled,
            )
        return MlpTrainable(hidden=self.hidden, temperature=self.temperature)


@dataclass
class Generation:
    index: int
    best: float  # best mean fitness among the perturbations
    mean: float  # mean over the perturbations
    centre: float  # the unperturbed weights, on the same seeds
    seconds: float


_META_FIELDS = (
    "population", "sigma", "learning_rate", "episodes", "max_steps", "seed", "fitness",
    "temperature", "init_scale", "map",
)  # fmt: skip


@dataclass
class Evolved:
    genome: object  # the trainable's genome: `save(path, meta)` writes the brain file
    config: EvolveConfig
    history: list[Generation] = field(default_factory=list)
    parent: dict | None = None  # the meta of the weights this run carried on from

    @property
    def meta(self) -> dict:
        """What goes into the file: how these weights came to be, ancestors included."""
        last = self.history[-1] if self.history else None
        settings = asdict(self.config)
        inherited = 0 if self.parent is None else int(self.parent.get("total_generations", 0))
        return {
            "generations": len(self.history),
            "total_generations": inherited + len(self.history),
            **{name: settings[name] for name in _META_FIELDS},
            **self.config.trainable().describe(),
            # the centre as it was evaluated in the last generation, one step before these
            # weights: the saved weights themselves are never evaluated
            "last_centre_fitness": None if last is None else last.centre,
            "parent": self.parent,
        }


# --- evaluation --------------------------------------------------------------------------------

_env: NeuroGardenEnv | None = None
_trainable = None
_fitness: Fitness = fitness_lifespan


def _setup(config: EvolveConfig) -> None:
    """Worker initialiser: one environment and one trainable (its graph, if any) per process."""
    global _env, _trainable, _fitness
    _env = NeuroGardenEnv(map=config.map, config=Config(), max_steps=config.max_steps)
    _trainable = config.trainable()
    _fitness = FITNESSES[config.fitness]


def live(
    brain,
    env: NeuroGardenEnv,
    seed: int,
    fitness: Fitness = fitness_lifespan,
    temperature: float = 0.0,
) -> float:
    """One life: the fitness of this brain on this world seed.

    A small network's `Genome` is accepted in place of a brain (and flown at `temperature`).
    """
    if isinstance(brain, Genome):
        brain = EvolvedBrain(genome=brain, temperature=temperature)
    observation, info = env.reset(seed=seed)
    brain.reset(brain_seed(seed))
    while True:
        observation, _, terminated, truncated, info = env.step(brain.act(observation))
        if terminated or truncated:
            return fitness(info["stats"])


def _evaluate(task: tuple[np.ndarray, tuple[int, ...]]) -> float:
    vector, seeds = task
    brain = _trainable.brain(vector)
    return float(np.mean([live(brain, _env, seed, _fitness) for seed in seeds]))


def _rank_normalise(values: np.ndarray) -> np.ndarray:
    """Ranks mapped to [-0.5, 0.5]: the update cares who lived better, not by how much.

    Ties share the average of their ranks, so a mirrored pair that scored the same pulls
    nowhere; a generation where everyone scored the same steps nowhere at all.
    """
    if values.max() == values.min():
        return np.zeros(len(values), np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), np.float32)
    at = 0
    while at < len(order):
        end = at
        while end + 1 < len(order) and values[order[end + 1]] == values[order[at]]:
            end += 1
        ranks[order[at : end + 1]] = (at + end) / 2  # the average rank of the tied run
        at = end + 1
    return ranks / (len(values) - 1) - 0.5


def initial_genome(hidden: int, rng: np.random.Generator, scale: float = 1.0) -> Genome:
    """A small network's random start (see `MlpTrainable.initial`)."""
    trainable = MlpTrainable(hidden=hidden)
    return trainable.genome(trainable.initial(rng, scale))


# --- the strategy --------------------------------------------------------------------------


def evolve(
    config: EvolveConfig | None = None,
    start=None,
    on_generation: Callable[[Generation], None] | None = None,
    parent: dict | None = None,
    checkpoint: Callable[[Evolved], None] | None = None,
) -> Evolved:
    """Run the strategy; returns the weights at the centre after the last generation.

    `start` carries on from earlier weights (`parent` is their meta, kept in the file).
    `checkpoint(result)` is called after every generation with the weights so far, so a
    long run that is cut short has not been for nothing.
    """
    config = config if config is not None else EvolveConfig()
    # A run that carries on draws on from where its ancestors stopped: the same seed again
    # would breed on the same worlds and the same noise as its first generations did — or,
    # after a lesson (`dojo.distil`), on the very worlds the lesson was flown in.
    behind = 0 if parent is None else int(parent.get("total_generations", 0))
    taught = 0 if parent is None else int((parent.get("distilled") or {}).get("rounds_done", 0))
    carried = [config.seed, behind, taught] if behind or taught else config.seed
    rng = np.random.default_rng(carried)
    trainable = config.trainable()
    trainable.ready()  # refused here, not in every worker
    if start is None:  # all-zero weights would idle every fly to death: start somewhere
        theta = trainable.initial(rng, config.init_scale)
    else:
        trainable.check_start(start, parent)
        theta = start.to_vector()
    result = Evolved(trainable.genome(theta), config=config, parent=parent)
    half = config.population // 2

    def run(evaluate) -> None:
        nonlocal theta
        for index in range(config.generations):
            started = time.perf_counter()
            seeds = tuple(int(s) for s in rng.integers(0, 2**31 - 1, size=config.episodes))
            epsilon = rng.standard_normal((half, theta.size)).astype(np.float32)
            epsilon = np.concatenate([epsilon, -epsilon])  # mirrored: less noise per pair
            candidates = [(theta + config.sigma * e, seeds) for e in epsilon]
            scored = np.array(list(evaluate([*candidates, (theta, seeds)])), np.float32)
            centre, scored = float(scored[-1]), scored[:-1]
            weights = _rank_normalise(scored)
            gradient = (weights @ epsilon) / (config.population * config.sigma)
            theta = (theta + config.learning_rate * gradient).astype(np.float32)
            result.genome = trainable.genome(theta)
            generation = Generation(
                index=index,
                best=float(scored.max()),
                mean=float(scored.mean()),
                centre=centre,
                seconds=time.perf_counter() - started,
            )
            result.history.append(generation)
            if on_generation is not None:
                on_generation(generation)
            if checkpoint is not None:
                checkpoint(result)

    workers = config.workers
    if workers is not None and workers <= 1:
        _setup(config)
        run(lambda tasks: map(_evaluate, tasks))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_setup, initargs=(config,)
        ) as pool:
            run(lambda tasks: pool.map(_evaluate, tasks))
    return result
