"""Evolution in the dojo: find weights for the tiny brain by living many short lives.

A plain evolution strategy (Salimans et al. 2017): perturb the weights in mirrored pairs,
live one life per perturbation on the generation's seeds, rank the fitnesses, and step
the weights towards the perturbations that lived better. The default fitness is
`dojo.stats.fitness_forager`: lifespan weighted by wellbeing, plus a bounty per bite — a
slope to climb before the first extra tick of life is won, and a pull past the wall where
a fly drinks and rests but never eats. The hall of flies still ranks by lifespan alone.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from neurogarden.brains.base import brain_seed
from neurogarden.brains.evolved import DEFAULT_HIDDEN, EvolvedBrain, Genome
from neurogarden.dojo.env import NeuroGardenEnv
from neurogarden.dojo.features import TINY_SIZE
from neurogarden.dojo.stats import FITNESSES, EpisodeStats, fitness_lifespan
from neurogarden.engine import maps
from neurogarden.engine.config import Config
from neurogarden.engine.tiles import parse_map

Fitness = Callable[[EpisodeStats], float]


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
    genome: Genome
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
            # the centre as it was evaluated in the last generation, one step before these
            # weights: the saved weights themselves are never evaluated
            "last_centre_fitness": None if last is None else last.centre,
            "parent": self.parent,
        }


# --- evaluation --------------------------------------------------------------------------------

_env: NeuroGardenEnv | None = None
_shape: Genome | None = None
_fitness: Fitness = fitness_lifespan
_temperature = 0.0


def _setup(map_name: str, max_steps: int, hidden: int, fitness: str, temperature: float) -> None:
    """Worker initialiser: one environment and one genome shape per process."""
    global _env, _shape, _fitness, _temperature
    _env = NeuroGardenEnv(map=map_name, config=Config(), max_steps=max_steps)
    _shape = Genome.zeros(hidden)
    _fitness = FITNESSES[fitness]
    _temperature = temperature


def live(
    genome: Genome,
    env: NeuroGardenEnv,
    seed: int,
    fitness: Fitness = fitness_lifespan,
    temperature: float = 0.0,
) -> float:
    """One life: the fitness of these weights on this world seed."""
    brain = EvolvedBrain(genome=genome, temperature=temperature)
    observation, info = env.reset(seed=seed)
    brain.reset(brain_seed(seed))
    while True:
        observation, _, terminated, truncated, info = env.step(brain.act(observation))
        if terminated or truncated:
            return fitness(info["stats"])


def _evaluate(task: tuple[np.ndarray, tuple[int, ...]]) -> float:
    vector, seeds = task
    genome = _shape.with_vector(vector)
    return float(np.mean([live(genome, _env, seed, _fitness, _temperature) for seed in seeds]))


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
    """Random weights scaled by 1/sqrt(fan-in): scores that already depend on what is seen."""
    shape = Genome.zeros(hidden)
    return Genome(
        w1=(rng.standard_normal(shape.w1.shape) * scale / np.sqrt(TINY_SIZE)).astype(np.float32),
        b1=shape.b1,
        w2=(rng.standard_normal(shape.w2.shape) * scale / np.sqrt(hidden)).astype(np.float32),
        b2=shape.b2,
    )


# --- the strategy --------------------------------------------------------------------------


def evolve(
    config: EvolveConfig | None = None,
    start: Genome | None = None,
    on_generation: Callable[[Generation], None] | None = None,
    parent: dict | None = None,
) -> Evolved:
    """Run the strategy; returns the weights at the centre after the last generation.

    `start` carries on from earlier weights (`parent` is their meta, kept in the file).
    """
    config = config if config is not None else EvolveConfig()
    rng = np.random.default_rng(config.seed)
    shape = Genome.zeros(config.hidden)
    if start is None:  # all-zero weights would idle every fly to death: start somewhere
        start = initial_genome(config.hidden, rng, config.init_scale)
    elif start.hidden != config.hidden:
        raise ValueError(f"the start has {start.hidden} hidden neurons, the config {config.hidden}")
    theta = start.to_vector()
    result = Evolved(shape.with_vector(theta), config=config, parent=parent)
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
            result.genome = shape.with_vector(theta)
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

    workers = config.workers
    if workers is not None and workers <= 1:
        _setup(config.map, config.max_steps, config.hidden, config.fitness, config.temperature)
        run(lambda tasks: map(_evaluate, tasks))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_setup,
            initargs=(
                config.map,
                config.max_steps,
                config.hidden,
                config.fitness,
                config.temperature,
            ),
        ) as pool:
            run(lambda tasks: pool.map(_evaluate, tasks))
    return result
