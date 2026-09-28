"""Evolution in the dojo: find weights for the tiny brain by living many short lives.

A plain evolution strategy (Salimans et al. 2017): perturb the weights in mirrored pairs,
live one life per perturbation on the generation's seeds, rank the fitnesses, and step
the weights towards the perturbations that lived better. The default fitness is the
lifespan weighted by wellbeing (`dojo.stats.fitness_wellbeing`): a slope to climb before
the first extra tick of life is won. The hall of flies still ranks by lifespan alone.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from neurogarden.brains.base import brain_seed
from neurogarden.brains.evolved import DEFAULT_HIDDEN, EvolvedBrain, Genome
from neurogarden.dojo.env import NeuroGardenEnv
from neurogarden.dojo.features import TINY_SIZE
from neurogarden.dojo.stats import FITNESSES, EpisodeStats, fitness_lifespan
from neurogarden.engine.config import Config

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
    fitness: str = "wellbeing"  # see dojo.stats.FITNESSES; lifespan is what the garden ranks by
    init_scale: float = 1.0  # spread of the starting weights, relative to 1/sqrt(fan-in)
    temperature: float = 0.5  # softmax temperature the brains act at, evolving and after

    def __post_init__(self) -> None:
        if self.population < 2 or self.population % 2:
            raise ValueError("population must be an even number of at least 2")
        if self.generations < 0 or self.episodes < 1 or self.max_steps < 1:
            raise ValueError("generations, episodes and max_steps must be positive")
        if self.sigma <= 0 or self.learning_rate <= 0:
            raise ValueError("sigma and learning_rate must be positive")
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


@dataclass
class Evolved:
    genome: Genome
    history: list[Generation] = field(default_factory=list)
    config: EvolveConfig | None = None

    @property
    def meta(self) -> dict:
        last = self.history[-1] if self.history else None
        return {
            "generations": len(self.history),
            "population": self.config.population if self.config else None,
            "sigma": self.config.sigma if self.config else None,
            "learning_rate": self.config.learning_rate if self.config else None,
            "episodes": self.config.episodes if self.config else None,
            "max_steps": self.config.max_steps if self.config else None,
            "seed": self.config.seed if self.config else None,
            "fitness": self.config.fitness if self.config else None,
            "temperature": self.config.temperature if self.config else None,
            "final_centre_fitness": None if last is None else last.centre,
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

    A generation where everyone scored the same has nothing to say: all zeros, no step.
    """
    if values.max() == values.min():
        return np.zeros(len(values), np.float32)
    ranks = np.empty(len(values), np.float32)
    ranks[np.argsort(values, kind="stable")] = np.arange(len(values), dtype=np.float32)
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
) -> Evolved:
    """Run the strategy; returns the weights at the centre after the last generation."""
    config = config if config is not None else EvolveConfig()
    rng = np.random.default_rng(config.seed)
    shape = Genome.zeros(config.hidden)
    if start is None:  # all-zero weights would idle every fly to death: start somewhere
        start = initial_genome(config.hidden, rng, config.init_scale)
    theta = start.to_vector()
    result = Evolved(shape.with_vector(theta), config=config)
    half = config.population // 2

    def run(evaluate) -> None:
        nonlocal theta
        for index in range(config.generations):
            started = time.perf_counter()
            seeds = tuple(int(s) for s in rng.integers(0, 2**31 - 1, size=config.episodes))
            epsilon = rng.standard_normal((half, theta.size)).astype(np.float32)
            epsilon = np.concatenate([epsilon, -epsilon])  # mirrored: less noise per pair
            candidates = [(theta + config.sigma * e, seeds) for e in epsilon]
            fitness = np.array(list(evaluate([*candidates, (theta, seeds)])), np.float32)
            centre, fitness = float(fitness[-1]), fitness[:-1]
            weights = _rank_normalise(fitness)
            gradient = (weights @ epsilon) / (config.population * config.sigma)
            theta = (theta + config.learning_rate * gradient).astype(np.float32)
            result.genome = shape.with_vector(theta)
            generation = Generation(
                index=index,
                best=float(fitness.max()),
                mean=float(fitness.mean()),
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
