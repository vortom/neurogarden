"""Distillation: a connectome brain learns its readout from a teacher it can ask.

Evolving the numbers around a big graph from nothing is slow, and it stalls at the wall the
small network met too: drink, rest, starve. The small evolved network is already past that
wall and senses the same 25 features, so it can say, for any moment of any life, how likely
it would be to take each action. The connectome brain's readout is linear in the pooled
activity of its descending neurons, so learning from those answers is a softmax regression:

  round 1    the teacher flies and the connectome network listens (its state depends only on
             what the fly senses); the readout is fitted to the teacher's probabilities
  round 2+   the student flies its own lives, the teacher says what it would have done in
             every moment the student met, and the readout is fitted again on everything so
             far — a student learns to recover from its own mistakes only by making them

Every life of a lesson begins at some hour of the day, not at dawn (`any_hour`): a fly in a
live garden hatches whenever its owner joins.

The wiring is never trained, and neither is the input side here (the 25 gains and the network
gain keep their starting values). What comes out is an ordinary `ConnectomeGenome`: a brain
to fly, or a start for `evolve --start`, which also breeds the input side.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from neurogarden.brains.base import brain_seed
from neurogarden.brains.connectome import (
    GAIN_CEILING,
    INITIAL_GAIN,
    ConnectomeGenome,
    ConnectomeTrainable,
)
from neurogarden.brains.evolved import EvolvedBrain
from neurogarden.connectome.data import DEFAULT_MIN_SYNAPSES, VARIANTS
from neurogarden.connectome.model import DEFAULT_SUBSTEPS, POOLED, require_scipy
from neurogarden.dojo.env import NeuroGardenEnv
from neurogarden.engine import maps
from neurogarden.engine.config import Config
from neurogarden.engine.tiles import parse_map


@dataclass(frozen=True)
class DistilConfig:
    rounds: int = 4  # fits: the first on the teacher's lives, the others on the student's own
    lives: int = 4  # lives flown per round
    max_steps: int = 2400
    seed: int = 0
    l2: float = 1e-4  # holds the readout's weights, each in units of its feature's spread
    workers: int | None = None  # processes flying the lives; 0 or 1 = in this process
    map: str = "drosoville"
    temperature: float = 0.5  # the student's: it draws from the softmax of its scores at this
    teacher: str | None = None  # an evolved brain's weights; None = the shipped ones
    graph: str = "central"
    min_synapses: int = DEFAULT_MIN_SYNAPSES
    control: int | None = None  # learn on the row-shuffled graph of this seed
    substeps: int = DEFAULT_SUBSTEPS
    pooled: int = POOLED  # features the readout sees
    # The input side is not learned here; these set it (evolution can move it afterwards).
    gain: float = INITIAL_GAIN
    input_gain: float = 1.0
    # Lives begin at any hour of the day, as they do in a live garden: a lesson should show
    # the student the lives it is going to live, not only those that begin at dawn.
    any_hour: bool = True

    def __post_init__(self) -> None:
        counts = (self.rounds, self.lives, self.max_steps, self.min_synapses, self.substeps)
        if min(*counts, self.pooled) < 1:
            raise ValueError(
                "rounds, lives, max_steps, min_synapses, substeps and pooled must be positive"
            )
        if self.l2 < 0 or min(self.temperature, self.gain, self.input_gain) <= 0:
            raise ValueError(
                "l2 must not be negative; temperature, gain and input_gain must be positive"
            )
        if self.gain > GAIN_CEILING:
            raise ValueError(
                f"gain must be at most {GAIN_CEILING}: from 1 up the network may keep activity "
                "of its own and stops forgetting how a life began"
            )
        if self.workers is not None and self.workers < 0:
            raise ValueError("workers must be None (all cores), 0 or 1 (in-process), or more")
        if self.graph not in VARIANTS:
            raise ValueError(f"graph must be one of {list(VARIANTS)}")
        if self.map not in maps.available():
            parse_map(self.map)  # raw map text is fine; anything else is refused here

    def trainable(self) -> ConnectomeTrainable:
        return ConnectomeTrainable(
            graph=self.graph,
            min_synapses=self.min_synapses,
            control=self.control,
            substeps=self.substeps,
            temperature=self.temperature,
            pooled=self.pooled,
            gain=self.gain,
            input_gain=self.input_gain,
        )


@dataclass
class Round:
    index: int
    flown_by: str  # "teacher" or "student"
    lifespans: list[int]  # of the lives flown this round
    ticks: int  # labelled moments so far, this round's included
    agreement: float  # how often the fitted student's favourite action is the teacher's
    divergence: float  # mean KL(teacher || student) over the moments, in nats
    settled: bool  # False: the fit ran out of iterations before it stopped improving
    seconds: float


_FIT_STEPS = 2000  # iterations a fit may take; a round reports whether it settled within them
_META_FIELDS = ("rounds", "lives", "max_steps", "seed", "l2", "map", "gain", "input_gain")


@dataclass
class Distilled:
    genome: ConnectomeGenome
    config: DistilConfig
    history: list[Round] = field(default_factory=list)
    teacher: dict = field(default_factory=dict)  # the teacher's own meta
    teacher_digest: str = ""  # a hash of the teacher's weights: which teacher, whatever its name

    @property
    def meta(self) -> dict:
        """What goes into the file: an ordinary connectome brain's meta, and how it learned."""
        settings = asdict(self.config)
        last = self.history[-1] if self.history else None
        return {
            **self.config.trainable().describe(),
            "temperature": self.config.temperature,
            "generations": 0,
            "total_generations": 0,  # none bred yet: `evolve --start` counts from here
            "parent": None,
            "any_hour": self.config.any_hour,  # `evolve --start` breeds on as it was taught
            "distilled": {
                **{name: settings[name] for name in _META_FIELDS},
                "rounds_done": len(self.history),
                "teacher": Path(self.config.teacher).name if self.config.teacher else "evolved-v1",
                "teacher_digest": self.teacher_digest,
                "teacher_generations": self.teacher.get("total_generations"),
                "ticks": 0 if last is None else last.ticks,
                "agreement": None if last is None else last.agreement,
                "divergence": None if last is None else last.divergence,
            },
        }


# --- flying a life and writing down what the teacher would have done ---------------------------

_env: NeuroGardenEnv | None = None
_trainable: ConnectomeTrainable | None = None
_teacher: EvolvedBrain | None = None


def _setup(config: DistilConfig) -> None:
    """Worker initialiser: one environment, one graph and one teacher per process."""
    global _env, _trainable, _teacher
    _env = NeuroGardenEnv(
        map=config.map, config=Config(), max_steps=config.max_steps, any_hour=config.any_hour
    )
    _trainable = config.trainable()
    _teacher = EvolvedBrain(path=config.teacher)


def _fly(task: tuple[np.ndarray, int, bool]) -> tuple[np.ndarray, np.ndarray, int]:
    """One life on `seed`, flown by the teacher or by the student built from `vector`.

    Either way the student's network senses every tick and the teacher labels it. Returns
    what the readout saw per tick, the teacher's probabilities per tick, and the lifespan.
    """
    vector, seed, teacher_flies = task
    student = _trainable.brain(vector)
    observation, info = _env.reset(seed=seed)
    student.reset(brain_seed(seed))
    _teacher.reset(brain_seed(seed))
    seen, said = [], []
    while True:
        said.append(_teacher.probabilities(observation))
        action = student.act(observation)  # moves the network on, whoever flies
        seen.append(student.features)
        if teacher_flies:
            action = _teacher.act(observation)
        observation, _, terminated, truncated, info = _env.step(action)
        if terminated or truncated:
            lifespan = info["stats"].lifespan
            return np.array(seen, np.float32), np.array(said, np.float32), lifespan


def fit_readout(
    features: np.ndarray, targets: np.ndarray, l2: float
) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
    """Softmax regression to soft targets: `(w, b, agreement, divergence, settled)`.

    `features @ w + b` are log-probabilities up to a constant: the student that draws from
    their softmax is as close to the teacher's probabilities as a linear readout gets.
    `l2` holds both `w` and `b`: without it, the bias of an action a sure teacher never
    takes would run off towards minus infinity. `settled` is False when the fit ran out of
    iterations; agreement and divergence are measured on the moments it was fitted to.

    The fit is made on standardised features — each in units of its own spread — and handed
    back for the features as they are: some pooled features move tens of times more than
    others, and a penalty on raw weights would silence the quiet ones.
    """
    require_scipy()
    from scipy.optimize import minimize

    raw = np.asarray(features, np.float64)
    centre, spread = raw.mean(axis=0), raw.std(axis=0)
    # A feature that does not move says nothing — and one that moves only in its last digits
    # is rounding, which standardising would blow up into a weight no float32 could carry.
    still = spread <= 1e-6 * np.maximum(1.0, np.abs(centre))
    spread = np.where(still, 1.0, spread)
    x = np.where(still, 0.0, (raw - centre) / spread)
    p = np.asarray(targets, np.float64)
    ticks, width = x.shape
    actions = p.shape[1]

    def log_probabilities(flat: np.ndarray) -> np.ndarray:
        w, b = flat[: width * actions].reshape(width, actions), flat[width * actions :]
        scores = x @ w + b
        scores -= scores.max(axis=1, keepdims=True)
        return scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))

    def loss(flat: np.ndarray) -> tuple[float, np.ndarray]:
        log_p = log_probabilities(flat)
        slope = (np.exp(log_p) - p) / ticks
        value = -(p * log_p).sum() / ticks + l2 * (flat**2).sum()
        return value, np.concatenate([(x.T @ slope).ravel(), slope.sum(axis=0)]) + 2 * l2 * flat

    start = np.zeros(width * actions + actions)
    found = minimize(loss, start, jac=True, method="L-BFGS-B", options={"maxiter": _FIT_STEPS})
    log_p = log_probabilities(found.x)
    agreement = float((log_p.argmax(axis=1) == p.argmax(axis=1)).mean())
    divergence = float((p * (np.log(p + 1e-12) - log_p)).sum(axis=1).mean())
    w, b = found.x[: width * actions].reshape(width, actions), found.x[width * actions :]
    w = w / spread[:, None]  # back to the features as the brain will see them
    b = b - centre @ w
    return w, b, agreement, divergence, bool(found.success)


# --- the rounds --------------------------------------------------------------------------------


def distil(
    config: DistilConfig | None = None,
    on_round: Callable[[Round], None] | None = None,
    checkpoint: Callable[[Distilled], None] | None = None,
) -> Distilled:
    """Teach a connectome brain its readout; returns the student after the last round.

    `checkpoint(result)` is called after every round with the student so far.
    """
    config = config if config is not None else DistilConfig()
    rng = np.random.default_rng(config.seed)
    trainable = config.trainable()
    trainable.ready()  # a graph or a width that will not do is refused here,
    teacher = EvolvedBrain(path=config.teacher)  # and so is a teacher: not in a worker
    # The first student knows nothing: round 1 is flown by the teacher anyway.
    result = Distilled(
        trainable.blank(),
        config=config,
        teacher=dict(teacher.meta),
        teacher_digest=hashlib.sha256(teacher.genome.to_vector().tobytes()).hexdigest()[:16],
    )
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    def run(fly) -> None:
        for index in range(config.rounds):
            started = time.perf_counter()
            seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=config.lives)]
            vector = result.genome.to_vector()
            lives = list(fly([(vector, seed, index == 0) for seed in seeds]))
            features.extend(seen for seen, _, _ in lives)
            targets.extend(said for _, said, _ in lives)
            w, b, agreement, divergence, settled = fit_readout(
                np.concatenate(features), np.concatenate(targets), config.l2
            )
            # The brain draws from softmax(scores / temperature): scale the fit to match.
            result.genome = ConnectomeGenome(
                gains=result.genome.gains,
                log_gain=result.genome.log_gain,
                w=(w * config.temperature).astype(np.float32),
                b=(b * config.temperature).astype(np.float32),
            )
            entry = Round(
                index=index,
                flown_by="teacher" if index == 0 else "student",
                lifespans=[lifespan for _, _, lifespan in lives],
                ticks=sum(len(said) for said in targets),
                agreement=agreement,
                divergence=divergence,
                settled=settled,
                seconds=time.perf_counter() - started,
            )
            result.history.append(entry)
            if on_round is not None:
                on_round(entry)
            if checkpoint is not None:
                checkpoint(result)

    workers = config.workers
    if workers is not None and workers <= 1:
        _setup(config)
        run(lambda tasks: map(_fly, tasks))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_setup, initargs=(config,)
        ) as pool:
            run(lambda tasks: pool.map(_fly, tasks))
    return result
