"""The connectome brain: real wiring, modelled dynamics, a learned way in and out.

The 25 tiny features drive fixed groups of neurons (mostly sensory), a leaky rate network runs on
the MaleCNS wiring for a few updates per tick, and a learned readout turns the pooled
activity of the descending neurons into one of seven actions. What evolution finds is 481
numbers — input gains, the network's gain, the readout; the wiring is never trained. The
same brain on a row-shuffled graph is the control. A model inspired by real wiring, not a
simulated fly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neurogarden.connectome import data
from neurogarden.connectome.model import (
    DEFAULT_LEAK,
    DEFAULT_SUBSTEPS,
    MODEL_VERSION,
    POOLED,
    Encoding,
    Projection,
    RateNetwork,
    Wiring,
    make_network,
)
from neurogarden.dojo.features import DEFAULT_AGE_SCALE, FEATURES_VERSION, TINY_SIZE, tiny_features
from neurogarden.engine.body import Action
from neurogarden.engine.rng import SplitMix64

from .evolved import choose, default_weights, read_weights, sparkline, weights_meta

DEFAULT_WEIGHTS = "connectome-v1.npz"
CONTROL_WEIGHTS = "connectome-random-v1.npz"
READOUT_GAIN = 10.0  # pooled descending activity is small; this puts the first scores near 1
INITIAL_GAIN = 3.0  # the network gain evolution starts from: enough to carry a smell through
THOUGHT_NEURONS = 16
_ACTIONS = len(Action)

_pieces: dict[tuple, tuple[Encoding, Projection]] = {}  # per graph and seeds, shared by brains


def pieces(wiring: Wiring, encoding_seed: int, projection_seed: int):
    key = (wiring.digest, encoding_seed, projection_seed)
    if key not in _pieces:
        _pieces[key] = (
            Encoding.for_wiring(wiring, encoding_seed),
            Projection.for_wiring(wiring, projection_seed),
        )
    return _pieces[key]


@dataclass
class ConnectomeGenome:
    """What is learned around the wiring: 25 input gains, the network gain, the readout."""

    gains: np.ndarray  # (TINY_SIZE,) how hard each feature drives its neurons
    log_gain: np.ndarray  # (1,) the network gain, as its logarithm so it stays positive
    w: np.ndarray  # (POOLED, actions)
    b: np.ndarray  # (actions,)

    @classmethod
    def zeros(cls) -> ConnectomeGenome:
        return cls(
            gains=np.zeros(TINY_SIZE, np.float32),
            log_gain=np.zeros(1, np.float32),
            w=np.zeros((POOLED, _ACTIONS), np.float32),
            b=np.zeros(_ACTIONS, np.float32),
        )

    @property
    def size(self) -> int:
        return sum(part.size for part in self.parts())

    @property
    def gain(self) -> float:
        return float(np.exp(self.log_gain[0]))

    def parts(self) -> tuple[np.ndarray, ...]:
        return (self.gains, self.log_gain, self.w, self.b)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([part.ravel() for part in self.parts()]).astype(np.float32)

    def with_vector(self, vector: np.ndarray) -> ConnectomeGenome:
        if vector.shape != (self.size,):
            raise ValueError(f"expected {self.size} weights, got {vector.shape}")
        out, at = [], 0
        for part in self.parts():
            out.append(vector[at : at + part.size].reshape(part.shape).astype(np.float32))
            at += part.size
        return ConnectomeGenome(*out)

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        """An .npz with the learned arrays and a `meta` JSON string: which graph, how bred."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        stamp = {"features_version": FEATURES_VERSION, "brain": "connectome", **(meta or {})}
        np.savez(
            path,
            gains=self.gains,
            log_gain=self.log_gain,
            w=self.w,
            b=self.b,
            meta=json.dumps(stamp),
        )
        return path

    @classmethod
    def load(cls, source) -> tuple[ConnectomeGenome, dict]:
        """From a path or a packaged resource. ValueError for anything that is not one."""
        genome, meta = read_weights(source, "a connectome brain's weights", cls._unpack)
        if meta.get("model_version", MODEL_VERSION) != MODEL_VERSION:
            raise ValueError(
                f"{getattr(source, 'name', source)}: bred for connectome model "
                f"{meta['model_version']}, this is {MODEL_VERSION} (the senses or the readout "
                "changed since)"
            )
        return genome, meta

    @classmethod
    def _unpack(cls, file) -> tuple[ConnectomeGenome, dict]:
        meta = weights_meta(file)
        names = ("gains", "log_gain", "w", "b")
        genome = cls(*(np.asarray(file[part], dtype=np.float32) for part in names))
        for part, shape in zip(genome.parts(), cls.zeros().parts(), strict=True):
            if part.shape != shape.shape:
                raise ValueError(f"a {part.shape} array where {shape.shape} was expected")
        return genome, meta


class ConnectomeBrain:
    """A fly thinking with a wiring. Give it weights (a file names its graph, which must be in
    the cache), or a genome and a wiring directly.

    The network's state persists across ticks within a life — this brain has a memory the
    small evolved network lacks — and is cleared by `reset`.
    """

    weights = DEFAULT_WEIGHTS

    def __init__(
        self,
        seed: int = 0,
        path: str | Path | None = None,
        genome: ConnectomeGenome | None = None,
        wiring: Wiring | None = None,
        temperature: float | None = None,
        substeps: int | None = None,
        leak: float | None = None,
        encoding_seed: int | None = None,
        projection_seed: int | None = None,
        backend: str = "numpy",
    ) -> None:
        self._seed = seed
        if genome is not None:
            if wiring is None:
                raise ValueError("a genome needs the wiring it was bred on")
            self.genome, self.meta = genome, {}
        else:
            source = path if path is not None else default_weights(self.weights)
            self.genome, self.meta = ConnectomeGenome.load(source)
            if wiring is None:
                wiring = data.load(
                    self.meta.get("graph", "central"),
                    int(self.meta.get("min_synapses", data.DEFAULT_MIN_SYNAPSES)),
                    self.meta.get("control"),
                )
            bred_on = self.meta.get("digest")
            if bred_on is not None and bred_on != wiring.digest:
                raise ValueError(
                    f"these weights were bred on graph {bred_on}, this one is {wiring.digest}: "
                    "rebuild the graph with the same settings (`neurogarden connectome build`)"
                )
        self.wiring = wiring

        def setting(given, key, default):
            return given if given is not None else self.meta.get(key, default)

        self.temperature = float(setting(temperature, "temperature", 0.0) or 0.0)
        self.network: RateNetwork = make_network(
            wiring,
            int(setting(substeps, "substeps", DEFAULT_SUBSTEPS)),
            float(setting(leak, "leak", DEFAULT_LEAK)),
            backend,
        )
        self.encoding, self.projection = pieces(
            wiring,
            int(setting(encoding_seed, "encoding_seed", 0)),
            int(setting(projection_seed, "projection_seed", 0)),
        )
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self._rng = SplitMix64(self._seed if seed is None else seed)
        self.rate = self.network.zeros()
        self._pooled = np.zeros(POOLED, np.float32)

    # The three pieces `act` is made of, so a flock can step many brains in one multiply.

    def current(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        features = tiny_features(observation, DEFAULT_AGE_SCALE)
        return self.encoding.current(features, self.genome.gains, self.wiring.n)

    def decide(self, rate: np.ndarray) -> int:
        """Keep the new state and choose an action from the descending neurons' activity."""
        self.rate = rate
        self._pooled = self.projection.pool(rate) * READOUT_GAIN
        return choose(self._pooled @ self.genome.w + self.genome.b, self.temperature, self._rng)

    def act(self, observation: dict[str, np.ndarray]) -> int:
        current = self.current(observation)
        return self.decide(self.network.step(self.rate, current, self.genome.gain))

    def thought(self) -> str:
        """Sixteen of the pooled descending features, as a sparkline."""
        return sparkline(np.tanh(self._pooled[:THOUGHT_NEURONS]))


class RandomGraphBrain(ConnectomeBrain):
    """The control: the same brain bred on a wiring whose rows were shuffled."""

    weights = CONTROL_WEIGHTS


class ConnectomeFlock:
    """Several connectome brains on one wiring, stepped together in one multiply."""

    def __init__(self, brains: list[ConnectomeBrain]) -> None:
        if not brains:
            raise ValueError("a flock needs at least one brain")
        first = brains[0]
        for brain in brains:
            same = (
                brain.wiring is first.wiring
                and brain.network.substeps == first.network.substeps
                and brain.network.leak == first.network.leak
            )
            if not same:
                raise ValueError("a flock's brains must share one wiring and its dynamics")
        self.brains = brains
        self.network = first.network

    def act(self, observations: list[dict | None]) -> list[int | None]:
        """One action per brain; None in, None out (a fly with nothing to see this tick)."""
        active = [index for index, seen in enumerate(observations) if seen is not None]
        actions: list[int | None] = [None] * len(self.brains)
        if not active:
            return actions
        brains = [self.brains[index] for index in active]
        rate = np.stack([brain.rate for brain in brains], axis=1)
        current = np.stack(
            [
                brain.current(observations[index])
                for brain, index in zip(brains, active, strict=True)
            ],
            axis=1,
        )
        gain = np.array([brain.genome.gain for brain in brains], np.float32)
        rate = self.network.step(rate, current, gain)
        for column, (brain, index) in enumerate(zip(brains, active, strict=True)):
            actions[index] = brain.decide(np.ascontiguousarray(rate[:, column]))
        return actions


@dataclass(frozen=True)
class ConnectomeTrainable:
    """What the evolution strategy needs to breed a connectome brain (picklable for workers:
    each worker loads the graph from the cache once)."""

    graph: str = "central"
    min_synapses: int = data.DEFAULT_MIN_SYNAPSES
    control: int | None = None
    substeps: int = DEFAULT_SUBSTEPS
    leak: float = DEFAULT_LEAK
    temperature: float = 0.0
    encoding_seed: int = 0
    projection_seed: int = 0
    kind = "connectome"

    @property
    def size(self) -> int:
        return ConnectomeGenome.zeros().size

    def wiring(self) -> Wiring:
        return data.load(self.graph, self.min_synapses, self.control)

    def initial(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        """Every sense heard, the gain that carries a signal, a small random readout."""
        shape = ConnectomeGenome.zeros()
        w = rng.standard_normal(shape.w.shape) * scale / np.sqrt(POOLED)
        return ConnectomeGenome(
            gains=np.ones(TINY_SIZE, np.float32),
            log_gain=np.full(1, np.log(INITIAL_GAIN), np.float32),
            w=w.astype(np.float32),
            b=shape.b,
        ).to_vector()

    def genome(self, vector: np.ndarray) -> ConnectomeGenome:
        return ConnectomeGenome.zeros().with_vector(vector)

    def brain(self, vector: np.ndarray) -> ConnectomeBrain:
        return ConnectomeBrain(
            genome=self.genome(vector),
            wiring=self.wiring(),
            temperature=self.temperature,
            substeps=self.substeps,
            leak=self.leak,
            encoding_seed=self.encoding_seed,
            projection_seed=self.projection_seed,
        )

    def check_start(self, start, parent: dict | None = None) -> None:
        """`parent` is the start's meta: weights bred on one wiring are not carried on to
        another, where the same numbers would drive and read other pathways."""
        if not isinstance(start, ConnectomeGenome):
            raise ValueError("the start is not a connectome brain's weights")
        bred_on = (parent or {}).get("digest")
        if bred_on is not None and bred_on != self.wiring().digest:
            raise ValueError(
                f"the start was bred on graph {bred_on}, this run's is {self.wiring().digest} "
                "(another graph, pruning or control): weights do not carry over between wirings"
            )

    def describe(self) -> dict:
        """What the file records so the brain finds its graph again."""
        return {
            "brain": self.kind,
            **self.wiring().describe(),
            "model_version": MODEL_VERSION,
            "substeps": self.substeps,
            "leak": self.leak,
            "encoding_seed": self.encoding_seed,
            "projection_seed": self.projection_seed,
        }
