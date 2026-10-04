"""The connectome brain: real wiring, modelled dynamics, a learned way in and out.

The 25 tiny features drive fixed groups of neurons (mostly sensory), a leaky rate network runs on
the MaleCNS wiring for a few updates per tick, and a learned readout turns the pooled
activity of the descending neurons into one of seven actions. What is learned is 481
numbers — input gains, the network's gain, the readout — taught by the evolved brain
(`dojo.distil`) or bred (`dojo.evolve`); the wiring is never trained. The same brain on a
row-shuffled graph is the control. A model inspired by real wiring, not a simulated fly.
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
# The network gain stays below 1. There every update is a contraction: each neuron's inputs
# sum to at most 1 and tanh never stretches, so two states move together by a factor of at
# most (1 - leak) + leak * gain per update, the network forgets how a life began within a few
# ticks, and its state is an echo of what the fly senses. At exactly 1 that factor is 1 and
# nothing is promised; at 1.2 and above (measured) the real wiring keeps activity of its own:
# two copies hearing the same life from different beginnings never meet, and a readout taught
# in one such regime is lost in another — a fly born at another hour, or one whose flock was
# restarted. Nothing is lost by staying below either: fitted alike, the pooled features give
# the senses back better at gain 0.5 than at 3 (the spec has both rows).
GAIN_CEILING = 0.95
INITIAL_GAIN = 0.5  # the network gain a new brain starts with
READOUT_GAIN = 500.0  # an echo is faint: this brings the pooled features to about 1
THOUGHT_NEURONS = 16
_ACTIONS = len(Action)

_pieces: dict[tuple, tuple[Encoding, Projection]] = {}  # per graph and seeds, shared by brains


def pieces(wiring: Wiring, encoding_seed: int, projection_seed: int, pooled: int = POOLED):
    key = (wiring.digest, encoding_seed, projection_seed, pooled)
    if key not in _pieces:
        _pieces[key] = (
            Encoding.for_wiring(wiring, encoding_seed),
            Projection.for_wiring(wiring, projection_seed, pooled),
        )
    return _pieces[key]


@dataclass
class ConnectomeGenome:
    """What is learned around the wiring: 25 input gains, the network gain, the readout."""

    gains: np.ndarray  # (TINY_SIZE,) how hard each feature drives its neurons
    log_gain: np.ndarray  # (1,) the network gain, as its logarithm so it stays positive
    w: np.ndarray  # (pooled, actions): its height says how many features the readout sees
    b: np.ndarray  # (actions,)

    @classmethod
    def zeros(cls, pooled: int = POOLED) -> ConnectomeGenome:
        return cls(
            gains=np.zeros(TINY_SIZE, np.float32),
            log_gain=np.zeros(1, np.float32),
            w=np.zeros((pooled, _ACTIONS), np.float32),
            b=np.zeros(_ACTIONS, np.float32),
        )

    @property
    def pooled(self) -> int:
        return self.w.shape[0]

    @property
    def size(self) -> int:
        return sum(part.size for part in self.parts())

    @property
    def gain(self) -> float:
        """The network gain, held at the ceiling whatever the number was bred to."""
        return float(min(np.exp(self.log_gain[0]), GAIN_CEILING))

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
        if not all(np.isfinite(part).all() for part in genome.parts()):
            raise ValueError("the weights are not all finite numbers")
        pooled = genome.w.shape[0] if genome.w.ndim == 2 else -1
        for part, shape in zip(genome.parts(), cls.zeros(max(pooled, 1)).parts(), strict=True):
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
            if not all(np.isfinite(part).all() for part in genome.parts()):
                raise ValueError("these weights are not all finite numbers")
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
            self.genome.pooled,
        )
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self._rng = SplitMix64(self._seed if seed is None else seed)
        self.rate = self.network.zeros()
        self._pooled = np.zeros(self.genome.pooled, np.float32)

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

    @property
    def features(self) -> np.ndarray:
        """What the readout saw at the last tick: the pooled descending activity."""
        return self._pooled

    def thought(self) -> str:
        """Sixteen of the pooled descending features, as a sparkline."""
        return sparkline(np.tanh(self._pooled[:THOUGHT_NEURONS]))


class RandomGraphBrain(ConnectomeBrain):
    """The control: the same brain, taught the same lesson on a wiring whose rows were
    shuffled."""

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
    pooled: int = POOLED  # features the descending neurons are pooled into for the readout
    gain: float = INITIAL_GAIN  # the network gain a new genome starts with
    input_gain: float = 1.0  # and how hard every sense drives its neurons, to start with
    kind = "connectome"

    @property
    def size(self) -> int:
        return ConnectomeGenome.zeros(self.pooled).size

    def wiring(self) -> Wiring:
        return data.load(self.graph, self.min_synapses, self.control)

    def ready(self) -> Wiring:
        """Load the graph and lay out the senses and the readout on it: whatever will not do
        (no graph in the cache, more pooled features than descending neurons) is refused
        here, once, before any worker process is started."""
        wiring = self.wiring()
        pieces(wiring, self.encoding_seed, self.projection_seed, self.pooled)
        return wiring

    def blank(self) -> ConnectomeGenome:
        """Every sense heard, the gain that carries a signal, and a readout that says nothing
        yet: where a brain that will be taught starts."""
        shape = ConnectomeGenome.zeros(self.pooled)
        return ConnectomeGenome(
            gains=np.full(TINY_SIZE, self.input_gain, np.float32),
            log_gain=np.full(1, np.log(self.gain), np.float32),
            w=shape.w,
            b=shape.b,
        )

    def initial(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        """The same with a small random readout: where a brain that will be bred starts."""
        genome = self.blank()
        w = rng.standard_normal(genome.w.shape) * scale / np.sqrt(self.pooled)
        genome.w[:] = w.astype(np.float32)
        return genome.to_vector()

    def genome(self, vector: np.ndarray) -> ConnectomeGenome:
        return ConnectomeGenome.zeros(self.pooled).with_vector(vector)

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
        if start.pooled != self.pooled:
            raise ValueError(
                f"the start reads {start.pooled} pooled features, this run {self.pooled}"
            )
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
            "pooled": self.pooled,
            "substeps": self.substeps,
            "leak": self.leak,
            "encoding_seed": self.encoding_seed,
            "projection_seed": self.projection_seed,
        }
