"""The evolved brain: a tiny network whose weights were found by evolution in the dojo.

Twenty-five numbers in (the tiny features), a hidden layer of a few neurons, seven action
scores out. With a temperature the action is drawn from the softmax of the scores (the way
it was evolved: a policy that always does the same thing gives evolution nothing to rank);
without one, the highest score wins. No rules, no memory: everything it knows about living
in Drosoville is in the weights, and the weights came from lifespans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from neurogarden.dojo.features import DEFAULT_AGE_SCALE, FEATURES_VERSION, TINY_SIZE, tiny_features
from neurogarden.engine.body import Action
from neurogarden.engine.rng import SplitMix64

DEFAULT_HIDDEN = 16
DEFAULT_WEIGHTS = "evolved-v1.npz"  # shipped in neurogarden/brains/weights/
_SPARKS = "▁▂▃▄▅▆▇█"


@dataclass
class Genome:
    """The weights of one brain, as arrays (to think with) or as one vector (to evolve)."""

    w1: np.ndarray  # (TINY_SIZE, hidden)
    b1: np.ndarray  # (hidden,)
    w2: np.ndarray  # (hidden, actions)
    b2: np.ndarray  # (actions,)

    @classmethod
    def zeros(cls, hidden: int = DEFAULT_HIDDEN, actions: int = len(Action)) -> Genome:
        return cls(
            w1=np.zeros((TINY_SIZE, hidden), np.float32),
            b1=np.zeros(hidden, np.float32),
            w2=np.zeros((hidden, actions), np.float32),
            b2=np.zeros(actions, np.float32),
        )

    @property
    def hidden(self) -> int:
        return self.b1.shape[0]

    @property
    def size(self) -> int:
        return sum(part.size for part in self.parts())

    def parts(self) -> tuple[np.ndarray, ...]:
        return (self.w1, self.b1, self.w2, self.b2)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([part.ravel() for part in self.parts()]).astype(np.float32)

    def with_vector(self, vector: np.ndarray) -> Genome:
        """A genome of the same shape carrying these weights."""
        if vector.shape != (self.size,):
            raise ValueError(f"expected {self.size} weights, got {vector.shape}")
        out, at = [], 0
        for part in self.parts():
            out.append(vector[at : at + part.size].reshape(part.shape).astype(np.float32))
            at += part.size
        return Genome(*out)

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        """An .npz with the four arrays and a `meta` JSON string (how it came to be).

        numpy appends `.npz` to a name without it; the path written is returned, so nobody
        is told a file exists under a name it does not.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        stamp = {"features_version": FEATURES_VERSION, "hidden": self.hidden, **(meta or {})}
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, meta=json.dumps(stamp))
        return path

    @classmethod
    def load(cls, source) -> tuple[Genome, dict]:
        """From a path, or anything with `open("rb")` (a packaged resource). ValueError for
        a file that is not a brain of this shape."""
        name = str(source) if isinstance(source, str | Path) else getattr(source, "name", source)
        try:
            if hasattr(source, "open") and not isinstance(source, Path):
                with source.open("rb") as handle, np.load(handle) as data:
                    genome, meta = cls._unpack(data, name)
            else:
                with np.load(source) as data:
                    genome, meta = cls._unpack(data, name)
        except (KeyError, ValueError, TypeError, OSError) as err:
            if isinstance(err, FileNotFoundError):
                raise
            raise ValueError(f"{name}: not an evolved brain's weights ({err})") from None
        if meta.get("features_version", FEATURES_VERSION) != FEATURES_VERSION:
            raise ValueError(f"{name}: features_version {meta['features_version']} is not ours")
        return genome, meta

    @classmethod
    def _unpack(cls, data, name: str) -> tuple[Genome, dict]:
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
        if not isinstance(meta, dict):
            raise ValueError("meta is not an object")
        genome = cls(
            *(np.asarray(data[part], dtype=np.float32) for part in ("w1", "b1", "w2", "b2"))
        )
        hidden = genome.w1.shape[1] if genome.w1.ndim == 2 else -1
        expected = Genome.zeros(max(hidden, 1))
        for part, shape in zip(genome.parts(), expected.parts(), strict=True):
            if part.shape != shape.shape:
                raise ValueError(f"a {part.shape} array where {shape.shape} was expected")
        if meta.get("hidden", hidden) != hidden:
            raise ValueError(f"meta says {meta['hidden']} hidden neurons, the arrays {hidden}")
        return genome, meta


def hidden_activation(genome: Genome, features: np.ndarray) -> np.ndarray:
    return np.tanh(features @ genome.w1 + genome.b1)


def output_scores(genome: Genome, hidden: np.ndarray) -> np.ndarray:
    """One score per action from the hidden layer; the policy acts on these."""
    return hidden @ genome.w2 + genome.b2


def scores(genome: Genome, features: np.ndarray) -> np.ndarray:
    """One score per action from what is seen: the whole network in one call."""
    return output_scores(genome, hidden_activation(genome, features))


def default_weights():
    """The shipped weights as a packaged resource: readable wherever the package lives."""
    return resources.files("neurogarden.brains").joinpath("weights", DEFAULT_WEIGHTS)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - values.max())
    return shifted / shifted.sum()


def choose(values: np.ndarray, temperature: float, rng: SplitMix64) -> int:
    """The action for these scores: the highest at temperature 0, else drawn from the softmax
    with the brain's own generator, so a life is reproducible from its seed."""
    if temperature <= 0:
        return int(np.argmax(values))
    draw = rng.next_u64() / 2**64
    picked = np.searchsorted(np.cumsum(softmax(values / temperature)), draw)
    return int(min(picked, len(values) - 1))


def sparkline(values: np.ndarray) -> str:
    """Activities in [-1, 1] as one glyph each: a brain scope in a speech bubble."""
    levels = np.clip(((values + 1) / 2 * (len(_SPARKS) - 1)).round(), 0, 7).astype(int)
    return "".join(_SPARKS[level] for level in levels)


@dataclass(frozen=True)
class MlpTrainable:
    """What the evolution strategy needs to breed the small network (picklable for workers)."""

    hidden: int = DEFAULT_HIDDEN
    temperature: float = 0.0
    kind = "evolved"

    @property
    def size(self) -> int:
        return Genome.zeros(self.hidden).size

    def initial(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        """Random weights scaled by 1/sqrt(fan-in): scores that already depend on what is seen."""
        shape = Genome.zeros(self.hidden)
        w1 = rng.standard_normal(shape.w1.shape) * scale / np.sqrt(TINY_SIZE)
        w2 = rng.standard_normal(shape.w2.shape) * scale / np.sqrt(self.hidden)
        return Genome(w1.astype(np.float32), shape.b1, w2.astype(np.float32), shape.b2).to_vector()

    def genome(self, vector: np.ndarray) -> Genome:
        return Genome.zeros(self.hidden).with_vector(vector)

    def brain(self, vector: np.ndarray) -> EvolvedBrain:
        return EvolvedBrain(genome=self.genome(vector), temperature=self.temperature)

    def check_start(self, start) -> None:
        if getattr(start, "hidden", None) != self.hidden:
            raise ValueError(
                f"the start has {getattr(start, 'hidden', '?')} hidden neurons, "
                f"the config {self.hidden}"
            )

    def describe(self) -> dict:
        return {"brain": self.kind}


class EvolvedBrain:
    """Acts on the scores (drawn at `temperature`, else the highest); `thought()` is a glimpse
    of its hidden layer for spectators.

    `temperature=None` reads it from the weights file (how the brain was evolved); pass a
    number to override, 0 for the highest score always.
    """

    def __init__(
        self,
        seed: int = 0,
        path: str | Path | None = None,
        genome: Genome | None = None,
        temperature: float | None = None,
    ) -> None:
        self._seed = seed
        if genome is not None:
            self.genome, self.meta = genome, {}
        else:
            self.genome, self.meta = Genome.load(path if path is not None else default_weights())
        if temperature is None:
            temperature = float(self.meta.get("temperature") or 0.0)
        self.temperature = temperature
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self._rng = SplitMix64(self._seed if seed is None else seed)
        self._hidden = np.zeros(self.genome.hidden, np.float32)

    def act(self, observation: dict[str, np.ndarray]) -> int:
        features = tiny_features(observation, DEFAULT_AGE_SCALE)
        self._hidden = hidden_activation(self.genome, features)
        return choose(output_scores(self.genome, self._hidden), self.temperature, self._rng)

    def thought(self) -> str:
        """The hidden layer as a sparkline: a brain scope in a speech bubble."""
        return sparkline(self._hidden)
