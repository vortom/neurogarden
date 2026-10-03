"""The wiring and what runs on it: a leaky rate network, the senses in, the readout out.

Everything here is a modelling choice layered on a graph: the connectome gives the
connections; the equation, the gain, which neurons "smell fruit" and how descending
activity is pooled are ours, and the spec says so in one place.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from neurogarden.dojo.features import TINY_SIZE
from neurogarden.engine.rng import SplitMix64

DEFAULT_SUBSTEPS = 3  # updates per world tick: a smell reaches the descending neurons in 3-5
DEFAULT_LEAK = 0.5
POOLED = 64  # features the descending neurons are pooled into for the readout
GROUP_MAX = 128  # neurons one feature drives at most, so no sense shouts over the others
# What the learned numbers lean on besides the graph: which neurons each feature drives, how
# the descending neurons are pooled, the update rule. Change any of them and weights bred
# before mean something else — bump this, and old files are refused instead of misread.
# 2: the gain is held at or below 1 and the pooled features are scaled for that (1 let the
# network run at gain 3, where it kept activity of its own).
MODEL_VERSION = 2

_EXTRA = "the connectome brain needs scipy: install neurogarden[connectome]"


def require_scipy():
    try:
        import scipy.sparse as sparse
    except ImportError:
        raise ValueError(_EXTRA) from None
    return sparse


@dataclass
class Wiring:
    """A signed, row-normalised graph and what each neuron is. `matrix[post, pre]`."""

    matrix: object  # scipy.sparse.csr_matrix, float32, (n, n)
    superclass: np.ndarray  # str per neuron
    klass: np.ndarray  # str per neuron ("none" where the annotation has no class)
    variant: str = "central"
    min_synapses: int = 5
    control: int | None = None  # seed of the row shuffle, for the random-graph control
    digest: str = field(default="")
    networks: dict = field(default_factory=dict, repr=False, compare=False)  # see make_network

    def __post_init__(self) -> None:
        # One layout wherever the graph came from: every row's columns ascending. The sums
        # then add up in the same order on every machine, and torch's CSR requires it.
        if not self.matrix.has_sorted_indices:
            self.matrix = self.matrix.sorted_indices()
        if not self.digest:
            self.digest = digest_of(self.matrix, self.superclass, self.klass)

    @property
    def n(self) -> int:
        return self.matrix.shape[0]

    @property
    def connections(self) -> int:
        return int(self.matrix.nnz)

    def where(self, *, klass: tuple[str, ...] = (), superclass: tuple[str, ...] = ()) -> np.ndarray:
        """Indices of the neurons of these classes or superclasses, ascending."""
        mask = np.isin(self.klass, klass) | np.isin(self.superclass, superclass)
        return np.flatnonzero(mask)

    def randomised(self, seed: int) -> Wiring:
        """The control: every neuron receives another neuron's inputs (rows permuted).

        Same neurons, same number of connections, same weights and signs, same balance of
        input per neuron — but who listens to whom is chance: the senses still reach the
        descending neurons, by pathways no fly ever had.
        """
        order = _shuffled(np.arange(self.n), SplitMix64(seed))  # ours: the same on any numpy
        return Wiring(
            matrix=self.matrix[order].tocsr(),
            superclass=self.superclass,
            klass=self.klass,
            variant=self.variant,
            min_synapses=self.min_synapses,
            control=seed,
        )

    def describe(self) -> dict:
        """What a brain file records about the graph it was bred on."""
        return {
            "graph": self.variant,
            "min_synapses": self.min_synapses,
            "control": self.control,
            "neurons": self.n,
            "connections": self.connections,
            "digest": self.digest,
        }


def digest_of(matrix, superclass: np.ndarray, klass: np.ndarray) -> str:
    """A short content hash of the graph and of what each neuron is: a brain refuses to think
    with another one. The labels count because they decide which neurons a sense drives and
    which are read out; the arrays are hashed in fixed types, so the hash does not depend on
    how scipy chose to store them."""
    matrix = matrix.tocsr()
    if not matrix.has_sorted_indices:  # a Wiring's already are; a bare matrix may not be
        matrix = matrix.sorted_indices()
    sha = hashlib.sha256()
    sha.update(np.asarray(matrix.shape, "<i8").tobytes())
    for part, kind in ((matrix.indptr, "<i8"), (matrix.indices, "<i8"), (matrix.data, "<f4")):
        sha.update(np.ascontiguousarray(part, dtype=kind).tobytes())
    for labels in (superclass, klass):
        sha.update("\x00".join(str(label) for label in labels).encode())
        sha.update(b"\x01")
    return sha.hexdigest()[:16]


def normalise_rows(matrix):
    """Each neuron's inputs sum to at most 1 in magnitude, so activity stays bounded."""
    sparse = require_scipy()
    total = np.asarray(abs(matrix).sum(axis=1)).ravel()
    scale = np.divide(1.0, total, out=np.zeros_like(total, dtype=np.float64), where=total > 0)
    return (sparse.diags(scale.astype(np.float32)) @ matrix).tocsr().astype(np.float32)


# --- dynamics --------------------------------------------------------------------------------


class RateNetwork:
    """r <- (1 - leak) r + leak tanh(gain W r + input), `substeps` times per world tick.

    `rate` and `current` are (n,) for one fly or (n, k) for k flies stepped together;
    `gain` is a number, or one per fly.
    """

    backend = "numpy"

    def __init__(
        self, wiring: Wiring, substeps: int = DEFAULT_SUBSTEPS, leak: float = DEFAULT_LEAK
    ) -> None:
        if substeps < 1 or not 0 < leak <= 1:
            raise ValueError("substeps must be at least 1 and leak in (0, 1]")
        require_scipy()
        self.wiring = wiring
        self.substeps = substeps
        self.leak = leak

    def zeros(self, flies: int | None = None) -> np.ndarray:
        shape = (self.wiring.n,) if flies is None else (self.wiring.n, flies)
        return np.zeros(shape, np.float32)

    def step(self, rate: np.ndarray, current: np.ndarray, gain) -> np.ndarray:
        matrix, leak = self.wiring.matrix, self.leak
        gain = np.asarray(gain, np.float32)
        for _ in range(self.substeps):
            drive = (matrix @ rate) * gain + current
            rate = (1 - leak) * rate + leak * np.tanh(drive)
        return rate.astype(np.float32, copy=False)


class TorchRateNetwork(RateNetwork):
    """The same update on a torch device: the full graph and many flies in one multiply."""

    backend = "torch"

    def __init__(
        self,
        wiring: Wiring,
        substeps: int = DEFAULT_SUBSTEPS,
        leak: float = DEFAULT_LEAK,
        device: str | None = None,
    ) -> None:
        super().__init__(wiring, substeps, leak)
        try:
            import torch
        except ImportError:
            raise ValueError("the torch backend needs torch installed") from None
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        matrix = wiring.matrix
        self._matrix = torch.sparse_csr_tensor(
            torch.from_numpy(matrix.indptr.astype(np.int64)),
            torch.from_numpy(matrix.indices.astype(np.int64)),
            torch.from_numpy(matrix.data.astype(np.float32)),
            size=matrix.shape,
            device=self.device,
        )

    def step(self, rate: np.ndarray, current: np.ndarray, gain) -> np.ndarray:
        torch = self._torch
        single = rate.ndim == 1
        r = torch.from_numpy(np.ascontiguousarray(rate, np.float32)).to(self.device)
        c = torch.from_numpy(np.ascontiguousarray(current, np.float32)).to(self.device)
        g = torch.as_tensor(np.asarray(gain, np.float32), device=self.device)
        if single:
            r, c = r[:, None], c[:, None]
        for _ in range(self.substeps):
            r = (1 - self.leak) * r + self.leak * torch.tanh((self._matrix @ r) * g + c)
        out = r.cpu().numpy()
        return out[:, 0] if single else out


def make_network(
    wiring: Wiring,
    substeps: int = DEFAULT_SUBSTEPS,
    leak: float = DEFAULT_LEAK,
    backend: str = "numpy",
    device: str | None = None,
) -> RateNetwork:
    """The network for this wiring and these settings — one, shared by every brain on it: on
    torch the graph is put on the device once, not once per fly. A network keeps no state
    (each brain holds its own rates), so sharing is safe."""
    key = (substeps, leak, backend, device)
    if key not in wiring.networks:
        if backend == "numpy":
            wiring.networks[key] = RateNetwork(wiring, substeps, leak)
        elif backend == "torch":
            wiring.networks[key] = TorchRateNetwork(wiring, substeps, leak, device)
        else:
            raise ValueError(f"unknown backend {backend!r}; choose numpy or torch")
    return wiring.networks[key]


# --- senses in ---------------------------------------------------------------------------------

# Which neurons each bundle of tiny features drives: (first feature, how many, classes,
# superclasses). The order follows dojo.features.tiny_features: smell (fruit, humidity,
# nest: own tile + N/E/S/W each), touch (bumped, on resource, water adjacent, on nest),
# body (satiety, hydration, energy, health, age), light.
_BUNDLES = (
    (0, 5, ("olfactory",), ()),
    (5, 5, ("hygrosensory",), ()),
    (10, 5, ("thermosensory", "chemosensory"), ()),
    (15, 1, ("mechanosensory", "mechanosensory_tactile"), ()),
    (16, 2, ("gustatory",), ()),
    (18, 1, ("mechanosensory_proprioceptive", "unknown_sensory"), ()),
    (19, 5, ("DAN",), ("cb_endocrine",)),
    (24, 1, (), ("visual_projection",)),
)
_FALLBACK = ("cb_sensory", "sensory_ascending", "ol_sensory", "vnc_sensory")


def _shuffled(indices: np.ndarray, rng: SplitMix64) -> np.ndarray:
    indices = indices.copy()
    for last in range(len(indices) - 1, 0, -1):  # Fisher-Yates on our own generator
        other = rng.randbelow(last + 1)
        indices[last], indices[other] = indices[other], indices[last]
    return indices


@dataclass
class Encoding:
    """Feature f drives the neurons `groups[f]`: a seeded, disjoint choice per class."""

    groups: list[np.ndarray]
    seed: int

    def __post_init__(self) -> None:
        self._neurons = np.concatenate(self.groups)
        self._feature = np.concatenate(
            [np.full(len(group), index) for index, group in enumerate(self.groups)]
        )

    @classmethod
    def for_wiring(cls, wiring: Wiring, seed: int = 0) -> Encoding:
        rng = SplitMix64(seed)
        taken = np.zeros(wiring.n, bool)
        spare = _shuffled(wiring.where(superclass=_FALLBACK), rng)
        groups: list[np.ndarray | None] = [None] * TINY_SIZE
        for first, count, klass, superclass in _BUNDLES:
            pool = wiring.where(klass=klass, superclass=superclass)
            pool = _shuffled(pool[~taken[pool]], rng)
            if len(pool) < count:  # a small or foreign graph: borrow other sensory neurons
                extra = spare[~taken[spare] & ~np.isin(spare, pool)][: count - len(pool)]
                pool = np.concatenate([pool, extra])
            if len(pool) < count:
                raise ValueError(
                    f"this graph has too few neurons of {klass or superclass} "
                    f"to carry {count} features"
                )
            size = min(len(pool) // count, GROUP_MAX)
            for offset in range(count):
                group = np.sort(pool[offset * size : (offset + 1) * size])
                groups[first + offset] = group
                taken[group] = True
        return cls(groups, seed)

    def current(self, features: np.ndarray, gains: np.ndarray, n: int) -> np.ndarray:
        """The input current for one fly: its 25 features, scaled, onto their neurons."""
        current = np.zeros(n, np.float32)
        current[self._neurons] = (features * gains)[self._feature]
        return current


# --- readout -----------------------------------------------------------------------------------


@dataclass
class Projection:
    """The descending neurons pooled into POOLED signed buckets: fixed, seeded, not learned."""

    descending: np.ndarray  # indices of the descending neurons
    bucket: np.ndarray  # which bucket each one feeds
    weight: np.ndarray  # its sign over the square root of the bucket's size
    seed: int

    @classmethod
    def for_wiring(cls, wiring: Wiring, seed: int = 0, pooled: int = POOLED) -> Projection:
        descending = wiring.where(superclass=("descending_neuron",))
        if len(descending) < pooled:
            raise ValueError(
                f"this graph has {len(descending)} descending neurons; {pooled} are needed"
            )
        rng = SplitMix64(seed)
        order = _shuffled(np.arange(len(descending)), rng)
        bucket = np.empty(len(descending), np.int64)
        bucket[order] = np.arange(len(descending)) % pooled  # every bucket gets its share
        sign = np.array([1.0 if rng.randbelow(2) else -1.0 for _ in descending], np.float32)
        counts = np.bincount(bucket, minlength=pooled)
        return cls(descending, bucket, sign / np.sqrt(counts[bucket]).astype(np.float32), seed)

    def pool(self, rate: np.ndarray) -> np.ndarray:
        """(POOLED,) for one fly's rate vector, (POOLED, k) for k flies."""
        active = rate[self.descending]
        pooled = self.bucket.max() + 1
        if active.ndim == 1:
            return np.bincount(self.bucket, weights=active * self.weight, minlength=pooled).astype(
                np.float32
            )
        out = np.zeros((pooled, active.shape[1]), np.float32)
        np.add.at(out, self.bucket, active * self.weight[:, None])
        return out
