"""MaleCNS on disk: fetch the published files, build a graph from them, cache it, load it.

The source is the MaleCNS v1.0 "flat connectome" (Janelia FlyEM, CC-BY 4.0,
https://male-cns.janelia.org/download/). Nothing here is downloaded unless asked for,
and nothing lives in the repository: the cache is ~/.cache/neurogarden/malecns, or
wherever NEUROGARDEN_CACHE points.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .model import Wiring, normalise_rows, require_scipy

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "transmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
DATASET = "male-cns:v1.0"
VARIANTS = ("central", "full")
# The central brain: no optic lobes, no nerve cord — what sits between the senses of the
# head and the neurons that descend to the body.
CENTRAL = (
    "cb_intrinsic", "cb_sensory", "descending_neuron", "ascending_neuron", "cb_motor",
    "visual_projection", "visual_centrifugal", "cb_endocrine", "sensory_ascending",
)  # fmt: skip
# A modelling choice: which transmitters excite and which inhibit. Everything not listed
# (dopamine, serotonin, octopamine, unclear, missing) counts as excitatory.
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
DEFAULT_MIN_SYNAPSES = 5
_CHUNK = 1 << 20

_wirings: dict[tuple, Wiring] = {}  # loaded graphs, shared by every brain in the process


def cache_dir() -> Path:
    custom = os.environ.get("NEUROGARDEN_CACHE")
    base = Path(custom) if custom else Path.home() / ".cache" / "neurogarden"
    return base / "malecns"


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.feather as feather
    except ImportError:
        raise ValueError(
            "building the graph needs pyarrow: install neurogarden[connectome]"
        ) from None
    return feather


def fetch(
    cache: Path | None = None, on_progress: Callable[[str, int, int | None], None] | None = None
) -> list[Path]:
    """Download the three source files that are not in the cache yet; returns their paths."""
    cache = cache or cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FILES.values():
        target = cache / name
        paths.append(target)
        if target.exists():
            continue
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(BASE_URL + name) as response, open(partial, "wb") as out:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None
            done = 0
            while chunk := response.read(_CHUNK):
                out.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(name, done, total)
        partial.replace(target)  # a file is either whole or absent
    return paths


def graph_paths(variant: str, min_synapses: int, cache: Path | None = None) -> tuple[Path, Path]:
    cache = cache or cache_dir()
    stem = f"graph-{variant}-ms{min_synapses}"
    return cache / f"{stem}.npz", cache / f"{stem}.neurons.npz"


def assemble(
    ids: np.ndarray,
    superclass: np.ndarray,
    klass: np.ndarray,
    sign: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    synapses: np.ndarray,
    variant: str = "central",
    min_synapses: int = DEFAULT_MIN_SYNAPSES,
) -> Wiring:
    """A Wiring from neuron and connection tables (ids ascending; pre/post are body ids)."""
    sparse = require_scipy()
    if variant not in VARIANTS:
        raise ValueError(f"unknown graph {variant!r}; choose from {list(VARIANTS)}")
    keep = np.ones(len(ids), bool) if variant == "full" else np.isin(superclass, CENTRAL)
    kept_ids = ids[keep]
    strong = synapses >= min_synapses
    inside = strong & np.isin(pre, kept_ids) & np.isin(post, kept_ids)
    rows = np.searchsorted(kept_ids, post[inside])
    cols = np.searchsorted(kept_ids, pre[inside])
    values = synapses[inside].astype(np.float32) * sign[keep][cols]
    n = len(kept_ids)
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(n, n), dtype=np.float32)
    matrix.sum_duplicates()
    return Wiring(
        matrix=normalise_rows(matrix),
        superclass=superclass[keep],
        klass=klass[keep],
        variant=variant,
        min_synapses=min_synapses,
    )


def build(
    variant: str = "central",
    min_synapses: int = DEFAULT_MIN_SYNAPSES,
    cache: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Wiring:
    """Build a graph from the fetched files and cache it. Takes a minute or two."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc

    feather = _require_pyarrow()
    cache = cache or cache_dir()
    say = on_progress or (lambda text: None)
    sources = {key: cache / name for key, name in FILES.items()}
    missing = [path.name for path in sources.values() if not path.exists()]
    if missing:
        raise ValueError(f"not in {cache}: {missing}; run `neurogarden connectome fetch`")

    annotations = feather.read_table(
        sources["annotations"], columns=["bodyId", "status", "superclass", "class"]
    )
    traced = annotations.filter(pc.equal(annotations["status"], "Traced"))
    traced = traced.take(pc.sort_indices(traced["bodyId"]))
    ids = traced["bodyId"].to_numpy()
    superclass = np.array([value or "none" for value in traced["superclass"].to_pylist()])
    klass = np.array([value or "none" for value in traced["class"].to_pylist()])
    say(f"{len(ids)} traced neurons")

    transmitters = feather.read_table(sources["transmitters"], columns=["body", "consensus_nt"])
    id_set = pa.array(ids)
    transmitters = transmitters.filter(pc.is_in(transmitters["body"], value_set=id_set))
    nt = dict(
        zip(transmitters["body"].to_pylist(), transmitters["consensus_nt"].to_pylist(), strict=True)
    )
    sign = np.array([SIGN.get(nt.get(int(body)) or "", 1.0) for body in ids], np.float32)

    # The weights file lists every segment pair; keep those between traced neurons.
    pre_parts, post_parts, weight_parts = [], [], []
    with pa.memory_map(str(sources["weights"])) as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            mask = pc.and_(
                pc.is_in(batch["body_pre"], value_set=id_set),
                pc.is_in(batch["body_post"], value_set=id_set),
            )
            batch = batch.filter(mask)
            pre_parts.append(batch["body_pre"].to_numpy())
            post_parts.append(batch["body_post"].to_numpy())
            weight_parts.append(batch["weight"].to_numpy())
    pre, post = np.concatenate(pre_parts), np.concatenate(post_parts)
    synapses = np.concatenate(weight_parts)
    say(f"{len(pre)} connections between them")

    wiring = assemble(ids, superclass, klass, sign, pre, post, synapses, variant, min_synapses)
    save(wiring, cache)
    say(f"{wiring.n} neurons and {wiring.connections} connections in graph {variant}")
    return wiring


def save(wiring: Wiring, cache: Path | None = None) -> tuple[Path, Path]:
    """Write a (non-control) graph into the cache under its variant's name."""
    sparse = require_scipy()
    cache = cache or cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    graph_path, neurons_path = graph_paths(wiring.variant, wiring.min_synapses, cache)
    sparse.save_npz(graph_path, wiring.matrix, compressed=False)
    meta = {"dataset": DATASET, **wiring.describe()}
    np.savez(neurons_path, superclass=wiring.superclass, klass=wiring.klass, meta=json.dumps(meta))
    return graph_path, neurons_path


def load(
    variant: str = "central",
    min_synapses: int = DEFAULT_MIN_SYNAPSES,
    control: int | None = None,
    cache: Path | None = None,
) -> Wiring:
    """A cached graph (built once with `neurogarden connectome build`), or its control."""
    sparse = require_scipy()
    cache = cache or cache_dir()
    key = (str(cache), variant, min_synapses, control)
    if key in _wirings:
        return _wirings[key]
    if control is not None:
        wiring = load(variant, min_synapses, None, cache).randomised(control)
    else:
        graph_path, neurons_path = graph_paths(variant, min_synapses, cache)
        if not graph_path.exists() or not neurons_path.exists():
            raise ValueError(
                f"no {variant} graph (>= {min_synapses} synapses) in {cache}: run "
                "`neurogarden connectome fetch` and `neurogarden connectome build`"
            )
        with np.load(neurons_path) as table:
            superclass, klass = table["superclass"], table["klass"]
        wiring = Wiring(
            matrix=sparse.load_npz(graph_path).tocsr(),
            superclass=superclass,
            klass=klass,
            variant=variant,
            min_synapses=min_synapses,
        )
    _wirings[key] = wiring
    return wiring


def forget() -> None:
    """Drop the graphs held in memory (tests, or after a rebuild)."""
    _wirings.clear()


def info(cache: Path | None = None) -> dict:
    """What the cache holds: source files and built graphs, with sizes."""
    cache = cache or cache_dir()
    sources = {
        key: (cache / name).stat().st_size if (cache / name).exists() else None
        for key, name in FILES.items()
    }
    graphs = []
    for path in sorted(cache.glob("graph-*.neurons.npz")) if cache.exists() else []:
        with np.load(path) as table:
            graphs.append(json.loads(str(table["meta"])))
    return {"cache": str(cache), "sources": sources, "graphs": graphs}
