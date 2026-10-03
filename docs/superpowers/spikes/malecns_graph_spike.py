"""Spike (sub-project 6): turn the MaleCNS flat files into a sparse signed graph and time it.

Throwaway measurement code, kept so the numbers in the SP6 spec can be reproduced:

    uv run --with pyarrow --with scipy python docs/superpowers/spikes/malecns_graph_spike.py

Reads ~/.cache/neurogarden/malecns/*.feather (see the roadmap for the URLs), keeps the
neurons whose status is "Traced", keeps the connections between them, signs each
connection by the presynaptic neuron's consensus neurotransmitter, and reports sizes,
candidate input/output populations and the cost of one update on this machine.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import scipy.sparse as sp

CACHE = Path.home() / ".cache" / "neurogarden" / "malecns"
ANNOTATIONS = CACHE / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
TRANSMITTERS = CACHE / "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = CACHE / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
GRAPH = CACHE / "graph-spike.npz"
NEURONS = CACHE / "neurons-spike.npz"

# A modelling choice, not a fact in the data: which transmitters excite, which inhibit.
SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
DEFAULT_SIGN = (
    1.0  # dopamine, serotonin, octopamine, unclear, missing: treated as weakly excitatory
)


def clock(label: str, started: float) -> float:
    now = time.perf_counter()
    print(f"[{now - started:7.1f}s] {label}", flush=True)
    return now


def main() -> None:
    started = time.perf_counter()
    annotations = feather.read_table(
        ANNOTATIONS, columns=["bodyId", "status", "superclass", "class", "type", "somaSide"]
    )
    traced = annotations.filter(pc.equal(annotations["status"], "Traced"))
    order = pc.sort_indices(traced["bodyId"])
    traced = traced.take(order)
    ids = traced["bodyId"].to_numpy()
    n = len(ids)
    clock(f"neurons: {annotations.num_rows} annotated bodies, {n} traced", started)

    superclass = [value or "none" for value in traced["superclass"].to_pylist()]
    klass = [value or "none" for value in traced["class"].to_pylist()]
    print("superclass:", dict(Counter(superclass).most_common()))
    print("class:", dict(Counter(klass).most_common()))

    transmitters = feather.read_table(TRANSMITTERS, columns=["body", "consensus_nt"])
    keep = pc.is_in(transmitters["body"], value_set=pa.array(ids))
    transmitters = transmitters.filter(keep)
    nt_of = dict(
        zip(transmitters["body"].to_pylist(), transmitters["consensus_nt"].to_pylist(), strict=True)
    )
    nts = [nt_of.get(int(body)) or "missing" for body in ids]
    print("consensus_nt:", dict(Counter(nts).most_common()))
    sign = np.array([SIGN.get(nt, DEFAULT_SIGN) for nt in nts], dtype=np.float32)
    clock("neurotransmitters joined", started)

    # The weights file lists every segment pair (151.9M rows); read it batch by batch and
    # keep the pairs whose both ends are traced neurons.
    id_set = pa.array(ids)
    pre_parts, post_parts, weight_parts = [], [], []
    rows = kept = 0
    with pa.memory_map(str(WEIGHTS)) as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            rows += batch.num_rows
            mask = pc.and_(
                pc.is_in(batch["body_pre"], value_set=id_set),
                pc.is_in(batch["body_post"], value_set=id_set),
            )
            batch = batch.filter(mask)
            kept += batch.num_rows
            pre_parts.append(batch["body_pre"].to_numpy())
            post_parts.append(batch["body_post"].to_numpy())
            weight_parts.append(batch["weight"].to_numpy())
    pre = np.searchsorted(ids, np.concatenate(pre_parts))
    post = np.searchsorted(ids, np.concatenate(post_parts))
    weight = np.concatenate(weight_parts).astype(np.float32)
    clock(f"connections: {rows} segment pairs read, {kept} between traced neurons", started)
    print(
        "synapses per connection: "
        f"total {int(weight.sum())}, median {np.median(weight):.0f}, max {int(weight.max())}; "
        f"connections with >= 5 synapses: {int((weight >= 5).sum())}"
    )

    # W[post, pre]: activity of `pre` neurons drives `post` neurons, signed by the transmitter.
    matrix = sp.csr_matrix((weight * sign[pre], (post, pre)), shape=(n, n), dtype=np.float32)
    clock(f"CSR built: {matrix.nnz} nonzeros, {matrix.data.nbytes / 1e6:.0f} MB data", started)
    sp.save_npz(GRAPH, matrix, compressed=False)
    np.savez(
        NEURONS,
        ids=ids,
        sign=sign,
        superclass=np.array(superclass),
        klass=np.array(klass),
        type=np.array([value or "" for value in traced["type"].to_pylist()]),
        side=np.array([value or "" for value in traced["somaSide"].to_pylist()]),
    )
    clock(f"saved {GRAPH.name} ({GRAPH.stat().st_size / 1e6:.0f} MB) and {NEURONS.name}", started)

    rng = np.random.default_rng(0)
    for columns in (1, 10, 64):
        activity = rng.random((n, columns), dtype=np.float32)
        matrix @ activity  # warm up
        tick = time.perf_counter()
        repeats = 10
        for _ in range(repeats):
            matrix @ activity
        per = (time.perf_counter() - tick) / repeats
        print(
            f"one update, {columns:>2} flies batched: {per * 1000:7.1f} ms "
            f"({per * 1000 / columns:6.1f} ms per fly)"
        )

    # The brain-only subgraph between senses and descending neurons, for CPU-sized runs.
    superclass = np.array(superclass)
    brain = np.isin(
        superclass,
        ["cb_intrinsic", "cb_sensory", "descending_neuron", "ascending_neuron", "cb_motor",
         "visual_projection", "visual_centrifugal", "cb_endocrine", "sensory_ascending"],
    )  # fmt: skip
    sub = matrix[brain][:, brain].tocsr()
    activity = rng.random((sub.shape[0], 10), dtype=np.float32)
    sub @ activity
    tick = time.perf_counter()
    for _ in range(10):
        sub @ activity
    per = (time.perf_counter() - tick) / 10
    print(
        f"central-brain subgraph (no optic lobes, no VNC): {sub.shape[0]} neurons, {sub.nnz} "
        f"nonzeros; one update for 10 flies: {per * 1000:.1f} ms"
    )
    clock("done", started)


if __name__ == "__main__":
    main()
