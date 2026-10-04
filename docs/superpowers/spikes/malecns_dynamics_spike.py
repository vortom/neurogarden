"""Spike (sub-project 6): how big are the pruned graphs, and do rate dynamics carry a signal
from the sensory neurons to the descending neurons in a few steps?

    uv run --with scipy python docs/superpowers/spikes/malecns_dynamics_spike.py

Needs graph-spike.npz and neurons-spike.npz from malecns_graph_spike.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CACHE = Path.home() / ".cache" / "neurogarden" / "malecns"
CENTRAL = (
    "cb_intrinsic", "cb_sensory", "descending_neuron", "ascending_neuron", "cb_motor",
    "visual_projection", "visual_centrifugal", "cb_endocrine", "sensory_ascending",
)  # fmt: skip


def timed(matrix, flies: int = 1, repeats: int = 10) -> float:
    activity = np.random.default_rng(0).random((matrix.shape[0], flies), dtype=np.float32)
    matrix @ activity
    started = time.perf_counter()
    for _ in range(repeats):
        matrix @ activity
    return (time.perf_counter() - started) / repeats * 1000


def normalised(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Each neuron's inputs sum to at most 1 in magnitude: activity stays bounded."""
    total = np.asarray(abs(matrix).sum(axis=1)).ravel()
    scale = np.divide(1.0, total, out=np.zeros_like(total), where=total > 0)
    return (sp.diags(scale.astype(np.float32)) @ matrix).tocsr()


def main() -> None:
    full = sp.load_npz(CACHE / "graph-spike.npz").tocsr()
    neurons = np.load(CACHE / "neurons-spike.npz")
    superclass, klass = neurons["superclass"], neurons["klass"]
    central = np.isin(superclass, CENTRAL)

    def prune(matrix, minimum):
        kept = matrix.copy()
        kept.data[np.abs(kept.data) < minimum] = 0
        kept.eliminate_zeros()
        return kept

    variants = {
        "full": full,
        "full, >=5 synapses": prune(full, 5),
        "central": full[central][:, central].tocsr(),
        "central, >=5 synapses": prune(full[central][:, central].tocsr(), 5),
    }
    for name, matrix in variants.items():
        print(
            f"{name:<24} {matrix.shape[0]:>7} neurons {matrix.nnz:>9} connections  "
            f"1 fly {timed(matrix):6.1f} ms   10 flies {timed(matrix, 10):7.1f} ms"
        )

    # Rate dynamics on the pruned central brain: r <- (1 - leak) r + leak tanh(gain W r + input)
    graph = normalised(variants["central, >=5 synapses"])
    sub_class, sub_super = klass[central], superclass[central]
    olfactory = np.flatnonzero(sub_class == "olfactory")
    gustatory = np.flatnonzero(sub_class == "gustatory")
    descending = np.flatnonzero(sub_super == "descending_neuron")
    print(
        f"\npruned central brain: {len(olfactory)} olfactory, {len(gustatory)} gustatory, "
        f"{len(descending)} descending neurons"
    )
    for gain in (1.0, 2.0, 4.0):
        for leak in (0.5, 1.0):
            rate = np.zeros(graph.shape[0], np.float32)
            current = np.zeros(graph.shape[0], np.float32)
            current[olfactory[: len(olfactory) // 2]] = 1.0  # smell on one half of the nose
            trace = []
            for step in range(12):
                if step == 6:  # the smell moves to the other half
                    current[:] = 0
                    current[olfactory[len(olfactory) // 2 :]] = 1.0
                rate = (1 - leak) * rate + leak * np.tanh(gain * (graph @ rate) + current)
                trace.append(rate[descending].copy())
            trace = np.array(trace)
            moved = np.abs(trace[11] - trace[5]).mean()
            print(
                f"gain {gain:3.1f} leak {leak:3.1f}: descending |activity| by step "
                + " ".join(f"{np.abs(t).mean():.3f}" for t in trace[[0, 1, 2, 3, 5, 11]])
                + f"   active(>0.01) {int((np.abs(trace[5]) > 0.01).sum()):>4}/{len(descending)}"
                + f"   change after the smell moved {moved:.4f}"
            )


if __name__ == "__main__":
    main()
