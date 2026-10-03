"""Spike (sub-project 6): how much of what the fly senses can be read back from the
descending neurons?

The teacher (the shipped evolved brain) flies; its feature sequence is replayed through the
network under several settings at once (one column each). Per setting, on a held-out life:
ridge R^2 of the 25 features from the 64 pooled features and from all descending neurons, and
how well a readout fitted on the pooled features matches the teacher's action probabilities.

    uv run python docs/superpowers/spikes/malecns_throughput_spike.py SUBSTEPS LEAK \\
        [real|CONTROL_SEED] [ENCODING_SEED]

Needs the cached central graph (`neurogarden connectome build`). About 17 minutes for
3 substeps on one core.
"""

import sys
import time

import numpy as np
from scipy.optimize import minimize

from neurogarden.brains import EvolvedBrain
from neurogarden.brains.connectome import READOUT_GAIN
from neurogarden.brains.evolved import choose, scores, softmax
from neurogarden.connectome import data
from neurogarden.connectome.model import Encoding, Projection, make_network
from neurogarden.dojo import NeuroGardenEnv
from neurogarden.dojo.features import DEFAULT_AGE_SCALE, tiny_features
from neurogarden.engine.rng import SplitMix64

SUBSTEPS, LEAK = int(sys.argv[1]), float(sys.argv[2])
CONTROL = None if len(sys.argv) < 4 or sys.argv[3] == "real" else int(sys.argv[3])
ENCODING_SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
TEMPERATURE, STEPS = 0.5, 2000
SETTINGS = [(3.0, 1.0)] + [(g, s) for g in (1.0, 2.0, 4.0, 8.0) for s in (0.5, 2.0)]  # gain, scale
GROUPS = {
    "fruit": range(0, 5),
    "humid": range(5, 10),
    "nest": range(10, 15),
    "touch": range(15, 19),
    "body": range(19, 24),
    "light": range(24, 25),
}
teacher = EvolvedBrain()


def trajectory(seeds):
    """The teacher's lives: features and action probabilities per tick, and episode starts."""
    env = NeuroGardenEnv(max_steps=STEPS)
    features, targets, starts = [], [], []
    for seed in seeds:
        rng = SplitMix64(seed)
        observation, _ = env.reset(seed=seed)
        starts.append(len(features))
        while True:
            x = tiny_features(observation, DEFAULT_AGE_SCALE)
            p = softmax(scores(teacher.genome, x) / TEMPERATURE)
            features.append(x)
            targets.append(p)
            action = choose(np.log(p + 1e-12) * TEMPERATURE, TEMPERATURE, rng)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return np.array(features, np.float32), np.array(targets, np.float64), set(starts)


def replay(features, starts, network, encoding, projection):
    """(ticks, 64, k) pooled and (ticks, descending, k) rates for the k settings."""
    k = len(SETTINGS)
    gain = np.array([g for g, _ in SETTINGS], np.float32)
    scale = np.array([s for _, s in SETTINGS], np.float32)
    ones = np.ones(features.shape[1], np.float32)
    rate = network.zeros(k)
    pooled = np.zeros((len(features), 64, k), np.float32)
    descending = np.zeros((len(features), len(projection.descending), k), np.float32)
    for tick, x in enumerate(features):
        if tick in starts:
            rate = network.zeros(k)
        current = encoding.current(x, ones, network.wiring.n)[:, None] * scale[None, :]
        rate = network.step(rate, current, gain)
        pooled[tick] = projection.pool(rate)
        descending[tick] = rate[projection.descending]
    return pooled, descending


def ridge_r2(x, y, x_held, y_held, l2=1e-2):
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mean_x, y - mean_y
    w = np.linalg.solve(xc.T @ xc + l2 * len(x) * np.eye(x.shape[1]), xc.T @ yc)
    predicted = (x_held - mean_x) @ w + mean_y
    residual = ((y_held - predicted) ** 2).sum(axis=0)
    total = ((y_held - y_held.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - residual / np.maximum(total, 1e-9)


def distil(x, targets, x_held, targets_held, l2=1e-4):
    n, d = x.shape
    classes = targets.shape[1]

    def logp_of(features, flat):
        w, b = flat[: d * classes].reshape(d, classes), flat[d * classes :]
        logits = features @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True)), w

    def loss(flat):
        logp, w = logp_of(x, flat)
        grad = (np.exp(logp) - targets) / n
        value = -(targets * logp).sum() / n + l2 * (w**2).sum()
        return value, np.concatenate([(x.T @ grad + 2 * l2 * w).ravel(), grad.sum(axis=0)])

    found = minimize(
        loss, np.zeros(d * classes + classes), jac=True, method="L-BFGS-B", options={"maxiter": 300}
    )
    logp, _ = logp_of(x_held, found.x)
    kl = (targets_held * (np.log(targets_held + 1e-12) - logp)).sum(axis=1).mean()
    agree = (logp.argmax(axis=1) == targets_held.argmax(axis=1)).mean()
    return float(kl), float(agree)


started = time.perf_counter()
wiring = data.load("central", 5, CONTROL)
network = make_network(wiring, SUBSTEPS, LEAK)
encoding = Encoding.for_wiring(wiring, ENCODING_SEED)
projection = Projection.for_wiring(wiring, 0)
train_x, train_p, train_starts = trajectory(range(100, 103))
held_x, held_p, held_starts = trajectory(range(200, 201))
pooled, descending = replay(train_x, train_starts, network, encoding, projection)
held_pooled, held_descending = replay(held_x, held_starts, network, encoding, projection)
name = "real" if CONTROL is None else f"control {CONTROL}"
print(
    f"{name}  substeps {SUBSTEPS} leak {LEAK} encoding seed {ENCODING_SEED}: {len(train_x)} "
    f"training ticks, {len(held_x)} held out ({time.perf_counter() - started:.0f}s)",
    flush=True,
)
varying = held_x.std(axis=0) > 1e-6
for column, (gain, scale) in enumerate(SETTINGS):
    line = [f"gain {gain:>3} in x{scale:<3}"]
    for label, x, x_held in (
        ("pooled", pooled, held_pooled),
        ("desc", descending, held_descending),
    ):
        r2 = ridge_r2(
            x[:, :, column].astype(np.float64),
            train_x.astype(np.float64),
            x_held[:, :, column].astype(np.float64),
            held_x.astype(np.float64),
        )
        parts = " ".join(
            f"{group} {np.mean([r2[i] for i in indices if varying[i]] or [np.nan]):5.2f}"
            for group, indices in GROUPS.items()
        )
        line.append(f"{label} R2: {parts}")
    kl, agree = distil(
        pooled[:, :, column].astype(np.float64) * READOUT_GAIN,
        train_p,
        held_pooled[:, :, column].astype(np.float64) * READOUT_GAIN,
        held_p,
    )
    line.append(f"distil pooled KL {kl:.3f} agree {agree:.2f}")
    activity = float(np.abs(held_descending[:, :, column]).mean())
    line.append(f"|desc| {activity:.3f}")
    print(" | ".join(line), flush=True)
print(f"done ({time.perf_counter() - started:.0f}s)")
