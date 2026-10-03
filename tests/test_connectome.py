"""The connectome fly on a synthetic wiring: nothing here downloads or needs the real graph."""

import asyncio
import json
import sys

import numpy as np
import pytest

from neurogarden import cli
from neurogarden.brains import BRAINS, ConnectomeBrain, ConnectomeGenome, EvolvedBrain, Genome
from neurogarden.brains.base import brain_seed
from neurogarden.brains.connectome import ConnectomeFlock, ConnectomeTrainable
from neurogarden.brains.evolved import default_weights, softmax, weights_kind
from neurogarden.connectome import data
from neurogarden.connectome.model import (
    POOLED,
    Encoding,
    Projection,
    RateNetwork,
    _shuffled,
    digest_of,
    make_network,
)
from neurogarden.dojo import NeuroGardenEnv, watch
from neurogarden.dojo import distil as distil_module
from neurogarden.dojo.distil import DistilConfig, distil, fit_readout
from neurogarden.dojo.evolve import EvolveConfig, evolve
from neurogarden.dojo.features import TINY_SIZE
from neurogarden.engine import Config
from neurogarden.engine.rng import SplitMix64
from neurogarden.sdk import fly_flock
from neurogarden.server import Server, ServerConfig
from test_brains import observation

# A small brain with every class the encoding asks for: (class, superclass, how many).
POPULATIONS = (
    ("olfactory", "cb_sensory", 40),
    ("hygrosensory", "cb_sensory", 15),
    ("thermosensory", "cb_sensory", 5),
    ("chemosensory", "cb_sensory", 5),
    ("mechanosensory", "cb_sensory", 10),
    ("gustatory", "cb_sensory", 10),
    ("mechanosensory_proprioceptive", "cb_sensory", 6),
    ("DAN", "cb_intrinsic", 10),
    ("none", "cb_endocrine", 5),
    ("none", "visual_projection", 20),
    ("none", "descending_neuron", 80),
    ("none", "cb_intrinsic", 150),
    ("none", "ol_intrinsic", 44),  # not part of the central brain
)


def tables(seed=0, connections=6000):
    rng = np.random.default_rng(seed)
    klass = np.concatenate([np.full(count, name) for name, _, count in POPULATIONS])
    superclass = np.concatenate([np.full(count, name) for _, name, count in POPULATIONS])
    n = len(klass)
    ids = np.arange(1000, 1000 + 3 * n, 3)  # body ids: ascending, not 0..n
    sign = np.where(rng.random(n) < 0.7, 1.0, -1.0).astype(np.float32)
    pre = ids[rng.integers(0, n, connections)]
    post = ids[rng.integers(0, n, connections)]
    synapses = rng.integers(1, 12, connections)
    return ids, superclass, klass, sign, pre, post, synapses


def tiny_wiring(variant="full", min_synapses=1, seed=0):
    return data.assemble(*tables(seed), variant=variant, min_synapses=min_synapses)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty connectome cache of our own, with the synthetic `full` graph built in it."""
    monkeypatch.setenv("NEUROGARDEN_CACHE", str(tmp_path))
    data.forget()
    data.save(tiny_wiring())
    yield data.cache_dir()
    data.forget()


# --- the graph -----------------------------------------------------------------------------


def test_assemble_keeps_strong_connections_signs_them_and_normalises_rows():
    ids, superclass, klass, sign, pre, post, synapses = tables()
    wiring = data.assemble(ids, superclass, klass, sign, pre, post, synapses, "full", 5)
    assert wiring.n == len(ids) and wiring.variant == "full" and wiring.min_synapses == 5
    dense = wiring.matrix.toarray()
    strong = synapses >= 5
    row = np.searchsorted(ids, post[strong])
    col = np.searchsorted(ids, pre[strong])
    assert (dense != 0).sum() == len(set(zip(row.tolist(), col.tolist(), strict=True)))
    assert np.all(np.sign(dense[row, col]) == sign[col])  # the sender's transmitter decides
    totals = np.abs(dense).sum(axis=1)
    assert np.allclose(totals[totals > 0], 1.0, atol=1e-5)  # every neuron's inputs sum to one
    assert wiring.digest == data.assemble(*tables(), "full", 5).digest
    assert wiring.digest != tiny_wiring(seed=1).digest
    wide = wiring.matrix.copy()  # the same graph, stored the way another scipy might
    wide.indices, wide.indptr = wide.indices.astype(np.int64), wide.indptr.astype(np.int64)
    assert digest_of(wide, superclass, klass) == wiring.digest
    relabelled = klass.copy()
    relabelled[0] = "gustatory"  # what a neuron is decides what drives it: part of the hash
    assert digest_of(wiring.matrix, superclass, relabelled) != wiring.digest
    assert wiring.matrix.has_sorted_indices  # one layout, wherever the graph came from
    assert make_network(wiring) is make_network(wiring)  # and one network for all its brains
    assert make_network(wiring, substeps=1) is not make_network(wiring)


def test_the_central_graph_leaves_the_optic_lobes_out():
    central = tiny_wiring("central")
    assert central.n == tiny_wiring().n - 44 and "ol_intrinsic" not in set(central.superclass)
    assert len(central.where(superclass=("descending_neuron",))) == 80
    assert len(central.where(klass=("olfactory",), superclass=("cb_endocrine",))) == 45
    with pytest.raises(ValueError, match="unknown graph"):
        data.assemble(*tables(), variant="thorax")


def test_the_control_shuffles_who_receives_what_and_nothing_else():
    wiring = tiny_wiring()
    control = wiring.randomised(3)
    assert control.connections == wiring.connections and control.control == 3
    assert control.digest != wiring.digest and control.digest == wiring.randomised(3).digest
    assert control.digest != wiring.randomised(4).digest
    assert np.array_equal(np.sort(control.matrix.data), np.sort(wiring.matrix.data))
    rows = lambda w: np.sort(np.asarray(abs(w.matrix).sum(axis=1)).ravel())  # noqa: E731
    assert np.allclose(rows(control), rows(wiring))
    assert control.describe()["control"] == 3 and wiring.describe()["control"] is None
    order = _shuffled(np.arange(wiring.n), SplitMix64(3))  # our generator, not numpy's
    assert np.array_equal(control.matrix.toarray(), wiring.matrix.toarray()[order])
    assert control.matrix.has_sorted_indices


def test_build_reads_the_published_tables_and_caches_the_graph(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.feather as feather

    monkeypatch.setenv("NEUROGARDEN_CACHE", str(tmp_path))
    data.forget()
    cache = data.cache_dir()
    with pytest.raises(ValueError, match="connectome fetch"):
        data.build("full", 1)
    ids, superclass, klass, sign, pre, post, synapses = tables()
    orphan = np.array([7, 8])  # bodies that are not traced neurons, and pairs that touch them
    annotations = pa.table(
        {
            "bodyId": np.concatenate([ids[::-1], orphan]),  # not sorted in the file
            "status": ["Traced"] * len(ids) + ["Orphan", None],
            "superclass": [*superclass[::-1], None, None],
            "class": [None if k == "none" else k for k in klass[::-1]] + [None, None],
        }
    )
    nt = np.where(sign > 0, "acetylcholine", "gaba")
    transmitters = pa.table(
        {"body": np.concatenate([ids, orphan]), "consensus_nt": [*nt, "dopamine", None]}
    )
    weights = pa.table(
        {
            "body_pre": np.concatenate([pre, [7, ids[0]]]),
            "body_post": np.concatenate([post, [ids[1], 8]]),
            "weight": np.concatenate([synapses, [50, 50]]),
        }
    )
    cache.mkdir(parents=True)
    feather.write_feather(annotations, cache / data.FILES["annotations"])
    feather.write_feather(transmitters, cache / data.FILES["transmitters"])
    feather.write_feather(weights, cache / data.FILES["weights"], chunksize=1000)

    said = []
    built = data.build("full", 1, on_progress=said.append)
    assert built.digest == tiny_wiring().digest  # the same graph as from the tables directly
    assert any("traced neurons" in line for line in said)
    data.forget()
    assert data.load("full", 1).digest == built.digest
    assert data.load("full", 1) is data.load("full", 1)  # one copy per process
    assert data.load("full", 1, control=2).digest == built.randomised(2).digest
    report = data.info()
    assert report["graphs"][0]["neurons"] == built.n and report["sources"]["weights"] > 0
    with pytest.raises(ValueError, match="connectome build"):
        data.load("central", 5)
    data.forget()


def test_a_half_written_cache_is_refused_not_misread(cache, capsys):
    graph_path, neurons_path = data.graph_paths("full", 1)
    assert {path.name for path in cache.iterdir()} == {graph_path.name, neurons_path.name}
    table = neurons_path.read_bytes()
    data.save(tiny_wiring(seed=9))  # a rebuild from other data…
    neurons_path.write_bytes(table)  # …that stopped between the two files
    data.forget()
    with pytest.raises(ValueError, match="do not match.*connectome build"):
        data.load("full", 1)
    graph_path.write_bytes(graph_path.read_bytes()[:100])
    with pytest.raises(ValueError, match="cannot be read.*connectome build"):
        data.load("full", 1)
    neurons_path.write_bytes(b"")
    with pytest.raises(ValueError, match="cannot be read"):
        data.load("full", 1)
    assert data.info()["broken"] == [neurons_path.name] and data.info()["graphs"] == []
    assert cli.main(["connectome", "info"]) == 0
    assert f"{neurons_path.name}: cannot be read" in capsys.readouterr().out


class Download:
    """What urlopen gives back: headers, and a body that may stop early."""

    def __init__(self, body: bytes, promised: int) -> None:
        self.body, self.headers = body, {"Content-Length": str(promised)}

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass

    def read(self, size: int) -> bytes:
        chunk, self.body = self.body[:size], self.body[size:]
        return chunk


def test_fetch_keeps_only_whole_files(tmp_path, monkeypatch):
    promised = {"n": 9}
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda url, timeout: Download(b"feather", promised["n"])
    )
    with pytest.raises(OSError, match="stopped at 7 of 9 bytes"):  # the connection dropped
        data.fetch(tmp_path)
    assert not any(path.suffix == ".feather" for path in tmp_path.iterdir())
    promised["n"] = 7
    seen = []
    paths = data.fetch(tmp_path, on_progress=lambda name, done, total: seen.append((done, total)))
    assert [path.read_bytes() for path in paths] == [b"feather"] * 3 and seen[0] == (7, 7)
    monkeypatch.setattr(data.urllib.request, "urlopen", None)  # nothing is fetched twice
    assert data.fetch(tmp_path) == paths


# --- dynamics, senses, readout ------------------------------------------------------------------


def test_the_rate_step_is_the_equation_it_says_it_is():
    wiring = tiny_wiring()
    network = RateNetwork(wiring, substeps=2, leak=0.5)
    rng = np.random.default_rng(0)
    rate = rng.random(wiring.n, dtype=np.float32)
    current = rng.random(wiring.n, dtype=np.float32)
    dense = wiring.matrix.toarray()
    expected = rate
    for _ in range(2):
        expected = 0.5 * expected + 0.5 * np.tanh(3.0 * (dense @ expected) + current)
    assert np.allclose(network.step(rate, current, 3.0), expected, atol=1e-5)
    assert not network.step(network.zeros(), network.zeros(), 3.0).any()  # silence stays silent
    # two flies stepped together are the two flies stepped apart
    rates = np.stack([rate, rate[::-1]], axis=1)
    currents = np.stack([current, current[::-1]], axis=1)
    together = network.step(rates, currents, np.array([3.0, 1.0], np.float32))
    assert np.allclose(together[:, 0], expected, atol=1e-5)
    assert np.allclose(together[:, 1], network.step(rate[::-1], current[::-1], 1.0), atol=1e-5)
    with pytest.raises(ValueError):
        RateNetwork(wiring, substeps=0)
    with pytest.raises(ValueError, match="unknown backend"):
        make_network(wiring, backend="abacus")


def test_the_torch_backend_agrees_with_numpy():
    pytest.importorskip("torch")
    wiring = tiny_wiring()
    rng = np.random.default_rng(1)
    rate = rng.random((wiring.n, 3), dtype=np.float32)
    current = rng.random((wiring.n, 3), dtype=np.float32)
    gain = np.array([1.0, 2.0, 4.0], np.float32)
    numpy_step = make_network(wiring).step(rate, current, gain)
    torch_step = make_network(wiring, backend="torch", device="cpu").step(rate, current, gain)
    assert np.allclose(numpy_step, torch_step, atol=1e-4)


def test_every_feature_drives_its_own_seeded_group_of_neurons():
    wiring = tiny_wiring()
    encoding = Encoding.for_wiring(wiring, seed=0)
    assert len(encoding.groups) == TINY_SIZE and all(len(group) > 0 for group in encoding.groups)
    everyone = np.concatenate(encoding.groups)
    assert len(set(everyone.tolist())) == len(everyone)  # no neuron hears two features
    assert set(wiring.klass[encoding.groups[0]]) == {"olfactory"}  # fruit smell, own tile
    assert set(wiring.klass[encoding.groups[5]]) == {"hygrosensory"}
    assert set(wiring.klass[encoding.groups[16]]) == {"gustatory"}
    assert set(wiring.superclass[encoding.groups[24]]) == {"visual_projection"}
    again = Encoding.for_wiring(wiring, seed=0)
    other = Encoding.for_wiring(wiring, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(encoding.groups, again.groups, strict=True))
    assert any(not np.array_equal(a, b) for a, b in zip(encoding.groups, other.groups, strict=True))
    features = np.arange(TINY_SIZE, dtype=np.float32)
    current = encoding.current(features, np.full(TINY_SIZE, 2.0, np.float32), wiring.n)
    assert np.all(current[encoding.groups[7]] == 14.0) and current.sum() == pytest.approx(
        sum(2.0 * index * len(group) for index, group in enumerate(encoding.groups))
    )
    bare = data.assemble(*tables(), variant="full", min_synapses=1)
    bare.klass[:] = "none"
    bare.superclass[:] = "cb_intrinsic"
    with pytest.raises(ValueError, match="too few neurons"):
        Encoding.for_wiring(bare)


def test_the_descending_neurons_are_pooled_into_balanced_signed_buckets():
    wiring = tiny_wiring()
    projection = Projection.for_wiring(wiring, seed=0)
    assert len(projection.descending) == 80
    counts = np.bincount(projection.bucket, minlength=POOLED)
    assert counts.min() >= 1 and counts.max() - counts.min() <= 1
    rate = np.random.default_rng(2).random(wiring.n, dtype=np.float32)
    pooled = projection.pool(rate)
    assert pooled.shape == (POOLED,)
    both = projection.pool(np.stack([rate, 2 * rate], axis=1))
    assert np.allclose(both[:, 0], pooled, atol=1e-6) and np.allclose(both[:, 1], 2 * pooled)
    few = data.assemble(*tables(), variant="full", min_synapses=1)
    few.superclass[few.superclass == "descending_neuron"] = "cb_intrinsic"
    with pytest.raises(ValueError, match="descending neurons"):
        Projection.for_wiring(few)


# --- the brain -----------------------------------------------------------------------------------


def trainable(**options):
    return ConnectomeTrainable(graph="full", min_synapses=1, temperature=0.5, **options)


def seeded_brain(cache, seed=0):
    spec = trainable()
    return spec.brain(spec.initial(np.random.default_rng(seed)))


def test_a_genome_round_trips_and_is_not_mistaken_for_another_brains(tmp_path, cache):
    spec = trainable()
    vector = spec.initial(np.random.default_rng(0))
    genome = spec.genome(vector)
    assert genome.size == spec.size == 481 and np.array_equal(genome.to_vector(), vector)
    assert genome.gain == pytest.approx(0.5) and np.all(genome.gains == 1.0)
    loud = spec.genome(vector)
    loud.log_gain[:] = np.log(3.0)  # however it was bred, the network stays a contraction:
    assert loud.gain == 1.0  # above 1 it would stop forgetting how a life began
    with pytest.raises(ValueError, match="expected 481"):
        genome.with_vector(np.zeros(3, np.float32))
    path = genome.save(tmp_path / "cns", spec.describe())
    assert path.name == "cns.npz"
    loaded, meta = ConnectomeGenome.load(path)
    assert np.array_equal(loaded.to_vector(), vector)
    assert meta["brain"] == "connectome" and meta["digest"] == data.load("full", 1).digest
    assert (meta["graph"], meta["min_synapses"], meta["substeps"]) == ("full", 1, 3)
    assert meta["model_version"] == 2
    later = genome.save(tmp_path / "later.npz", {**spec.describe(), "model_version": 3})
    with pytest.raises(ValueError, match="connectome model 3"):
        ConnectomeGenome.load(later)  # other senses or readout: the numbers mean other things
    cut = tmp_path / "cut.npz"
    cut.write_bytes(path.read_bytes()[:60])
    with pytest.raises(ValueError, match="not a connectome brain"):
        ConnectomeGenome.load(cut)
    with pytest.raises(ValueError, match="not an evolved brain"):
        Genome.load(cut)
    assert weights_kind(cut) == "evolved"
    cut.write_bytes(b"")  # what a full disk leaves behind
    with pytest.raises(ValueError, match="not a connectome brain"):
        ConnectomeGenome.load(cut)
    with pytest.raises(ValueError, match="not an evolved brain"):
        Genome.load(cut)
    assert weights_kind(cut) == "evolved"
    small = Genome.zeros(4).save(tmp_path / "mlp.npz")
    with pytest.raises(ValueError, match="not a connectome brain"):
        ConnectomeGenome.load(small)
    with pytest.raises(ValueError, match="not an evolved brain"):
        Genome.load(path)
    assert weights_kind(path) == "connectome" and weights_kind(small) == "evolved"


def test_the_brain_is_reproducible_remembers_within_a_life_and_forgets_at_birth(cache):
    seen = [observation(body=(700 - 10 * step, 700, 800, 1000, step)) for step in range(12)]
    first, second = seeded_brain(cache), seeded_brain(cache)
    first.reset(5)
    second.reset(5)
    lived = [first.act(o) for o in seen]
    assert lived == [second.act(o) for o in seen] and all(0 <= a < 7 for a in lived)
    assert first.rate.any()  # the network carries state from tick to tick
    thought = first.thought()
    assert len(thought) == 16 and set(thought) <= set("▁▂▃▄▅▆▇█")
    first.reset(5)
    assert not first.rate.any() and [first.act(o) for o in seen] == lived
    with pytest.raises(ValueError, match="needs the wiring"):
        ConnectomeBrain(genome=first.genome)


def test_weights_find_their_graph_in_the_cache_and_refuse_another(tmp_path, cache):
    spec = trainable()
    path = spec.genome(spec.initial(np.random.default_rng(0))).save(
        tmp_path / "cns.npz", {**spec.describe(), "temperature": 0.5}
    )
    brain = ConnectomeBrain(seed=1, path=path)
    assert brain.wiring is data.load("full", 1) and brain.temperature == 0.5
    assert brain.network.substeps == 3 and 0 <= brain.act(observation()) < 7
    control = trainable(control=7)
    control_path = control.genome(control.initial(np.random.default_rng(0))).save(
        tmp_path / "rnd.npz", control.describe()
    )
    assert ConnectomeBrain(path=control_path).wiring.control == 7  # the file names its control
    data.forget()
    data.save(tiny_wiring(seed=9))  # somebody rebuilt the graph from other data
    with pytest.raises(ValueError, match="bred on graph"):
        ConnectomeBrain(path=path)


def test_a_flock_thinks_in_one_multiply_what_each_would_think_alone(cache):
    seen = [observation(body=(600, 650 - 20 * step, 800, 1000, step)) for step in range(6)]
    alone = [seeded_brain(cache, seed) for seed in (0, 1, 2)]
    grouped = [seeded_brain(cache, seed) for seed in (0, 1, 2)]
    for index, brain in enumerate([*alone, *grouped]):
        brain.reset(index % 3)
    flock = ConnectomeFlock(grouped)
    for step, what in enumerate(seen):
        sees = [what, None if step == 2 else what, what]  # the second fly misses a tick
        expected = [
            None if o is None else brain.act(o) for brain, o in zip(alone, sees, strict=True)
        ]
        assert flock.act(sees) == expected
    assert np.allclose(grouped[1].rate, alone[1].rate, atol=1e-5)
    assert flock.act([None, None, None]) == [None, None, None]
    with pytest.raises(ValueError, match="share one wiring"):
        ConnectomeFlock([grouped[0], trainable(substeps=2).brain(grouped[0].genome.to_vector())])
    with pytest.raises(ValueError, match="at least one"):
        ConnectomeFlock([])


# --- breeding it -----------------------------------------------------------------------------


CONNECTOME = dict(
    brain="connectome", graph="full", min_synapses=1, generations=2, population=4,
    episodes=1, max_steps=30, workers=1,
)  # fmt: skip


def test_evolution_breeds_the_parts_around_the_wiring(tmp_path, cache):
    first = evolve(EvolveConfig(**CONNECTOME))
    second = evolve(EvolveConfig(**CONNECTOME))
    assert isinstance(first.genome, ConnectomeGenome) and len(first.history) == 2
    assert np.array_equal(first.genome.to_vector(), second.genome.to_vector())
    meta = first.meta
    assert (meta["brain"], meta["graph"], meta["control"]) == ("connectome", "full", None)
    assert meta["digest"] == data.load("full", 1).digest and meta["temperature"] == 0.5
    path = first.genome.save(tmp_path / "bred.npz", meta)
    assert ConnectomeBrain(path=path).temperature == 0.5
    on_control = evolve(EvolveConfig(**{**CONNECTOME, "control": 3}))
    assert on_control.meta["control"] == 3 and on_control.meta["digest"] != meta["digest"]
    resumed = evolve(EvolveConfig(**CONNECTOME), start=first.genome, parent=meta)
    assert resumed.meta["total_generations"] == 4
    with pytest.raises(ValueError, match="do not carry over"):  # onto the control's wiring
        evolve(EvolveConfig(**{**CONNECTOME, "control": 3}), start=first.genome, parent=meta)
    with pytest.raises(ValueError, match="not a connectome brain"):
        evolve(EvolveConfig(**CONNECTOME), start=Genome.zeros(4))
    with pytest.raises(ValueError, match="unknown brain"):
        EvolveConfig(brain="oracle")
    with pytest.raises(ValueError, match="graph must be"):
        EvolveConfig(graph="thorax")


# --- learning from a teacher -------------------------------------------------------------------


LESSON = dict(graph="full", min_synapses=1, rounds=2, lives=2, max_steps=40, workers=1, seed=3)


def test_a_teacher_says_how_likely_each_action_is():
    teacher, seen = EvolvedBrain(), observation()
    said = teacher.probabilities(seen)
    assert said.shape == (7,) and said.sum() == pytest.approx(1.0) and np.all(said >= 0)
    assert np.array_equal(said, teacher.probabilities(seen))  # asking moves nothing on
    sure = EvolvedBrain(temperature=0.0)
    assert sure.probabilities(seen).tolist().count(1.0) == 1
    assert int(np.argmax(sure.probabilities(seen))) == sure.act(seen)


def test_fitting_a_readout_finds_the_linear_teacher_behind_the_answers():
    rng = np.random.default_rng(0)
    features = rng.standard_normal((1500, 6))
    answers = np.array([softmax(row) for row in features @ rng.standard_normal((6, 7))])
    w, b, agreement, divergence, settled = fit_readout(features, answers, l2=0.0)
    assert w.shape == (6, 7) and b.shape == (7,) and settled
    assert agreement > 0.98 and divergence < 1e-3
    _, _, _, held_back, _ = fit_readout(features, answers, l2=10.0)  # a readout kept near zero
    assert held_back > 10 * divergence
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(distil_module, "_FIT_STEPS", 2)  # cut short: it says so
        assert not fit_readout(features, answers, l2=0.0)[4]
    # Loud features and faint ones are heard alike: the fit is the same whatever their units.
    units = 10.0 ** rng.uniform(-4, 1, size=6)
    scaled_w, scaled_b, same_agreement, same_divergence, _ = fit_readout(
        features * units, answers, l2=1e-4
    )
    w, b, agreement, divergence, _ = fit_readout(features, answers, l2=1e-4)
    assert same_agreement == agreement and same_divergence == pytest.approx(divergence, rel=1e-3)
    assert np.allclose((features * units) @ scaled_w + scaled_b, features @ w + b, atol=1e-3)
    sure = np.eye(7)[answers.argmax(axis=1)]  # a teacher that never doubts, and never rests:
    sure[:, 6] = 0  # the bias of the action it never takes stays finite
    sure[sure.sum(axis=1) == 0, 0] = 1
    _, b, _, _, settled = fit_readout(features, sure, l2=1e-4)
    assert settled and np.all(np.isfinite(b)) and np.abs(b).max() < 100


def test_a_lesson_is_what_it_says_tick_for_tick(cache):
    config = DistilConfig(**{**LESSON, "rounds": 1, "lives": 1})
    taught = distil(config)
    world = int(np.random.default_rng(config.seed).integers(0, 2**31 - 1, size=1)[0])
    blank = config.trainable().blank().to_vector()
    distil_module._setup(config)
    seen, said, lifespan = distil_module._fly((blank, world, True))
    assert [lifespan] == taught.history[0].lifespans and len(seen) == len(said) == lifespan
    # Live the life again: every moment written down is what the readout saw then, and what
    # the teacher said of that same moment; and the life is the teacher's own.
    env = NeuroGardenEnv(max_steps=config.max_steps, any_hour=True)  # as a lesson's lives are
    teacher, student = EvolvedBrain(), config.trainable().brain(blank)
    now, _ = env.reset(seed=world)
    teacher.reset(brain_seed(world))
    student.reset(brain_seed(world))
    for tick in range(lifespan):
        assert np.allclose(said[tick], teacher.probabilities(now))
        student.act(now)
        assert np.array_equal(seen[tick], student.features)
        now, _, terminated, truncated, _ = env.step(teacher.act(now))
    assert terminated or truncated
    assert not np.array_equal(seen[3], seen[4])  # one tick off would be noticed
    # The brain draws from softmax(scores / temperature): exactly the distribution fitted.
    w, b, *_ = fit_readout(seen, said, config.l2)
    genome = taught.genome
    for moment in seen[::7]:
        drawn_from = softmax((moment @ genome.w + genome.b) / config.temperature)
        assert np.allclose(drawn_from, softmax(moment @ w + b), atol=1e-5)
    assert not np.allclose(softmax(seen[0] @ genome.w + genome.b), softmax(seen[0] @ w + b))


def test_distilling_teaches_the_readout_round_by_round(tmp_path, cache):
    first = distil(DistilConfig(**LESSON))
    second = distil(DistilConfig(**LESSON))
    assert np.array_equal(first.genome.to_vector(), second.genome.to_vector())
    teacher_round, student_round = first.history
    assert (teacher_round.flown_by, student_round.flown_by) == ("teacher", "student")
    assert len(student_round.lifespans) == 2 and all(0 < n <= 40 for n in student_round.lifespans)
    assert student_round.ticks > teacher_round.ticks >= 2  # every moment so far is kept
    assert 0 <= student_round.agreement <= 1 and student_round.divergence < np.log(7)
    assert teacher_round.settled and student_round.settled
    spread = distil(DistilConfig(**{**LESSON, "workers": 2}))  # however the lives are flown
    assert np.array_equal(spread.genome.to_vector(), first.genome.to_vector())
    genome = first.genome
    assert genome.w.any() and np.all(genome.gains == 1.0) and genome.gain == pytest.approx(0.5)
    meta = first.meta
    assert (meta["brain"], meta["graph"], meta["temperature"]) == ("connectome", "full", 0.5)
    assert meta["digest"] == data.load("full", 1).digest and meta["total_generations"] == 0
    assert meta["any_hour"] is True  # taught on lives that begin at any hour, as live ones do
    told = meta["distilled"]
    assert (told["rounds_done"], told["teacher"], told["ticks"]) == (
        2,
        "evolved-v1",
        student_round.ticks,
    )
    assert told["teacher_generations"] == 120  # which teacher: the shipped one's ancestry,
    assert len(told["teacher_digest"]) == 16  # and a hash of its weights
    path = genome.save(tmp_path / "taught.npz", meta)
    assert 0 <= ConnectomeBrain(path=path).act(observation()) < 7
    # Evolution carries on from a taught brain, and remembers how it began.
    asked = []
    numpy_rng = np.random.default_rng
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(np.random, "default_rng", lambda seed: asked.append(seed) or numpy_rng(seed))
        bred = evolve(EvolveConfig(**CONNECTOME), start=genome, parent=meta)
    assert bred.meta["total_generations"] == 2 and bred.meta["parent"]["distilled"] == told
    assert bred.meta["any_hour"] is False  # the library takes what its config says…
    assert [0, 0, 2] in asked  # not the lesson's worlds again: the seed is mixed with its rounds
    seen = []
    distil(DistilConfig(**LESSON), on_round=seen.append, checkpoint=lambda r: seen.append(r.meta))
    assert [type(item).__name__ for item in seen] == ["Round", "dict", "Round", "dict"]
    assert seen[1]["distilled"]["rounds_done"] == 1


def test_the_readout_can_see_more_or_fewer_features(tmp_path, cache):
    narrow = distil(DistilConfig(**LESSON, pooled=16, gain=0.9, input_gain=2.0))
    genome = narrow.genome
    assert genome.pooled == 16 and genome.size == 25 + 1 + 16 * 7 + 7
    assert np.all(genome.gains == 2.0) and genome.gain == pytest.approx(0.9)
    assert narrow.meta["pooled"] == 16 and narrow.meta["distilled"]["input_gain"] == 2.0
    path = genome.save(tmp_path / "narrow.npz", narrow.meta)
    brain = ConnectomeBrain(path=path)
    assert brain.features.shape == (16,) and 0 <= brain.act(observation()) < 7
    assert len(brain.projection.descending) == 80 and brain.projection.bucket.max() == 15
    with pytest.raises(ValueError, match="16 pooled features, this run 64"):
        evolve(EvolveConfig(**CONNECTOME), start=genome, parent=narrow.meta)
    bred = evolve(EvolveConfig(**CONNECTOME, pooled=16), start=genome, parent=narrow.meta)
    assert bred.genome.pooled == 16 and bred.meta["pooled"] == 16
    with pytest.raises(ValueError, match="descending neurons"):  # more buckets than neurons
        distil(DistilConfig(**LESSON, pooled=81))


def test_distil_refuses_what_it_cannot_teach(tmp_path, cache):
    for wrong in (dict(rounds=0), dict(temperature=0.0), dict(pooled=0), dict(gain=0.0)):
        with pytest.raises(ValueError, match="must"):
            DistilConfig(**{**LESSON, **wrong})
    with pytest.raises(ValueError, match="stops forgetting how a life began"):
        DistilConfig(**{**LESSON, "gain": 1.5})
    with pytest.raises(ValueError, match="graph must be"):
        DistilConfig(graph="thorax")
    spec = trainable()
    student = spec.genome(spec.initial(np.random.default_rng(0))).save(
        tmp_path / "cns.npz", spec.describe()
    )
    with pytest.raises(ValueError, match="not an evolved brain"):  # a student is no teacher
        distil(DistilConfig(**LESSON, teacher=str(student)))


def test_the_shipped_brain_and_its_control_were_taught_alike():
    """No graph needed: the two files say how they came to be, and the saying must match —
    a control taught differently would be no control."""
    real, how = ConnectomeGenome.load(default_weights("connectome-v1.npz"))
    control, how_control = ConnectomeGenome.load(default_weights("connectome-random-v1.npz"))
    assert real.size == control.size == 481 and not np.array_equal(real.w, control.w)
    assert how["control"] is None and how_control["control"] == 1
    assert how["digest"] != how_control["digest"]
    same = ("brain", "graph", "min_synapses", "neurons", "connections", "model_version", "pooled")
    same += ("substeps", "leak", "encoding_seed", "projection_seed", "temperature")
    assert {key: how[key] for key in same} == {key: how_control[key] for key in same}
    lesson, lesson_control = how["distilled"], how_control["distilled"]
    alike = ("rounds", "lives", "max_steps", "seed", "l2", "map", "gain", "input_gain")
    alike += ("rounds_done", "teacher", "teacher_digest")
    assert {key: lesson[key] for key in alike} == {key: lesson_control[key] for key in alike}
    assert lesson["rounds_done"] == lesson["rounds"]  # a whole lesson, not one cut short
    assert np.array_equal(real.gains, control.gains) and real.gain == control.gain


# --- flying many -------------------------------------------------------------------------------


FAST = dict(port=0, tps=50.0, npcs=[], hello_timeout=0.5)
SHORT = Config(initial_satiety=6, initial_health=5)


def test_a_flock_joins_as_many_owners_and_reports_every_life(cache):
    async def scenario():
        async with Server(ServerConfig(**FAST, config=SHORT)) as server:
            brains = [seeded_brain(cache, seed) for seed in range(3)]
            owners = ["cns-1", "cns-2", "cns-3"]
            told = []
            lives = await fly_flock(server.url, brains, owners, lives=2, on_life=told.append)
            scores = {state.owner: state.lineage for state in server.runner.roster.scores()}
            return lives, told, scores

    lives, told, scores = asyncio.run(scenario())
    assert len(lives) == 6 and lives == told
    assert scores == {"cns-1": 2, "cns-2": 2, "cns-3": 2}
    assert {life.owner for life in lives} == {"cns-1", "cns-2", "cns-3"}
    assert all(life.stats.lifespan > 0 and life.name for life in lives)


class Idler:
    """A brain that only notes what it was seeded with."""

    def __init__(self) -> None:
        self.seeds = []

    def reset(self, seed=None) -> None:
        self.seeds.append(seed)

    def act(self, observation) -> int:
        return 0


def test_a_flock_of_ordinary_brains_and_its_refusals():
    twins = [Idler(), Idler()]

    async def scenario():
        async with Server(ServerConfig(**FAST, config=SHORT)) as server:
            brains = [BRAINS["random"](seed=1), BRAINS["scripted"](seed=2)]
            lives = await fly_flock(server.url, brains, ["a", "b"], lives=1)
            with pytest.raises(ValueError, match="its own owner"):
                await fly_flock(server.url, brains, ["a", "a"])
            with pytest.raises(ValueError, match="as many owners"):
                await fly_flock(server.url, brains, ["a"])
            with pytest.raises(ValueError, match="as many seeds"):
                await fly_flock(server.url, brains, ["a", "b"], seeds=[1])
            await fly_flock(server.url, twins, ["c", "d"], lives=2, seeds=[5, 6])
            return lives

    assert sorted(life.owner for life in asyncio.run(scenario())) == ["a", "b"]
    # Every fly its own stream of chance, every life another: never one fly twice.
    first, second = twins
    assert first.seeds == [brain_seed((5 << 32) ^ 1), brain_seed((5 << 32) ^ 2)]
    assert len(set(first.seeds + second.seeds)) == 4
    nowhere = "ws://127.0.0.1:9"  # nothing to live: nobody is even called
    assert asyncio.run(fly_flock(nowhere, twins, ["c", "d"], lives=0)) == []


# --- the commands --------------------------------------------------------------------------------


def test_connectome_info_and_refusals_without_a_graph(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NEUROGARDEN_CACHE", str(tmp_path / "empty"))
    data.forget()
    assert cli.main(["connectome", "info"]) == 0
    out = capsys.readouterr().out
    assert out.count("missing") == 3 and "no graph built yet" in out
    assert cli.main(["connectome", "build"]) == 1
    assert "connectome fetch" in capsys.readouterr().err
    argv = ["evolve", "--brain", "connectome", "--out", str(tmp_path / "x.npz"), "--workers", "1"]
    assert cli.main(argv) == 1
    assert "connectome build" in capsys.readouterr().err
    assert cli.main(["distil", "--out", str(tmp_path / "x.npz"), "--workers", "1"]) == 1
    assert "connectome build" in capsys.readouterr().err
    assert cli.main(["flock", "--count", "0"]) == 1
    assert "at least 1" in capsys.readouterr().err
    assert watch.main(["--brain", "connectome"]) == 1  # the dojo's own entry point, too
    assert capsys.readouterr().err.startswith("neurogarden: ")
    monkeypatch.setitem(sys.modules, "pyarrow", None)  # scipy alone: the extra is half there
    assert cli.main(["connectome", "build"]) == 1
    assert "install neurogarden[connectome]" in capsys.readouterr().err


def test_evolve_and_flock_commands_on_a_cached_graph(tmp_path, cache, capsys):
    out = str(tmp_path / "cns.npz")
    argv = [
        "evolve", "--brain", "connectome", "--graph", "full", "--min-synapses", "1", "--out", out,
        "--generations", "1", "--population", "2", "--episodes", "1", "--max-steps", "20",
        "--workers", "1",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    printed = capsys.readouterr().out
    assert "evolving a connectome brain on the full graph (400 neurons" in printed
    again = str(tmp_path / "again.npz")
    resume = ["evolve", "--start", out, "--out", again, "--generations", "1", "--population", "2"]
    assert cli.main([*resume, "--episodes", "1", "--max-steps", "20", "--workers", "1"]) == 0
    assert "connectome brain on the full graph" in capsys.readouterr().out  # inherited
    with np.load(again) as file:
        meta = json.loads(str(file["meta"]))
    assert meta["total_generations"] == 2 and meta["parent"]["brain"] == "connectome"
    assert cli.main([*resume, "--control", "1", "--workers", "1"]) == 1
    assert "do not carry over" in capsys.readouterr().err
    taught = str(tmp_path / "taught")
    lesson = ["distil", "--graph", "full", "--min-synapses", "1", "--out", taught, "--pooled", "16"]
    lesson += ["--rounds", "2", "--lives", "1", "--max-steps", "30", "--workers", "1"]
    assert cli.main(lesson) == 0
    printed = capsys.readouterr().out
    assert "teaching a connectome brain on the full graph (400 neurons" in printed
    assert "round 1/2: the teacher flew" in printed and "round 2/2: the student flew" in printed
    assert f"wrote {taught}.npz" in printed
    bred = str(tmp_path / "bred.npz")  # evolution carries on, with the taught brain's shape
    carry_on = ["evolve", "--start", taught + ".npz", "--out", bred, "--generations", "1"]
    carry_on += ["--population", "2", "--episodes", "1", "--max-steps", "20", "--workers", "1"]
    assert cli.main(carry_on) == 0
    # A taught brain is carried on in small steps (the usual ones would undo the lesson) and
    # on lives born at any hour, as it was taught.
    printed = capsys.readouterr().out
    assert "sigma 0.01, learning rate 0.0005, born at any hour" in printed
    with np.load(bred) as file:
        meta = json.loads(str(file["meta"]))
    assert meta["pooled"] == 16 and meta["parent"]["distilled"]["rounds_done"] == 2
    assert (meta["sigma"], meta["learning_rate"], meta["any_hour"]) == (0.01, 0.0005, True)
    further = [*carry_on[:2], bred, *carry_on[3:]]  # and what was bred from it keeps them
    assert cli.main(further) == 0
    assert "sigma 0.01, learning rate 0.0005" in capsys.readouterr().out
    assert cli.main([*carry_on, "--sigma", "0.2", "--no-any-hour"]) == 0  # unless told otherwise
    assert "sigma 0.2, learning rate 0.0005, born at dawn" in capsys.readouterr().out
    assert cli.main([*lesson, "--teacher", out]) == 1  # a connectome brain is no teacher
    assert "not an evolved brain" in capsys.readouterr().err
    shuffled = str(tmp_path / "mine.npz")  # the control, bred under a name of its own
    assert cli.main([*argv[:-2], "--control", "1", "--out", shuffled, "--workers", "1"]) == 0
    capsys.readouterr()
    assert cli.main(["connectome", "info"]) == 0
    assert "graph full (>= 1 synapses): 400 neurons" in capsys.readouterr().out

    async def scenario():
        async with Server(ServerConfig(**FAST, config=SHORT)) as server:
            argv = ["flock", "--weights", out, "--count", "2", "--lives", "1", "--url", server.url]
            first = await asyncio.to_thread(cli.main, argv)
            argv = ["flock", "--weights", shuffled, "--count", "1", "--lives", "1"]
            return first, await asyncio.to_thread(cli.main, [*argv, "--url", server.url])

    assert asyncio.run(scenario()) == (0, 0)
    printed = capsys.readouterr().out
    assert "flying 2 connectome flies as cns-1 … cns-2" in printed  # owners from the file's name
    assert "cns-1's" in printed and "cns-2's" in printed
    assert (
        "flying 1 connectome-random flies as mine-1 … mine-1" in printed and "mine-1's" in printed
    )


def test_a_server_whose_npc_cannot_think_makes_no_world(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROGARDEN_CACHE", str(tmp_path / "empty"))  # no graph to think with
    data.forget()
    path = str(tmp_path / "world.db")
    config = ServerConfig(**{**FAST, "npcs": [("connectome", 1)]}, archive=path)

    async def scenario():
        async with Server(config):
            pass

    with pytest.raises((ValueError, OSError)):
        asyncio.run(scenario())
    assert not (tmp_path / "world.db").exists()  # refused before a world was made for it
