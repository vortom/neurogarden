"""The evolved brain: its features, its genome, the strategy that finds it, and its thoughts."""

import json
import re

import numpy as np
import pytest

from neurogarden.brains import BRAINS, EvolvedBrain, Genome
from neurogarden.brains.base import think
from neurogarden.brains.evolved import default_weights, output_scores, scores
from neurogarden.dojo import NeuroGardenEnv, TinyObservation
from neurogarden.dojo.evolve import EvolveConfig, _rank_normalise, evolve, initial_genome, live
from neurogarden.dojo.features import FEATURES_VERSION, TINY_SIZE, tiny_features
from neurogarden.dojo.stats import (
    FITNESSES,
    EpisodeStats,
    fitness_forager,
    fitness_lifespan,
    fitness_wellbeing,
)
from neurogarden.engine import Config, World, maps
from neurogarden.protocol.messages import SAY_MAX, TEXT_PATTERN
from neurogarden.server import WorldRunner
from neurogarden.server.ports import LocalPort
from test_brains import observation
from test_runner import FakePort

SMALL = EvolveConfig(generations=2, population=4, episodes=1, max_steps=40, workers=1, hidden=4)


def test_tiny_features_are_what_the_wrapper_feeds_a_network():
    env = NeuroGardenEnv(max_steps=10)
    wrapped = TinyObservation(NeuroGardenEnv(max_steps=10))
    raw, _ = env.reset(seed=3)
    flat, _ = wrapped.reset(seed=3)
    features = tiny_features(raw)
    assert features.shape == (TINY_SIZE,) and features.dtype == np.float32
    assert np.array_equal(features, flat)
    assert 0.0 <= features.min() and features.max() <= 1.0


def test_a_genome_round_trips_through_its_vector_and_a_file(tmp_path):
    rng = np.random.default_rng(1)
    genome = initial_genome(6, rng)
    assert genome.size == TINY_SIZE * 6 + 6 + 6 * 7 + 7
    again = genome.with_vector(genome.to_vector())
    assert all(np.array_equal(a, b) for a, b in zip(genome.parts(), again.parts(), strict=True))
    with pytest.raises(ValueError, match="expected"):
        genome.with_vector(np.zeros(3, np.float32))
    path = tmp_path / "brain.npz"
    genome.save(path, {"generations": 2})
    loaded, meta = Genome.load(path)
    assert np.array_equal(loaded.w1, genome.w1) and loaded.hidden == 6
    assert meta == {"features_version": FEATURES_VERSION, "hidden": 6, "generations": 2}
    assert genome.save(tmp_path / "bare") == tmp_path / "bare.npz"  # numpy adds it; we say so
    with np.load(path) as data:
        stale = {name: data[name] for name in ("w1", "b1", "w2", "b2")}
    np.savez(path, **stale, meta=json.dumps({"features_version": FEATURES_VERSION + 1}))
    with pytest.raises(ValueError, match="features_version"):
        Genome.load(path)
    np.savez(path, **stale, meta=json.dumps({"hidden": 5}))
    with pytest.raises(ValueError, match="meta says 5"):
        Genome.load(path)
    np.savez(path, **{**stale, "b1": np.zeros(3, np.float32)}, meta=json.dumps({}))
    with pytest.raises(ValueError, match="where \\(6,\\) was expected"):
        Genome.load(path)
    np.savez(path, actions=np.zeros(3))  # an export, not a brain
    with pytest.raises(ValueError, match="not an evolved brain"):
        Genome.load(path)
    with pytest.raises(FileNotFoundError):
        Genome.load(tmp_path / "missing.npz")


def test_the_brain_acts_by_the_highest_score_and_thinks_in_sparklines():
    genome = Genome.zeros(4)
    genome.b2[3] = 1.0  # move_w wins whatever is seen
    brain = EvolvedBrain(genome=genome)
    brain.reset(0)
    assert brain.act(observation()) == 3
    features = tiny_features(observation())
    assert scores(genome, features).argmax() == 3
    assert np.array_equal(scores(genome, features), output_scores(genome, np.zeros(4, np.float32)))
    thought = brain.thought()
    assert thought == "▅" * 4  # a silent hidden layer sits in the middle
    genome.w1[:, 0] = 5.0  # the first neuron fires on anything
    brain.act(observation())
    assert brain.thought()[0] == "█" and think(brain) == brain.thought()
    assert len(think(brain)) <= SAY_MAX
    assert re.match(TEXT_PATTERN, think(brain))


def test_think_is_optional_and_never_breaks_a_brain():
    class Mute:
        pass

    class Loud:
        def thought(self):
            return "x" * 100

    class Broken:
        def thought(self):
            raise RuntimeError("no words")

    class Rude:
        def thought(self):
            return "two\nlines\x1b[2J"

    assert think(Mute()) is None
    assert think(Loud()) == "x" * SAY_MAX
    assert think(Broken()) is None
    assert think(Rude()) == "twolines[2J"  # what the protocol would refuse is gone


def test_a_hosted_brain_with_thoughts_shows_them_in_the_frame():
    runner = WorldRunner(World.from_map(maps.load("drosoville"), Config(), seed=1), tps=10.0)
    genome = Genome.zeros(3)
    genome.b2[0] = 1.0  # idles, so the fly lives through the test
    port = LocalPort(
        EvolvedBrain(genome=genome), "npc-evolved-1", runner, runner.catalog.bodies["fly"]
    )
    watcher = FakePort("w", role="spectator")
    runner.add_spectator(watcher)
    runner.attach(port)
    runner.request_join(port)
    for _ in range(7):
        runner.tick()
    shown = next(a for a in watcher.last("frame").agents if a.owner == "npc-evolved-1")
    assert shown.say == "▅" * 3


def test_fitnesses_reward_living_and_living_well():
    idle = EpisodeStats(lifespan=800, mean_wellbeing=0.4)
    fed = EpisodeStats(lifespan=800, mean_wellbeing=0.7)
    assert fitness_lifespan(idle) == fitness_lifespan(fed) == 800.0
    assert fitness_wellbeing(fed) > fitness_wellbeing(idle) > 0
    assert fitness_wellbeing(EpisodeStats()) == 0.0
    bitten = EpisodeStats(lifespan=800, mean_wellbeing=0.7, bites=3)
    assert fitness_forager(bitten) == fitness_wellbeing(fed) + 300.0
    assert set(FITNESSES) == {"lifespan", "wellbeing", "forager"}


def test_rank_normalisation_is_centred_and_silent_on_a_tie():
    weights = _rank_normalise(np.array([3.0, 1.0, 2.0], np.float32))
    assert list(weights) == [0.5, -0.5, 0.0]
    assert not _rank_normalise(np.array([5.0, 5.0, 5.0], np.float32)).any()
    # Three flies dead at the wall, one survivor: the tied ones share a rank, so a mirrored
    # pair among them pulls nowhere instead of a full step in the -epsilon direction.
    tied = _rank_normalise(np.array([799.0, 799.0, 799.0, 1200.0], np.float32))
    assert tied[0] == tied[1] == tied[2] and tied[3] == 0.5 and abs(tied.sum()) < 1e-6


def test_evolve_is_deterministic_and_lives_are_reproducible():
    first = evolve(SMALL)
    second = evolve(SMALL)
    assert len(first.history) == 2 and first.history[0].seconds > 0
    assert np.array_equal(first.genome.to_vector(), second.genome.to_vector())
    assert first.meta["generations"] == 2 and first.meta["population"] == 4
    assert first.meta["parent"] is None and first.meta["total_generations"] == 2
    env = NeuroGardenEnv(max_steps=40)
    assert live(first.genome, env, seed=9) == live(first.genome, env, seed=9) <= 40 * 2
    warm = [live(first.genome, env, seed=9, temperature=0.5) for _ in range(2)]
    assert warm[0] == warm[1]  # a drawn life is as reproducible as a chosen one
    resumed = evolve(SMALL, start=first.genome, parent=first.meta)
    assert not np.array_equal(resumed.genome.to_vector(), first.genome.to_vector())
    assert resumed.meta["total_generations"] == 4 and resumed.meta["parent"]["generations"] == 2
    with pytest.raises(ValueError, match="hidden neurons"):
        evolve(SMALL, start=Genome.zeros(2))


def test_evolve_gives_the_same_weights_whatever_the_worker_count():
    """All the randomness lives in the parent; workers only ever see vectors and seeds."""
    pooled = evolve(EvolveConfig(**{**SMALL.__dict__, "workers": 2}))
    assert np.array_equal(pooled.genome.to_vector(), evolve(SMALL).genome.to_vector())


def test_evolve_config_is_checked():
    with pytest.raises(ValueError, match="even"):
        EvolveConfig(population=3)
    with pytest.raises(ValueError, match="fitness"):
        EvolveConfig(fitness="glory")
    with pytest.raises(ValueError, match="positive"):
        EvolveConfig(sigma=0)
    with pytest.raises(ValueError, match="positive"):
        EvolveConfig(hidden=0)
    with pytest.raises(ValueError):
        EvolveConfig(map="not a map at all")
    with pytest.raises(ValueError, match="workers"):
        EvolveConfig(workers=-3)


def test_the_shipped_brain_loads_and_is_in_the_registry():
    assert BRAINS["evolved"] is EvolvedBrain
    assert default_weights().is_file()
    brain = BRAINS["evolved"](seed=0)
    assert brain.meta["features_version"] == FEATURES_VERSION and brain.meta["generations"] > 0
    assert brain.meta["total_generations"] == 120 and brain.meta["parent"] is not None
    assert brain.temperature == brain.meta["temperature"] == 0.5  # bred warm, lives warm
    assert brain.genome.size == 535 and brain.genome.hidden == 16
    brain.reset(1)
    picks = [brain.act(observation(body=(200, 700, 800, 1000, 50))) for _ in range(20)]
    assert all(0 <= pick < 7 for pick in picks) and len(set(picks)) > 1  # drawn, not fixed


def test_a_mouth_speaks_on_change_and_now_and_then():
    from neurogarden.brains.base import THOUGHT_REPEAT, Mouth

    class Steady:
        text = "same"

        def thought(self):
            return self.text

    brain = Steady()
    mouth = Mouth(brain)
    assert mouth.speak(0) == "same" and mouth.speak(1) is None  # only every THOUGHT_EVERY ticks
    assert mouth.speak(5) is None  # unchanged: nothing new to say
    brain.text = "new"
    assert mouth.speak(10) == "new"
    assert mouth.speak(10 + THOUGHT_REPEAT) == "new"  # said again before the bubble fades
    assert Mouth(object()).speak(0) is None
