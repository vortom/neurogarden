"""The evolved brain: its features, its genome, the strategy that finds it, and its thoughts."""

import json

import numpy as np
import pytest

from neurogarden.brains import BRAINS, EvolvedBrain, Genome
from neurogarden.brains.base import think
from neurogarden.brains.evolved import default_weights_path, scores
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
    with np.load(path) as data:
        stale = {name: data[name] for name in ("w1", "b1", "w2", "b2")}
    np.savez(path, **stale, meta=json.dumps({"features_version": FEATURES_VERSION + 1}))
    with pytest.raises(ValueError, match="features_version"):
        Genome.load(path)


def test_the_brain_acts_by_the_highest_score_and_thinks_in_sparklines():
    genome = Genome.zeros(4)
    genome.b2[3] = 1.0  # move_w wins whatever is seen
    brain = EvolvedBrain(genome=genome)
    brain.reset(0)
    assert brain.act(observation()) == 3
    assert scores(genome, tiny_features(observation())).argmax() == 3
    thought = brain.thought()
    assert thought == "▅" * 4  # a silent hidden layer sits in the middle
    genome.w1[:, 0] = 5.0  # the first neuron fires on anything
    brain.act(observation())
    assert brain.thought()[0] == "█" and think(brain) == brain.thought()
    assert len(think(brain)) <= SAY_MAX
    import re

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

    assert think(Mute()) is None
    assert think(Loud()) == "x" * SAY_MAX
    assert think(Broken()) is None


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


def test_evolve_is_deterministic_and_lives_are_reproducible():
    first = evolve(SMALL)
    second = evolve(SMALL)
    assert len(first.history) == 2 and first.history[0].seconds > 0
    assert np.array_equal(first.genome.to_vector(), second.genome.to_vector())
    assert first.meta["generations"] == 2 and first.meta["population"] == 4
    env = NeuroGardenEnv(max_steps=40)
    assert live(first.genome, env, seed=9) == live(first.genome, env, seed=9) <= 40 * 2
    resumed = evolve(SMALL, start=first.genome)
    assert not np.array_equal(resumed.genome.to_vector(), first.genome.to_vector())


def test_evolve_config_is_checked():
    with pytest.raises(ValueError, match="even"):
        EvolveConfig(population=3)
    with pytest.raises(ValueError, match="fitness"):
        EvolveConfig(fitness="glory")
    with pytest.raises(ValueError, match="positive"):
        EvolveConfig(sigma=0)


def test_the_shipped_brain_loads_and_is_in_the_registry():
    assert BRAINS["evolved"] is EvolvedBrain
    assert default_weights_path().is_file()
    brain = BRAINS["evolved"](seed=0)
    assert brain.meta["features_version"] == FEATURES_VERSION and brain.meta["generations"] > 0
    brain.reset(1)
    assert 0 <= brain.act(observation()) < 7
