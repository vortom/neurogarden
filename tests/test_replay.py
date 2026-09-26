import json
from pathlib import Path

import pytest

from neurogarden.engine.config import RULES_VERSION, Config
from neurogarden.engine.replay import ReplayMismatch, record, verify
from neurogarden.engine.rng import SplitMix64

GOLDEN = Path(__file__).parent / "data" / "golden_replay.json"

MAP = """
#######
#..T..#
#.N..~#
#######
"""


def small_replay():
    rng = SplitMix64(4)
    actions = [[rng.randbelow(7)] for _ in range(250)]
    return record(MAP, seed=3, actions=actions, config=Config(fruit_bites=2))


def test_record_stores_everything_needed_and_checkpoints():
    replay = small_replay()
    assert replay["rules_version"] == RULES_VERSION
    assert replay["map"] == MAP
    assert replay["config"]["fruit_bites"] == 2
    assert sorted(replay["checkpoints"], key=int) == ["100", "200", "250"]
    assert json.loads(json.dumps(replay)) == replay


def test_a_recorded_replay_verifies():
    verify(small_replay())


def test_a_changed_action_is_detected():
    replay = small_replay()
    # swap in REST (or IDLE for REST): the energy difference survives to the checkpoint
    replay["actions"][50] = [0 if replay["actions"][50][0] == 6 else 6]
    with pytest.raises(ReplayMismatch) as err:
        verify(replay)
    assert err.value.tick == 100


def test_a_changed_seed_is_detected():
    replay = small_replay()
    replay["seed"] += 1
    with pytest.raises(ReplayMismatch):
        verify(replay)


def test_other_rules_versions_are_rejected():
    replay = small_replay()
    replay["rules_version"] += 1
    with pytest.raises(ValueError):
        verify(replay)


def test_golden_replay_still_reproduces():
    """Fails when engine behaviour changed. If the change is intentional, bump
    RULES_VERSION and regenerate with `uv run python tests/make_golden.py`."""
    verify(json.loads(GOLDEN.read_text(encoding="utf-8")))


def test_checkpoint_every_must_be_positive():
    with pytest.raises(ValueError):
        record(MAP, seed=0, actions=[[0]], checkpoint_every=0)


def test_explicit_empty_spawns_are_respected_not_replaced_by_the_default_fly():
    replay = record(MAP, seed=0, actions=[[], []], spawns=[])
    assert replay["spawns"] == []
    verify(replay)  # zero agents, zero actions per tick: nothing to zip


def test_verify_rejects_a_replay_whose_action_count_does_not_match_the_spawns():
    replay = small_replay()
    replay["actions"][0] = [*replay["actions"][0], 0]  # one spawn, two actions on tick 0
    with pytest.raises(ValueError):
        verify(replay)
