import json

import numpy as np
import pytest

from neurogarden.engine.config import Config
from neurogarden.engine.fruit import add_fruit
from neurogarden.engine.snapshot import restore, snapshot, state_hash
from neurogarden.engine.state import Agent, new_state
from neurogarden.engine.tiles import parse_map

MAP = """
#####
#.N~#
#T..#
#####
"""


def make():
    state = new_state(parse_map(MAP), Config(fruit_bites=3), seed=11)
    state.rng.next_u64()
    state.tick = 42
    add_fruit(state, 1, 1)
    state.resource_age[1, 1] = 300
    agent = Agent(1, "fly", 2, 1, 3, 650, 640, 630, 990, age=42, bumped=True)
    state.agents[1] = agent
    state.occupant[1, 2] = 1
    state.next_agent_id = 2
    return state


def test_snapshot_survives_json():
    data = snapshot(make())
    assert json.loads(json.dumps(data)) == data


def test_restore_reproduces_the_state_exactly():
    original = make()
    copy = restore(json.loads(json.dumps(snapshot(original))))
    assert state_hash(copy) == state_hash(original)
    assert copy.config == original.config
    assert copy.rng.state == original.rng.state
    assert copy.agents == original.agents
    assert copy.trees == original.trees
    for name in ("terrain", "resource_kind", "resource_amount", "resource_age", "occupant"):
        restored, source = getattr(copy, name), getattr(original, name)
        assert restored.dtype == source.dtype, name
        assert np.array_equal(restored, source), name
        assert restored.flags.writeable, name


def test_hash_is_64_hex_characters_and_stable():
    assert len(state_hash(make())) == 64
    assert state_hash(make()) == state_hash(make())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: setattr(s, "tick", 43),
        lambda s: s.rng.next_u64(),
        lambda s: s.terrain.__setitem__((0, 0), 1),
        lambda s: s.resource_kind.__setitem__((2, 2), 1),
        lambda s: s.resource_age.__setitem__((1, 1), 301),
        lambda s: s.resource_amount.__setitem__((1, 1), 2),
        lambda s: s.occupant.__setitem__((2, 2), 5),
        lambda s: setattr(s.agents[1], "x", 3),
        lambda s: setattr(s.agents[1], "y", 2),
        lambda s: setattr(s.agents[1], "facing", 0),
        lambda s: setattr(s.agents[1], "satiety", 649),
        lambda s: setattr(s.agents[1], "alive", False),
        lambda s: setattr(s.agents[1], "health", 989),
        lambda s: setattr(s.agents[1], "age", 43),
        lambda s: setattr(s.agents[1], "bumped", False),
        lambda s: setattr(s, "next_agent_id", 3),
        # body: BODIES has only one registered body ("fly"), so there is no second valid name
        # to mutate agents[1].body to; get_body(...).index is still packed into the hash bytes.
    ],
)
def test_hash_covers_every_part_of_the_dynamic_state(mutate):
    state = make()
    before = state_hash(state)
    mutate(state)
    assert state_hash(state) != before


def test_derived_data_is_not_hashed():
    state = make()
    before = state_hash(state)
    state.fruit_version += 5
    assert state_hash(state) == before


def test_restore_rejects_other_rules_versions():
    data = snapshot(make())
    data["rules_version"] += 1
    with pytest.raises(ValueError):
        restore(data)
