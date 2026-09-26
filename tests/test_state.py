import numpy as np

from neurogarden.engine.clock import day_number, is_night, light_at
from neurogarden.engine.config import Config
from neurogarden.engine.state import Agent, new_state
from neurogarden.engine.tiles import Resource, parse_map

MAP = """
#####
#.F~#
#TN.#
#####
"""


def make_state(**overrides):
    return new_state(parse_map(MAP), Config(**overrides), seed=1)


def test_layers_have_spec_dtypes_and_shape():
    state = make_state()
    assert (state.width, state.height) == (5, 4)
    assert state.terrain.dtype == np.uint8
    assert state.resource_kind.dtype == np.uint8
    assert state.resource_amount.dtype == np.int16
    assert state.resource_age.dtype == np.int32
    assert state.occupant.dtype == np.int32


def test_map_fruit_starts_full_and_fresh():
    state = make_state(fruit_bites=3)
    assert state.resource_kind[1, 2] == Resource.FRUIT
    assert state.resource_amount[1, 2] == 3
    assert state.resource_age[1, 2] == 0


def test_walkable_is_ground_and_nest_in_bounds_only():
    state = make_state()
    assert state.walkable(1, 1)  # ground
    assert state.walkable(2, 2)  # nest
    assert not state.walkable(3, 1)  # water
    assert not state.walkable(1, 2)  # tree
    assert not state.walkable(0, 0)  # rock
    assert not state.walkable(-1, 1)
    assert not state.walkable(5, 1)


def test_trees_are_derived_from_terrain():
    assert make_state().trees == ((1, 2),)


def test_living_is_ascending_and_skips_the_dead():
    state = make_state()
    for agent_id, alive in ((2, True), (1, True), (3, False)):
        state.agents[agent_id] = Agent(agent_id, "fly", 1, 1, 2, 700, 700, 800, 1000, alive=alive)
    assert [a.id for a in state.living()] == [1, 2]


def test_state_does_not_alias_the_parsed_map():
    parsed = parse_map(MAP)
    state = new_state(parsed, Config(), seed=0)
    state.terrain[1, 1] = 2
    assert parsed.terrain[1, 1] == 1


def test_light_follows_the_spec_table():
    cfg = Config()
    assert light_at(0, cfg) == 0
    assert light_at(50, cfg) == 500
    assert light_at(100, cfg) == 1000
    assert light_at(699, cfg) == 1000
    assert light_at(700, cfg) == 1000
    assert light_at(750, cfg) == 500
    assert light_at(800, cfg) == 0
    assert light_at(1199, cfg) == 0
    assert light_at(1200 + 50, cfg) == 500  # wraps every day


def test_night_is_light_below_500():
    cfg = Config()
    assert is_night(49, cfg)
    assert not is_night(50, cfg)
    assert not is_night(750, cfg)
    assert is_night(751, cfg)
    assert is_night(1000, cfg)


def test_day_number_starts_at_one():
    cfg = Config()
    assert day_number(0, cfg) == 1
    assert day_number(1199, cfg) == 1
    assert day_number(1200, cfg) == 2
