import numpy as np

from neurogarden.engine.body import observation_spec
from neurogarden.engine.config import Config
from neurogarden.engine.fruit import add_fruit, remove_fruit
from neurogarden.engine.senses import ScentFields, observe, scent_field, vision_radius
from neurogarden.engine.state import Agent, new_state
from neurogarden.engine.tiles import Terrain, parse_map

MAP = """
#########
#.......#
#.#####.#
#...N..~#
#########
"""


def make(x=1, y=1, tick=200, **overrides):
    state = new_state(parse_map(MAP), Config(**overrides), seed=1)
    state.tick = tick
    agent = Agent(1, "fly", x, y, 2, 700, 600, 500, 900, age=7)
    state.agents[1] = agent
    state.occupant[y, x] = 1
    return state, agent


def test_scent_falls_off_linearly_with_path_distance():
    terrain = parse_map(".....").terrain
    field = scent_field(terrain, [(0, 0)], 4)
    assert field.dtype == np.int16
    assert field[0].tolist() == [1000, 750, 500, 250, 0]


def test_rock_blocks_scent_so_distance_is_a_path():
    terrain = parse_map(MAP).terrain
    field = scent_field(terrain, [(3, 1)], 20)
    assert field[2, 3] == 0  # rock never smells
    assert field[1, 5] == 1000 * (20 - 2) // 20
    # (3, 3) is 2 tiles from the source as the crow flies, but 6 around the wall
    assert field[3, 3] == 1000 * (20 - 6) // 20


def test_nearest_source_wins():
    terrain = parse_map(".......").terrain
    field = scent_field(terrain, [(0, 0), (6, 0)], 10)
    assert field[0].tolist() == [1000, 900, 800, 700, 800, 900, 1000]


def test_no_sources_means_an_all_zero_field():
    assert not scent_field(parse_map("...").terrain, [], 5).any()


def test_observation_matches_the_body_spec():
    state, agent = make()
    observation = observe(state, ScentFields(state), agent)
    spec = observation_spec("fly")
    assert set(observation) == set(spec)
    for name, channel in spec.items():
        assert observation[name].shape == channel.shape, name
        assert observation[name].dtype == np.dtype(channel.dtype), name
        assert observation[name].min() >= channel.low and observation[name].max() <= channel.high


def test_smell_rows_are_fruit_humidity_nest_and_columns_own_n_e_s_w():
    state, agent = make(x=6, y=3, smell_range_humidity=10, smell_range_nest=16)
    smell = observe(state, ScentFields(state), agent)["smell"]
    assert smell[0].tolist() == [0, 0, 0, 0, 0]  # no fruit anywhere
    # water at (7, 3): own tile d=1, north (6,2) is rock, east is the water itself
    assert smell[1].tolist() == [900, 0, 1000, 0, 800]
    # nest at (4, 3): own tile d=2, west (5,3) d=1, east (7,3) is water d=3
    assert smell[2].tolist() == [875, 0, 812, 0, 937]


def test_fruit_field_is_cached_until_the_fruit_changes():
    state, agent = make(x=2, y=1)
    scents = ScentFields(state)
    assert observe(state, scents, agent)["smell"][0, 0] == 0
    add_fruit(state, 1, 1)
    assert observe(state, scents, agent)["smell"][0].tolist()[:1] == [916]
    cached = scents.current(state)[0]
    assert scents.current(state)[0] is cached  # no recompute without a change
    remove_fruit(state, 1, 1)
    assert observe(state, scents, agent)["smell"][0, 0] == 0


def test_vision_radius_follows_light():
    state, _ = make()
    assert [vision_radius(light, state) for light in (0, 499, 500, 999, 1000)] == [1, 1, 2, 2, 3]


def test_vision_by_day_is_north_up_with_rock_outside_the_map():
    state, agent = make(x=1, y=1)
    add_fruit(state, 3, 1)
    state.occupant[3, 1] = 2  # another agent two tiles south
    vision = observe(state, ScentFields(state), agent)["vision"]
    assert vision[3, 3].tolist() == [Terrain.GROUND, 0, 1]  # centre: self
    assert vision[3, 5].tolist() == [Terrain.GROUND, 1, 0]  # fruit two tiles east
    assert vision[5, 3].tolist() == [Terrain.GROUND, 0, 2]  # other agent two tiles south
    assert vision[2, 3].tolist() == [Terrain.ROCK, 0, 0]  # border rock to the north
    assert vision[0, 0].tolist() == [Terrain.ROCK, 0, 0]  # outside the map, within radius
    assert vision[4, 4].tolist() == [Terrain.ROCK, 0, 0]  # inner wall


def test_vision_at_night_is_void_beyond_radius_one():
    state, agent = make(x=1, y=1, tick=1000)
    add_fruit(state, 3, 1)
    vision = observe(state, ScentFields(state), agent)["vision"]
    assert vision[3, 5].tolist() == [Terrain.VOID, 0, 0]
    assert vision[3, 4].tolist() == [Terrain.GROUND, 0, 0]
    assert vision[0, 0].tolist() == [Terrain.VOID, 0, 0]


def test_touch_body_and_env():
    state, agent = make(x=6, y=3)
    agent.bumped = True
    add_fruit(state, 6, 3)
    observation = observe(state, ScentFields(state), agent)
    assert observation["touch"].tolist() == [1, 1, 1, 0]
    assert observation["body"].tolist() == [700, 600, 500, 900, 7]
    assert observation["env"].tolist() == [1000]

    state, agent = make(x=4, y=3, tick=50)
    observation = observe(state, ScentFields(state), agent)
    assert observation["touch"].tolist() == [0, 0, 0, 1]
    assert observation["env"].tolist() == [500]
