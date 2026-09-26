from neurogarden.engine.config import Config
from neurogarden.engine.fruit import (
    add_fruit,
    fruit_candidates,
    fruit_near,
    remove_fruit,
    seed_initial_fruit,
    update_fruit,
)
from neurogarden.engine.state import new_state
from neurogarden.engine.tiles import Resource, parse_map

ORCHARD = """
#######
#.....#
#..T..#
#..N..#
#######
"""


def make_state(map_text=ORCHARD, seed=1, **overrides):
    return new_state(parse_map(map_text), Config(**overrides), seed)


def fruit_tiles(state):
    ys, xs = (state.resource_kind == Resource.FRUIT).nonzero()
    return sorted(zip(xs.tolist(), ys.tolist(), strict=True))


def test_add_and_remove_bump_the_fruit_version():
    state = make_state()
    add_fruit(state, 1, 1)
    assert state.resource_amount[1, 1] == state.config.fruit_bites
    assert state.fruit_version == 1
    remove_fruit(state, 1, 1)
    assert state.resource_kind[1, 1] == Resource.NONE
    assert state.fruit_version == 2


def test_candidates_are_ground_without_resource_in_row_major_order():
    state = make_state(tree_radius=1)
    add_fruit(state, 2, 1)
    # radius 1 around the tree at (3, 2); (3, 3) is the nest, so not ground
    assert fruit_candidates(state, 3, 2) == [(3, 1), (4, 1), (2, 2), (4, 2), (2, 3), (4, 3)]


def test_fruit_near_counts_fruit_of_any_origin():
    state = make_state(tree_radius=1)
    add_fruit(state, 2, 1)
    add_fruit(state, 5, 3)  # outside radius 1
    assert fruit_near(state, 3, 2) == 1


def test_initial_fruit_is_deterministic_per_seed():
    a, b, c = make_state(seed=3), make_state(seed=3), make_state(seed=4)
    for state in (a, b, c):
        seed_initial_fruit(state)
    assert len(fruit_tiles(a)) == a.config.tree_initial_fruit
    assert fruit_tiles(a) == fruit_tiles(b)
    assert a.rng.state == b.rng.state
    assert a.rng.state != c.rng.state


def test_fruit_rots_at_its_lifetime():
    state = make_state(fruit_lifetime=3, fruit_spawn_permille=0)
    add_fruit(state, 1, 1)
    events = []
    update_fruit(state, events)
    update_fruit(state, events)
    assert fruit_tiles(state) == [(1, 1)]
    update_fruit(state, events)
    assert fruit_tiles(state) == []
    assert [(e.type, e.data) for e in events] == [("fruit_rotted", {"x": 1, "y": 1})]


def test_tree_at_its_maximum_draws_nothing():
    state = make_state(tree_max_fruit=1, tree_initial_fruit=1, fruit_spawn_permille=1000)
    add_fruit(state, 2, 2)
    before = state.rng.state
    update_fruit(state, [])
    assert state.rng.state == before
    assert fruit_tiles(state) == [(2, 2)]


def test_spawn_always_happens_at_1000_permille_and_never_at_0():
    sure = make_state(fruit_spawn_permille=1000)
    events = []
    update_fruit(sure, events)
    assert len(fruit_tiles(sure)) == 1
    assert [e.type for e in events] == ["fruit_spawned"]

    never = make_state(fruit_spawn_permille=0)
    update_fruit(never, [])
    assert fruit_tiles(never) == []


def test_no_candidate_means_one_draw_and_no_spawn():
    state = make_state("###\n#T#\n###", fruit_spawn_permille=1000)
    before = state.rng.state
    events = []
    update_fruit(state, events)
    assert events == []
    state_after_one_draw = type(state.rng)(0)
    state_after_one_draw.state = before
    state_after_one_draw.randbelow(1000)
    assert state.rng.state == state_after_one_draw.state
