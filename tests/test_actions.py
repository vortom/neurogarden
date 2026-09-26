from neurogarden.engine.actions import apply_action, water_adjacent
from neurogarden.engine.body import Action
from neurogarden.engine.config import Config
from neurogarden.engine.fruit import add_fruit
from neurogarden.engine.state import Agent, new_state
from neurogarden.engine.tiles import Resource, parse_map

MAP = """
######
#...~#
#.N..#
#T...#
######
"""


def make(x=2, y=1, satiety=500, hydration=500, **overrides):
    state = new_state(parse_map(MAP), Config(**overrides), seed=1)
    agent = Agent(1, "fly", x, y, 2, satiety, hydration, 800, 1000)
    state.agents[1] = agent
    state.occupant[y, x] = 1
    return state, agent


def test_move_into_free_ground_updates_position_occupant_and_facing():
    state, agent = make()
    events = []
    assert apply_action(state, agent, Action.MOVE_W, events) == Action.MOVE_W
    assert (agent.x, agent.y, agent.facing) == (1, 1, 3)
    assert state.occupant[1, 2] == 0 and state.occupant[1, 1] == 1
    assert [(e.type, e.data) for e in events] == [
        ("moved", {"from": (2, 1), "to": (1, 1), "direction": 3})
    ]


def test_blocked_moves_bump_but_still_turn():
    cases = (  # x, y, action, facing afterwards; blocked by rock, water, tree, rock
        (1, 1, Action.MOVE_N, 0),
        (3, 1, Action.MOVE_E, 1),
        (1, 2, Action.MOVE_S, 2),
        (1, 1, Action.MOVE_W, 3),
    )
    for x, y, action, facing in cases:
        state, agent = make(x=x, y=y)
        events = []
        apply_action(state, agent, action, events)
        assert (agent.x, agent.y) == (x, y)
        assert agent.bumped
        assert agent.facing == facing
        assert [(e.type, e.data) for e in events] == [("bumped", {"direction": facing})]


def test_out_of_bounds_blocks_like_rock():
    state = new_state(parse_map("..\n.."), Config(), seed=1)
    agent = Agent(1, "fly", 0, 0, 2, 500, 500, 800, 1000)
    state.agents[1] = agent
    state.occupant[0, 0] = 1
    apply_action(state, agent, Action.MOVE_N, [])
    assert (agent.x, agent.y) == (0, 0) and agent.bumped


def test_occupied_tile_blocks_movement():
    state, agent = make()
    state.occupant[1, 1] = 2
    apply_action(state, agent, Action.MOVE_W, [])
    assert (agent.x, agent.y) == (2, 1) and agent.bumped


def test_bumped_is_cleared_by_the_next_action():
    state, agent = make(x=1, y=1)
    apply_action(state, agent, Action.MOVE_N, [])
    assert agent.bumped
    apply_action(state, agent, Action.IDLE, [])
    assert not agent.bumped


def test_eating_takes_one_bite_and_removes_the_last_one():
    state, agent = make(fruit_bites=2, fruit_bite_satiety=150)
    add_fruit(state, 2, 1)
    events = []
    apply_action(state, agent, Action.CONSUME, events)
    assert agent.satiety == 650 and state.resource_amount[1, 2] == 1
    apply_action(state, agent, Action.CONSUME, events)
    assert agent.satiety == 800 and state.resource_kind[1, 2] == Resource.NONE
    assert [(e.type, e.data) for e in events] == [
        ("ate", {"bites_left": 1}),
        ("ate", {"bites_left": 0}),
    ]


def test_satiety_and_hydration_are_clamped_at_1000():
    state, agent = make(x=3, y=1, satiety=950, hydration=950)
    add_fruit(state, 3, 1)
    agent.hydration = 1000
    apply_action(state, agent, Action.CONSUME, [])  # eats: satiety lower
    assert agent.satiety == 1000
    agent.hydration = 950
    apply_action(state, agent, Action.CONSUME, [])  # drinks: hydration lower
    assert agent.hydration == 1000


def test_drinking_needs_adjacent_water():
    state, agent = make(x=3, y=1, drink_hydration=150)
    assert water_adjacent(state, 3, 1)
    events = []
    apply_action(state, agent, Action.CONSUME, events)
    assert agent.hydration == 650
    assert [e.type for e in events] == ["drank"]


def test_lower_need_decides_when_both_are_possible_and_tie_eats():
    for satiety, hydration, expected in ((300, 600, "ate"), (600, 300, "drank"), (500, 500, "ate")):
        state, agent = make(x=3, y=1, satiety=satiety, hydration=hydration)
        add_fruit(state, 3, 1)
        events = []
        apply_action(state, agent, Action.CONSUME, events)
        assert [e.type for e in events] == [expected]


def test_consume_with_nothing_available_fails():
    state, agent = make()
    events = []
    apply_action(state, agent, Action.CONSUME, events)
    assert [e.type for e in events] == ["consume_failed"]
    assert (agent.satiety, agent.hydration) == (500, 500)


def test_rest_reports_whether_on_nest():
    state, agent = make(x=2, y=2)
    events = []
    apply_action(state, agent, Action.REST, events)
    assert [(e.type, e.data) for e in events] == [("rested", {"on_nest": True})]


def test_invalid_action_becomes_idle_with_an_event():
    state, agent = make()
    events = []
    assert apply_action(state, agent, 99, events) == Action.IDLE
    assert [(e.type, e.data) for e in events] == [("invalid_action", {"action": "99"})]
    assert (agent.x, agent.y) == (2, 1)
