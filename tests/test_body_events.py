import pytest

from neurogarden.engine.body import (
    DIRECTIONS,
    MOVE_DIRECTION,
    Action,
    action_names,
    get_body,
    observation_spec,
)
from neurogarden.engine.events import Event, for_agent


def test_fly_has_seven_actions_in_spec_order():
    assert action_names("fly") == (
        "idle",
        "move_n",
        "move_e",
        "move_s",
        "move_w",
        "consume",
        "rest",
    )
    assert [int(a) for a in Action] == [0, 1, 2, 3, 4, 5, 6]


def test_directions_are_n_e_s_w_with_y_growing_south():
    assert DIRECTIONS == ((0, -1), (1, 0), (0, 1), (-1, 0))
    assert MOVE_DIRECTION[Action.MOVE_N] == 0
    assert MOVE_DIRECTION[Action.MOVE_W] == 3


def test_observation_spec_shapes_and_dtypes():
    spec = observation_spec("fly")
    assert {name: (s.shape, s.dtype) for name, s in spec.items()} == {
        "smell": ((3, 5), "int16"),
        "vision": ((7, 7, 3), "uint8"),
        "touch": ((4,), "uint8"),
        "body": ((5,), "int32"),
        "env": ((1,), "int16"),
    }


def test_fly_is_allocentric():
    assert get_body("fly").frame == "allocentric"


def test_unknown_body_is_a_value_error():
    with pytest.raises(ValueError):
        get_body("dragon")


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValueError):
        Event(0, "exploded")


def test_for_agent_strips_coordinates_only():
    event = Event(3, "moved", 1, {"from": (1, 1), "to": (1, 2), "direction": 2})
    stripped = for_agent(event)
    assert stripped.data == {"direction": 2}
    assert (stripped.tick, stripped.type, stripped.agent_id) == (3, "moved", 1)
    assert event.data["from"] == (1, 1)  # original untouched
