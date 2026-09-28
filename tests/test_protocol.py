import json
from pathlib import Path

import pytest

from neurogarden.engine import World, maps
from neurogarden.engine.body import observation_spec
from neurogarden.engine.rng import SplitMix64
from neurogarden.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    build_catalog,
    decode_client,
    decode_server,
    encode,
    export_schema,
    schema_text,
)
from neurogarden.protocol.messages import (
    Action,
    ActionMessage,
    Error,
    ErrorMessage,
    Hello,
    HelloMessage,
    JoinMessage,
    LeaveMessage,
    Replay,
    ReplayMessage,
    Say,
    SayMessage,
)

SCHEMA_FILE = Path(__file__).parent.parent / "protocol" / "v1" / "neurogarden.schema.json"
SERVER_DEFS = (
    "WelcomeMessage", "Welcome", "WorldInfo", "Catalog", "BodyInfo", "ChannelInfo",
    "JoinedMessage", "Joined", "ObservationMessage", "Observation", "Channels", "AgentEvent",
    "DiedMessage", "Died", "Stats", "WorldMessage", "WorldMap", "FrameMessage", "Frame",
    "AgentView", "WorldEvent", "OwnerScore", "ResourceView", "ChronicleMessage", "Chronicle",
    "ErrorMessage", "Error", "ReplayInfoMessage", "ReplayInfo",
)  # fmt: skip
CLIENT_DEFS = (
    "HelloMessage", "Hello", "JoinMessage", "Join", "ActionMessage", "Action",
    "LeaveMessage", "Leave", "SayMessage", "Say", "ReplayMessage", "Replay",
)  # fmt: skip


def hello(**overrides):
    fields = {"protocol": 1, "token": "dev", "owner": "alice", "role": "agent", "client": "t"}
    fields.update(overrides)
    return HelloMessage(payload=Hello(**fields))


def test_envelope_round_trip_for_every_client_message():
    messages = [
        hello(),
        JoinMessage(),
        ActionMessage(payload=Action(tick=3, action=2)),
        LeaveMessage(),
        SayMessage(payload=Say(text="fruit?")),
        ReplayMessage(payload=Replay(owner="alice", lineage=2, speed=8.0)),
    ]
    for message in messages:
        wire = json.loads(encode(message))
        assert set(wire) == {"v", "type", "payload"} and wire["v"] == PROTOCOL_VERSION
        assert decode_client(encode(message)) == message


def test_server_message_round_trip():
    message = ErrorMessage(payload=Error(code="malformed", message="nope", fatal=True))
    assert decode_server(encode(message)) == message


def test_unknown_type_is_ignored_not_rejected():
    assert decode_client('{"v": 1, "type": "dance", "payload": {}}') is None
    assert decode_server('{"v": 1, "type": "weather", "payload": {"rain": 1}}') is None


@pytest.mark.parametrize(
    "text, code",
    [
        ("not json", "malformed"),
        ("[1, 2]", "malformed"),
        ('{"v": 1, "payload": {}}', "malformed"),
        ('{"v": 2, "type": "hello", "payload": {}}', "unsupported_version"),
        ('{"v": 1, "type": "action", "payload": {"tick": -1, "action": 0}}', "malformed"),
        ('{"v": 1, "type": "action", "payload": {"tick": 0, "action": 1, "x": 1}}', "malformed"),
        ('{"v": 1, "type": "hello", "payload": {"protocol": 1, "token": "t"}}', "malformed"),
    ],
)
def test_bad_frames_raise_protocol_error_with_a_code(text, code):
    with pytest.raises(ProtocolError) as err:
        decode_client(text)
    assert err.value.code == code


def test_a_server_message_may_grow_a_field_but_a_client_message_may_not():
    grown = (
        '{"v": 1, "type": "chronicle", "payload": '
        '{"tick": 3, "text": "a fly hatches", "weather": "fine"}}'
    )
    message = decode_server(grown)
    assert message.payload.text == "a fly hatches"  # the unknown field is simply ignored
    with pytest.raises(ProtocolError) as err:
        decode_client('{"v": 1, "type": "say", "payload": {"text": "hi", "loudly": true}}')
    assert err.value.code == "malformed"


def test_the_schema_says_which_side_may_grow_by_addition():
    defs = export_schema()["$defs"]
    for name in SERVER_DEFS:
        assert defs[name].get("additionalProperties") is not False, name
    for name in CLIENT_DEFS:
        assert defs[name]["additionalProperties"] is False, name


def test_text_from_a_client_carries_no_control_characters():
    with pytest.raises(ValueError):
        Say(text="\x1b[2Jcleared")
    with pytest.raises(ValueError):
        Say(text="two\nlines")
    with pytest.raises(ValueError):
        Hello(protocol=1, token="dev", owner="alice", role="agent", client="rogue\x07")
    assert Say(text="fruit? 🍎").text == "fruit? 🍎"


def test_owner_names_are_restricted():
    with pytest.raises(ValueError):
        Hello(protocol=1, token="dev", owner="", role="agent")
    with pytest.raises(ValueError):
        Hello(protocol=1, token="dev", owner="x" * 65, role="agent")
    assert Hello(protocol=1, token="dev", owner="Fly_0.1-x", role="spectator").owner == "Fly_0.1-x"


def test_catalog_matches_the_engine_observation_spec():
    fly = build_catalog().bodies["fly"]
    spec = observation_spec("fly")
    assert set(fly.channels) == set(spec)
    for name, channel in fly.channels.items():
        assert channel.shape == list(spec[name].shape)
        assert channel.dtype == spec[name].dtype
        assert (channel.low, channel.high) == (spec[name].low, spec[name].high)
    assert fly.actions == ["idle", "move_n", "move_e", "move_s", "move_w", "consume", "rest"]
    assert fly.channels["smell"].axes == {
        "rows": ["fruit", "humidity", "nest"],
        "cols": ["own", "n", "e", "s", "w"],
    }
    vision = fly.channels["vision"]
    assert vision.tables["terrain"]["nest"] == 5 and vision.tables["occupant"]["self"] == 1
    assert vision.centre == [3, 3] and vision.north_up is True
    assert build_catalog().constants["night_light_threshold"] == 500


def test_the_catalog_lists_every_event_and_the_keys_a_fuzz_run_emits():
    events = build_catalog().events
    assert events["moved"] == ["from", "to", "direction"]
    assert events["drank"] == []
    seen = set()

    def check(result):
        for event in result.events:
            assert event.type in events, event.type
            assert set(event.data) <= set(events[event.type]), (event.type, event.data)
            seen.add(event.type)

    world = World.from_map(maps.load("drosoville"), seed=5)
    flies = [world.spawn(), world.spawn()]
    rng = SplitMix64(23)
    for _ in range(1500):
        check(world.step({fly: rng.randbelow(9) for fly in flies}))  # invalid ids included

    larder = World.from_map("#####\n#F.~#\n#####")  # a meal and a drink a random walk misses
    fly = larder.spawn()
    for action in (4, 5, 2, 2, 5):  # west onto the fruit, eat, back east twice, drink
        check(larder.step({fly: action}))
    assert {"moved", "bumped", "ate", "drank", "died", "damaged"} <= seen


def test_schema_is_a_2020_12_document_with_both_unions():
    schema = export_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    defs = schema["$defs"]
    assert {"ClientMessage", "ServerMessage", "HelloMessage", "FrameMessage"} <= set(defs)
    assert defs["ClientMessage"]["discriminator"]["propertyName"] == "type"
    assert len(defs["ClientMessage"]["oneOf"]) == 6
    assert len(defs["ServerMessage"]["oneOf"]) == 9
    assert json.loads(schema_text()) == schema


def test_committed_schema_matches_the_models():
    """Regenerate with: uv run neurogarden schema > protocol/v1/neurogarden.schema.json"""
    assert SCHEMA_FILE.read_text(encoding="utf-8") == schema_text()
