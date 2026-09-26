import json
from pathlib import Path

import pytest

from neurogarden.engine.body import observation_spec
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
    Say,
    SayMessage,
)

SCHEMA_FILE = Path(__file__).parent.parent / "protocol" / "v1" / "neurogarden.schema.json"


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


def test_schema_is_a_2020_12_document_with_both_unions():
    schema = export_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    defs = schema["$defs"]
    assert {"ClientMessage", "ServerMessage", "HelloMessage", "FrameMessage"} <= set(defs)
    assert defs["ClientMessage"]["discriminator"]["propertyName"] == "type"
    assert len(defs["ClientMessage"]["oneOf"]) == 5
    assert len(defs["ServerMessage"]["oneOf"]) == 8
    assert json.loads(schema_text()) == schema


def test_committed_schema_matches_the_models():
    """Regenerate with: uv run neurogarden schema > protocol/v1/neurogarden.schema.json"""
    assert SCHEMA_FILE.read_text(encoding="utf-8") == schema_text()
