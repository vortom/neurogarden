"""Protocol v1 messages: pydantic models are the source of truth, the JSON Schema is derived.

Every frame on the wire is an envelope ``{"v": 1, "type": "<name>", "payload": {...}}``.
Payload models carry no ``type``; the envelope classes below bind each payload to its name
so the two directions form discriminated unions.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

PROTOCOL_VERSION = 1
OWNER_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"
SAY_MAX = 40


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- catalog: what every number in an observation means ---------------------------------


class ChannelInfo(Model):
    shape: list[int]
    dtype: str
    low: int
    high: int
    axes: dict[str, list[str]] = Field(default_factory=dict)
    tables: dict[str, dict[str, int]] = Field(default_factory=dict)
    centre: list[int] | None = None
    north_up: bool | None = None


class BodyInfo(Model):
    frame: str
    actions: list[str]
    channels: dict[str, ChannelInfo]


class Catalog(Model):
    bodies: dict[str, BodyInfo]
    constants: dict[str, int]


# --- payloads: client -> server ----------------------------------------------------------


class Hello(Model):
    protocol: int
    token: str = Field(min_length=1, max_length=256)
    owner: str = Field(pattern=OWNER_PATTERN)
    role: Literal["agent", "spectator"]
    client: str = Field(default="", max_length=128)


class Join(Model):
    body: Literal["fly"] = "fly"


class Action(Model):
    tick: int = Field(ge=0)
    action: int = Field(ge=0)


class Leave(Model):
    pass


class Say(Model):
    """A brain's thought bubble: free text spectators can see over the fly."""

    text: str = Field(max_length=SAY_MAX)


# --- payloads: server -> client ----------------------------------------------------------


class WorldInfo(Model):
    name: str
    width: int
    height: int
    tps: float
    day_length: int
    rules_version: int


class Welcome(Model):
    protocol: int
    owner: str
    role: Literal["agent", "spectator"]
    world: WorldInfo
    catalog: Catalog
    motd: str = ""


class Joined(Model):
    agent_id: int
    lineage: int
    name: str
    tick: int
    reattached: bool


class Channels(Model):
    smell: list[list[int]]
    vision: list[list[list[int]]]
    touch: list[int]
    body: list[int]
    env: list[int]


class AgentEvent(Model):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class Observation(Model):
    tick: int
    deadline_ms: int
    channels: Channels
    events: list[AgentEvent]
    missed: int


class Stats(Model):
    lifespan: int
    days: int
    death_causes: list[str]
    bites: int
    drinks: int
    rest_ticks: int
    bumps: int
    tiles_explored: int
    mean_wellbeing: float


class Died(Model):
    agent_id: int
    lineage: int
    name: str
    tick: int
    causes: list[str]
    stats: Stats


class WorldMap(Model):
    map_name: str
    width: int
    height: int
    terrain: list[list[int]]


class AgentView(Model):
    agent_id: int
    owner: str
    lineage: int
    name: str
    x: int
    y: int
    facing: int
    satiety: int
    hydration: int
    energy: int
    health: int
    age: int
    alive: bool
    connected: bool
    mood: str
    say: str = ""


class WorldEvent(Model):
    type: str
    agent_id: int | None
    data: dict[str, Any] = Field(default_factory=dict)


class OwnerScore(Model):
    owner: str
    lives: int
    best_lifespan: int
    alive: bool


class Frame(Model):
    tick: int
    day: int
    light: int
    resources: list[list[int]]
    agents: list[AgentView]
    events: list[WorldEvent]
    scores: list[OwnerScore]


class Chronicle(Model):
    """One line of the naturalist's log, written by the server from what happened."""

    tick: int
    text: str


class Error(Model):
    code: str
    message: str
    fatal: bool


# --- envelopes ---------------------------------------------------------------------------


class _Envelope(Model):
    v: Literal[1] = PROTOCOL_VERSION


class HelloMessage(_Envelope):
    type: Literal["hello"] = "hello"
    payload: Hello


class JoinMessage(_Envelope):
    type: Literal["join"] = "join"
    payload: Join = Field(default_factory=Join)


class ActionMessage(_Envelope):
    type: Literal["action"] = "action"
    payload: Action


class LeaveMessage(_Envelope):
    type: Literal["leave"] = "leave"
    payload: Leave = Field(default_factory=Leave)


class SayMessage(_Envelope):
    type: Literal["say"] = "say"
    payload: Say


class WelcomeMessage(_Envelope):
    type: Literal["welcome"] = "welcome"
    payload: Welcome


class JoinedMessage(_Envelope):
    type: Literal["joined"] = "joined"
    payload: Joined


class ObservationMessage(_Envelope):
    type: Literal["observation"] = "observation"
    payload: Observation


class DiedMessage(_Envelope):
    type: Literal["died"] = "died"
    payload: Died


class WorldMessage(_Envelope):
    type: Literal["world"] = "world"
    payload: WorldMap


class FrameMessage(_Envelope):
    type: Literal["frame"] = "frame"
    payload: Frame


class ChronicleMessage(_Envelope):
    type: Literal["chronicle"] = "chronicle"
    payload: Chronicle


class ErrorMessage(_Envelope):
    type: Literal["error"] = "error"
    payload: Error


ClientMessage = Annotated[
    HelloMessage | JoinMessage | ActionMessage | LeaveMessage | SayMessage,
    Field(discriminator="type"),
]
ServerMessage = Annotated[
    WelcomeMessage
    | JoinedMessage
    | ObservationMessage
    | DiedMessage
    | WorldMessage
    | FrameMessage
    | ChronicleMessage
    | ErrorMessage,
    Field(discriminator="type"),
]

CLIENT_TYPES = frozenset({"hello", "join", "action", "leave", "say"})
SERVER_TYPES = frozenset(
    {"welcome", "joined", "observation", "died", "world", "frame", "chronicle", "error"}
)

client_adapter: TypeAdapter = TypeAdapter(ClientMessage)
server_adapter: TypeAdapter = TypeAdapter(ServerMessage)


class ProtocolError(ValueError):
    """The frame is not a valid message of this protocol version."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def encode(message: BaseModel) -> str:
    return message.model_dump_json()


def _decode(text: str | bytes, adapter: TypeAdapter, known: frozenset[str]):
    try:
        raw = json.loads(text)
    except (ValueError, TypeError) as err:
        raise ProtocolError("malformed", f"frame is not JSON: {err}") from None
    if not isinstance(raw, dict) or "type" not in raw:
        raise ProtocolError("malformed", "frame is not an envelope")
    if raw.get("v") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported_version", f"protocol version {raw.get('v')!r}")
    if not isinstance(raw["type"], str) or raw["type"] not in known:
        return None  # unknown types are ignored: the protocol grows by addition
    try:
        return adapter.validate_python(raw)
    except ValidationError as err:
        first = err.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise ProtocolError("malformed", f"{raw['type']}: {where}: {first['msg']}") from None


def decode_client(text: str | bytes):
    """A client → server message, or None for an unknown type. Raises ProtocolError."""
    return _decode(text, client_adapter, CLIENT_TYPES)


def decode_server(text: str | bytes):
    """A server → client message, or None for an unknown type. Raises ProtocolError."""
    return _decode(text, server_adapter, SERVER_TYPES)
