"""Protocol v1 messages: pydantic models are the source of truth, the JSON Schema is derived.

Every frame on the wire is an envelope ``{"v": 1, "type": "<name>", "payload": {...}}``.
Payload models carry no ``type``; the envelope classes below bind each payload to its name
so the two directions form discriminated unions.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

log = logging.getLogger("neurogarden.protocol")
PROTOCOL_VERSION = 1
OWNER_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"
TEXT_PATTERN = r"^[^\x00-\x1f\x7f]*$"  # no control characters, no escape sequences
SAY_MAX = 40
REPLAY_SPEED, REPLAY_SPEED_MIN, REPLAY_SPEED_MAX = 4.0, 0.25, 64.0
LINEAGE_MAX = 2**31 - 1  # more lives than any owner will have; fits every integer column


class Model(BaseModel):
    """Client → server: strict. The server never guesses what a client meant."""

    model_config = ConfigDict(extra="forbid")


class OpenModel(BaseModel):
    """Server → client: a newer server may add fields, and older clients ignore them."""

    model_config = ConfigDict(extra="ignore")


# --- catalog: what every number in an observation means ---------------------------------


class ChannelInfo(OpenModel):
    shape: list[int]
    dtype: str
    low: int
    high: int
    axes: dict[str, list[str]] = Field(default_factory=dict)
    tables: dict[str, dict[str, int]] = Field(default_factory=dict)
    centre: list[int] | None = None
    north_up: bool | None = None


class BodyInfo(OpenModel):
    frame: str
    actions: list[str]
    channels: dict[str, ChannelInfo]


class Catalog(OpenModel):
    bodies: dict[str, BodyInfo]
    constants: dict[str, int]
    events: dict[str, list[str]] = Field(default_factory=dict)  # event type -> its data keys


# --- payloads: client -> server ----------------------------------------------------------


class Hello(Model):
    protocol: int
    token: str = Field(min_length=1, max_length=256)
    owner: str = Field(pattern=OWNER_PATTERN)
    role: Literal["agent", "spectator"]
    client: str = Field(default="", max_length=128, pattern=TEXT_PATTERN)


class Join(Model):
    body: Literal["fly"] = "fly"


class Action(Model):
    tick: int = Field(ge=0)
    action: int = Field(ge=0)


class Leave(Model):
    pass


class Say(Model):
    """A brain's thought bubble: plain text spectators can see over the fly."""

    text: str = Field(max_length=SAY_MAX, pattern=TEXT_PATTERN)


class Replay(Model):
    """A spectator asks to watch an archived life again: a ghost, at `speed` × the world's pace."""

    owner: str = Field(pattern=OWNER_PATTERN)
    lineage: int = Field(ge=1, le=LINEAGE_MAX)
    speed: float = Field(default=REPLAY_SPEED, ge=REPLAY_SPEED_MIN, le=REPLAY_SPEED_MAX)


# --- payloads: server -> client ----------------------------------------------------------


class WorldInfo(OpenModel):
    name: str
    width: int
    height: int
    tps: float
    # The shape of a day, in ticks since it began: dawn ramp [0, dawn_end), day,
    # dusk ramp [dusk_start, night_start), night. A client names the time of day from these.
    day_length: int
    dawn_end: int
    dusk_start: int
    night_start: int
    rules_version: int


class Welcome(OpenModel):
    protocol: int
    owner: str
    role: Literal["agent", "spectator"]
    world: WorldInfo
    catalog: Catalog
    motd: str = ""


class Joined(OpenModel):
    agent_id: int
    lineage: int
    name: str
    tick: int
    reattached: bool


class Channels(OpenModel):
    smell: list[list[int]]
    vision: list[list[list[int]]]
    touch: list[int]
    body: list[int]
    env: list[int]


class AgentEvent(OpenModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class Observation(OpenModel):
    tick: int
    deadline_ms: int
    channels: Channels
    events: list[AgentEvent]
    missed: int


class Stats(OpenModel):
    lifespan: int
    days: int
    death_causes: list[str]
    bites: int
    drinks: int
    rest_ticks: int
    bumps: int
    tiles_explored: int
    mean_wellbeing: float


class Died(OpenModel):
    agent_id: int
    lineage: int
    name: str
    tick: int
    causes: list[str]
    stats: Stats


class WorldMap(OpenModel):
    map_name: str
    width: int
    height: int
    terrain: list[list[int]]


class AgentView(OpenModel):
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


class WorldEvent(OpenModel):
    type: str
    agent_id: int | None
    data: dict[str, Any] = Field(default_factory=dict)


class OwnerScore(OpenModel):
    owner: str
    lives: int
    best_lifespan: int
    alive: bool
    best_lineage: int = 0  # the life that set best_lifespan; 0 while none has ended


class ResourceView(OpenModel):
    """One thing worth eating or drinking, where a spectator can see it."""

    x: int
    y: int
    kind: int
    amount: int


class Frame(OpenModel):
    tick: int
    day: int
    light: int
    resources: list[ResourceView]
    agents: list[AgentView]
    events: list[WorldEvent]
    scores: list[OwnerScore]


class Chronicle(OpenModel):
    """One line of the naturalist's log, written by the server from what happened."""

    tick: int
    text: str


class Error(OpenModel):
    code: str
    message: str
    fatal: bool


class ReplayInfo(OpenModel):
    """Brackets a ghost: sent before its first frame (`done` false) and after its last."""

    owner: str
    lineage: int
    name: str
    born_tick: int
    died_tick: int | None  # None: the life was still going when the archive was read
    lifespan: int | None
    causes: list[str]
    speed: float
    done: bool


# --- envelopes ---------------------------------------------------------------------------


class _ClientEnvelope(Model):
    v: Literal[1] = PROTOCOL_VERSION


class _ServerEnvelope(OpenModel):
    v: Literal[1] = PROTOCOL_VERSION


class HelloMessage(_ClientEnvelope):
    type: Literal["hello"] = "hello"
    payload: Hello


class JoinMessage(_ClientEnvelope):
    type: Literal["join"] = "join"
    payload: Join = Field(default_factory=Join)


class ActionMessage(_ClientEnvelope):
    type: Literal["action"] = "action"
    payload: Action


class LeaveMessage(_ClientEnvelope):
    type: Literal["leave"] = "leave"
    payload: Leave = Field(default_factory=Leave)


class SayMessage(_ClientEnvelope):
    type: Literal["say"] = "say"
    payload: Say


class ReplayMessage(_ClientEnvelope):
    type: Literal["replay"] = "replay"
    payload: Replay


class WelcomeMessage(_ServerEnvelope):
    type: Literal["welcome"] = "welcome"
    payload: Welcome


class JoinedMessage(_ServerEnvelope):
    type: Literal["joined"] = "joined"
    payload: Joined


class ObservationMessage(_ServerEnvelope):
    type: Literal["observation"] = "observation"
    payload: Observation


class DiedMessage(_ServerEnvelope):
    type: Literal["died"] = "died"
    payload: Died


class WorldMessage(_ServerEnvelope):
    type: Literal["world"] = "world"
    payload: WorldMap


class FrameMessage(_ServerEnvelope):
    type: Literal["frame"] = "frame"
    payload: Frame


class ChronicleMessage(_ServerEnvelope):
    type: Literal["chronicle"] = "chronicle"
    payload: Chronicle


class ErrorMessage(_ServerEnvelope):
    type: Literal["error"] = "error"
    payload: Error


class ReplayInfoMessage(_ServerEnvelope):
    type: Literal["replay"] = "replay"
    payload: ReplayInfo


ClientMessage = Annotated[
    HelloMessage | JoinMessage | ActionMessage | LeaveMessage | SayMessage | ReplayMessage,
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
    | ErrorMessage
    | ReplayInfoMessage,
    Field(discriminator="type"),
]

CLIENT_TYPES = frozenset({"hello", "join", "action", "leave", "say", "replay"})
SERVER_TYPES = frozenset(
    {"welcome", "joined", "observation", "died", "world", "frame", "chronicle", "error", "replay"}
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
        log.debug("ignoring a message of unknown type %r", raw["type"])
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
