"""AsyncClient: the asyncio side of the SDK. Everything the sync facade builds on."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from neurogarden import __version__
from neurogarden.protocol.codec import decode_channels
from neurogarden.protocol.mailbox import Mailbox
from neurogarden.protocol.messages import (
    PROTOCOL_VERSION,
    Action,
    ActionMessage,
    AgentEvent,
    BodyInfo,
    Died,
    Frame,
    Hello,
    HelloMessage,
    Join,
    Joined,
    JoinMessage,
    LeaveMessage,
    ProtocolError,
    Say,
    SayMessage,
    Welcome,
    WorldMap,
    decode_server,
    encode,
)

DEFAULT_URL = "ws://127.0.0.1:8765"


class ConnectionLost(ConnectionError):
    def __init__(self, code: int | None, reason: str) -> None:
        super().__init__(f"connection closed ({code}): {reason}")
        self.code = code
        self.reason = reason


class ServerError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class Observation:
    tick: int
    deadline_ms: int
    channels: dict[str, np.ndarray]
    events: list[AgentEvent]
    missed: int


@dataclass
class Spectacle:
    """What a spectator sees: the static map, the latest frame, the chronicle so far."""

    world: WorldMap | None = None
    frame: Frame | None = None
    chronicle: list[tuple[int, str]] = field(default_factory=list)


class AsyncClient:
    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        owner: str,
        token: str = "dev",
        role: str = "agent",
        client: str = f"neurogarden-sdk/{__version__}",
    ) -> None:
        self.url = url
        self.owner = owner
        self.token = token
        self.role = role
        self.client = client
        self.welcome: Welcome | None = None
        self.joined: Joined | None = None
        self.last_died: Died | None = None
        self.spectacle = Spectacle()
        self._connection = None
        self._reader: asyncio.Task | None = None
        self._inbox = Mailbox()
        self._joined_waiters: list[asyncio.Future] = []

    # --- lifecycle ---------------------------------------------------------------------

    async def __aenter__(self) -> AsyncClient:
        try:
            self._connection = await connect(self.url)
        except OSError as err:  # refused, unreachable, DNS
            raise ConnectionLost(None, f"cannot reach {self.url}: {err}") from None
        hello = Hello(
            protocol=PROTOCOL_VERSION,
            token=self.token,
            owner=self.owner,
            role=self.role,
            client=self.client,
        )
        await self._connection.send(encode(HelloMessage(payload=hello)))
        first = await self._recv_one()
        if first is None or first.type != "welcome":
            got = getattr(first, "type", None)
            raise ProtocolError("malformed", f"expected welcome, got {got}")
        self.welcome = first.payload
        self._reader = asyncio.get_running_loop().create_task(self._read_loop())
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @property
    def body(self) -> BodyInfo:
        assert self.welcome is not None, "not connected"
        return self.welcome.catalog.bodies["fly"]

    async def _recv_one(self):
        try:
            text = await self._connection.recv()
        except ConnectionClosed as closed:
            raise ConnectionLost(*_close_info(closed)) from None
        message = decode_server(text)
        if message is not None and message.type == "error" and message.payload.fatal:
            raise ServerError(message.payload.code, message.payload.message)
        return message

    async def _read_loop(self) -> None:
        try:
            while True:
                message = await self._recv_one()
                if message is None:
                    continue
                if message.type == "joined":
                    self.joined = message.payload
                    for waiter in self._joined_waiters:
                        if not waiter.done():
                            waiter.set_result(message.payload)
                    self._joined_waiters.clear()
                elif message.type == "died":
                    self.last_died = message.payload
                    self.joined = None
                    self._inbox.put(message)
                elif message.type == "error":
                    continue  # non-fatal notices are informational
                elif message.type == "world":
                    self.spectacle.world = message.payload
                elif message.type == "chronicle":
                    self.spectacle.chronicle.append((message.payload.tick, message.payload.text))
                    self._inbox.put(message)
                else:
                    self._inbox.put(message)
        except (ConnectionLost, ServerError, ProtocolError) as err:
            self._inbox.close(err)
            for waiter in self._joined_waiters:
                if not waiter.done():
                    waiter.set_exception(err)
            self._joined_waiters.clear()

    # --- agent API ---------------------------------------------------------------------

    async def join(self, body: str = "fly") -> Joined:
        waiter = asyncio.get_running_loop().create_future()
        self._joined_waiters.append(waiter)
        await self._send(JoinMessage(payload=Join(body=body)))
        return await waiter

    async def next_observation(self) -> Observation | None:
        """The newest observation, or None once the fly has died."""
        while True:
            message = await self._inbox.get()
            if message is None:
                raise ConnectionLost(None, "client closed")
            if message.type == "observation":
                payload = message.payload
                return Observation(
                    tick=payload.tick,
                    deadline_ms=payload.deadline_ms,
                    channels=decode_channels(payload.channels, self.body),
                    events=payload.events,
                    missed=payload.missed,
                )
            if message.type == "died":
                return None

    async def observations(self):
        while (observation := await self.next_observation()) is not None:
            yield observation

    async def act(self, tick: int, action: int) -> None:
        await self._send(ActionMessage(payload=Action(tick=tick, action=int(action))))

    async def say(self, text: str) -> None:
        await self._send(SayMessage(payload=Say(text=text)))

    async def leave(self) -> None:
        await self._send(LeaveMessage())

    # --- spectator API -----------------------------------------------------------------

    async def next_frame(self) -> Frame | None:
        """The newest frame; chronicle lines are collected on `spectacle` meanwhile."""
        while True:
            message = await self._inbox.get()
            if message is None:
                return None
            if message.type == "frame":
                self.spectacle.frame = message.payload
                return message.payload

    async def frames(self):
        while (frame := await self.next_frame()) is not None:
            yield frame

    async def _send(self, message) -> None:
        try:
            await self._connection.send(encode(message))
        except ConnectionClosed as closed:
            raise ConnectionLost(*_close_info(closed)) from None


def _close_info(closed: ConnectionClosed) -> tuple[int | None, str]:
    close = closed.rcvd or closed.sent
    return (close.code, close.reason) if close is not None else (None, "connection lost")
