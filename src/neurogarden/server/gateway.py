"""The WebSocket edge: handshake, validation, dispatch to the runner, close codes."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging

from websockets.exceptions import ConnectionClosed

from neurogarden.protocol.messages import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode_client,
    encode,
)

from . import frames, history
from .archive import ArchiveError
from .frames import error_message
from .ports import CLOSE_REASON_MAX, RemotePort
from .roster import GhostRoster
from .runner import WorldRunner

log = logging.getLogger("neurogarden.server")

CLOSE_MALFORMED = 4000
CLOSE_UNSUPPORTED_VERSION = 4001
CLOSE_UNAUTHORIZED = 4002
CLOSE_HELLO_REQUIRED = 4003
CLOSE_SUPERSEDED = 4004
_CLOSE_CODES = {
    "malformed": CLOSE_MALFORMED,
    "unsupported_version": CLOSE_UNSUPPORTED_VERSION,
    "unauthorized": CLOSE_UNAUTHORIZED,
    "hello_required": CLOSE_HELLO_REQUIRED,
}


class Gateway:
    def __init__(self, runner: WorldRunner, token: str, hello_timeout: float = 5.0) -> None:
        self.runner = runner
        self.token = token
        self.hello_timeout = hello_timeout
        self._ghosts: dict[int, asyncio.Task] = {}  # id(port) -> the replay it is watching

    async def _fail(self, connection, code: str, message: str) -> None:
        """The full story goes in the error payload; the close reason is what fits in a frame."""
        try:
            await connection.send(encode(error_message(code, message, fatal=True)))
        except ConnectionClosed:
            return
        await connection.close(_CLOSE_CODES[code], message[:CLOSE_REASON_MAX])

    async def _handshake(self, connection) -> RemotePort | None:
        try:
            text = await asyncio.wait_for(connection.recv(), self.hello_timeout)
        except TimeoutError:
            await connection.close(CLOSE_HELLO_REQUIRED, "hello required")
            return None
        except ConnectionClosed:
            return None
        try:
            message = decode_client(text)
        except ProtocolError as err:
            await self._fail(connection, err.code, str(err))
            return None
        if message is None or message.type != "hello":
            await self._fail(connection, "hello_required", "the first message must be hello")
            return None
        hello = message.payload
        if hello.protocol != PROTOCOL_VERSION:
            await self._fail(connection, "unsupported_version", f"protocol {hello.protocol}")
            return None
        if not hmac.compare_digest(hello.token.encode(), self.token.encode()):
            await self._fail(connection, "unauthorized", "bad token")
            return None
        if self.runner.is_reserved(hello.owner):
            await self._fail(connection, "unauthorized", f"{hello.owner} is a hosted brain")
            return None
        port = RemotePort(connection, hello.owner, hello.role)
        try:
            await connection.send(encode(self.runner.welcome_for(hello.owner, hello.role)))
        except ConnectionClosed:
            return None
        log.info("%s connected as %s (%r)", hello.owner, hello.role, hello.client)
        return port

    async def handle(self, connection) -> None:
        port = await self._handshake(connection)
        if port is None:
            return
        pump = asyncio.get_running_loop().create_task(port.pump())
        try:
            if port.role == "agent":
                self.runner.attach(port)
                await self._agent_loop(connection, port)
            else:
                self.runner.add_spectator(port)
                await self._spectator_loop(connection, port)
        except ConnectionClosed:
            pass
        finally:
            if port.role == "agent":
                self.runner.detach(port)
            else:
                await self._stop_ghost(port)
                self.runner.remove_spectator(port)
            port.closing = True
            port.mailbox.close()
            pump.cancel()
            await port.shutdown()
            log.info("%s disconnected", port.owner)

    async def _agent_loop(self, connection, port: RemotePort) -> None:
        async for text in connection:
            if port.closing:
                return  # superseded: this connection no longer speaks for its owner
            try:
                message = decode_client(text)
            except ProtocolError as err:
                await self._fail(connection, err.code, str(err))
                return
            if message is None:
                continue
            if message.type == "join":
                self.runner.request_join(port, message.payload.body)
            elif message.type == "action":
                accepted = self.runner.submit_action(
                    port, message.payload.tick, message.payload.action
                )
                if not accepted and not self.runner.controls_a_fly(port):
                    port.deliver(error_message("no_fly", "join before acting"))
            elif message.type == "say":
                self.runner.say(port, message.payload.text)
            elif message.type == "leave":
                self.runner.detach(port)
            elif message.type == "hello":
                port.deliver(error_message("already_connected", "hello was already sent"))
            elif message.type == "replay":
                port.deliver(error_message("spectator", "only spectators watch replays"))

    async def _spectator_loop(self, connection, port: RemotePort) -> None:
        async for text in connection:
            if port.closing:
                return
            try:
                message = decode_client(text)
            except ProtocolError as err:
                await self._fail(connection, err.code, str(err))
                return
            if message is None:
                continue
            if message.type == "replay":
                await self._start_ghost(port, message.payload)
            elif message.type != "hello":
                port.deliver(error_message("spectator", f"spectators cannot {message.type}"))

    # --- ghosts: archived lives replayed to one spectator --------------------------------

    async def _start_ghost(self, port: RemotePort, request) -> None:
        await self._stop_ghost(port)
        life = self.runner.archive.life(request.owner, request.lineage)
        if life is None:
            port.deliver(
                error_message("no_such_life", f"{request.owner} has no life #{request.lineage}")
            )
            self._back_to_the_living(port)
            return
        self.runner.remove_spectator(port)  # a ghost watcher gets no live frames meanwhile
        self._ghosts[id(port)] = asyncio.get_running_loop().create_task(
            self._haunt(port, life, request.speed)
        )

    async def _stop_ghost(self, port: RemotePort) -> None:
        task = self._ghosts.pop(id(port), None)
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _back_to_the_living(self, port: RemotePort) -> None:
        """A spectator whose ghost is over (or never was) watches the live world again."""
        if not port.closing and port not in self.runner.spectators:
            self.runner.add_spectator(port)

    async def _haunt(self, port: RemotePort, life, speed: float) -> None:
        """Stream one archived life as frames, at `speed` × the world's own pace."""
        runner = self.runner
        period = 1.0 / (runner.tps * speed)
        last = None if life.died_tick is None else life.died_tick + 1  # through its last frame
        try:
            port.deliver(frames.world_message(runner.world, runner.map_name))
            port.deliver(frames.replay_message(life, speed, done=False))
            roster = GhostRoster(runner.archive.lives())
            said: dict[int, list[str]] = {}  # the naturalist's lines of those days, by tick
            until = runner.archive.next_tick() if last is None else last
            for tick, text in runner.archive.chronicle_between(life.born_tick, until):
                said.setdefault(tick, []).append(text)
            for moment in history.playback(runner.archive, life.born_tick, last):
                events = moment.result.events
                port.deliver(frames.frame_message(moment.world, roster, events, moment.tick, {}))
                for text in said.get(moment.tick, ()):
                    port.deliver(frames.chronicle_message(moment.tick, text))
                await asyncio.sleep(period)
            port.deliver(frames.replay_message(life, speed, done=True))
        except ArchiveError as err:
            log.warning("replay of %s #%d failed: %s", life.owner, life.lineage, err)
            port.deliver(error_message("replay_failed", str(err)))
        except Exception:  # a broken replay is this spectator's problem, not the world's
            log.exception("replay of %s #%d crashed", life.owner, life.lineage)
            port.deliver(error_message("replay_failed", "the replay could not be played"))
        finally:
            self._ghosts.pop(id(port), None)
        self._back_to_the_living(port)  # not reached when cancelled: a new ghost, or goodbye
