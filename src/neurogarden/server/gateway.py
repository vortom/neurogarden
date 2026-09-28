"""The WebSocket edge: handshake, validation, dispatch to the runner, close codes."""

from __future__ import annotations

import asyncio
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
_REBUILD_STRETCH = 50  # engine steps a ghost replays before yielding to the live world
REPLAY_INTERVAL = 1.0  # seconds a spectator waits between one ghost and the next
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
        self._ghosts: dict[RemotePort, asyncio.Task] = {}  # the replay a spectator is watching
        self._last_replay: dict[RemotePort, float] = {}  # when each one last asked for a ghost

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
                self._last_replay.pop(port, None)
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
        # Starting a ghost rebuilds a world (up to snapshot_every engine steps): one a second
        # per spectator is plenty for a person and too few for a client out to stall the world.
        now = self.runner.clock()
        if now - self._last_replay.get(port, -REPLAY_INTERVAL) < REPLAY_INTERVAL:
            port.deliver(error_message("replay_busy", "one replay a second, please"))
            return
        self._last_replay[port] = now
        await self._stop_ghost(port)
        life = self.runner.archive.life(request.owner, request.lineage)
        if life is None:
            port.deliver(
                error_message("no_such_life", f"{request.owner} has no life #{request.lineage}")
            )
            self._back_to_the_living(port)
            return
        self.runner.remove_spectator(port)  # a ghost watcher gets no live frames meanwhile
        self._ghosts[port] = asyncio.get_running_loop().create_task(
            self._haunt(port, life, request.speed)
        )

    async def _stop_ghost(self, port: RemotePort) -> None:
        task = self._ghosts.pop(port, None)
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise  # that one was aimed at us (shutdown), not at the ghost

    def _back_to_the_living(self, port: RemotePort) -> None:
        """A spectator whose ghost is over (or never was) watches the live world again."""
        if not port.closing and port not in self.runner.spectators:
            self.runner.add_spectator(port)

    async def _world_before(self, tick: int):
        """`history.world_at`, letting the live world tick in between stretches of replay."""
        steps = history.rebuild_steps(self.runner.archive, tick)
        rebuilt = next(steps)
        for count, _ in enumerate(steps):
            if count % _REBUILD_STRETCH == 0:
                await asyncio.sleep(0)
        return rebuilt.world

    async def _haunt(self, port: RemotePort, life, speed: float) -> None:
        """Stream one archived life as frames, at `speed` × the world's own pace."""
        runner = self.runner
        archive = runner.archive
        period = 1.0 / (runner.tps * speed)
        end = archive.life_end(life)  # fixed now: a life still going is not chased
        try:
            port.deliver(frames.world_message(runner.world, runner.map_name))
            port.deliver(frames.replay_message(life, speed, done=False))
            roster = GhostRoster(archive.lives())
            said: dict[int, list[str]] = {}  # the naturalist's lines of those days, by tick
            for tick, text in archive.chronicle_between(life.born_tick, end):
                said.setdefault(tick, []).append(text)
            world = await self._world_before(life.born_tick)
            for moment in history.playback(archive, life.born_tick, end, world):
                # A reader slower than the ghost would only have this frame replaced by the
                # next (latest wins), so it is not even built — except the last one, which
                # every watcher must see: the fly as it ended.
                if moment.tick == end - 1 or not port.mailbox.holds("frame"):
                    events = moment.result.events
                    frame = frames.frame_message(moment.world, roster, events, moment.tick, {})
                    port.deliver(frame)
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
            if self._ghosts.get(port) is asyncio.current_task():
                del self._ghosts[port]
        self._back_to_the_living(port)  # not reached when cancelled: a new ghost, or goodbye
