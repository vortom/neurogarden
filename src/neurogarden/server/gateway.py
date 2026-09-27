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

from .frames import error_message
from .ports import CLOSE_REASON_MAX, RemotePort
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

    async def _spectator_loop(self, connection, port: RemotePort) -> None:
        async for text in connection:
            if port.closing:
                return
            try:
                message = decode_client(text)
            except ProtocolError as err:
                await self._fail(connection, err.code, str(err))
                return
            if message is not None and message.type != "hello":
                port.deliver(error_message("spectator", f"spectators cannot {message.type}"))
