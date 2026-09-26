"""The WebSocket edge: handshake, validation, dispatch to the runner, close codes."""

from __future__ import annotations

import asyncio
import hmac
import logging
from http import HTTPStatus

from websockets.exceptions import ConnectionClosed

from neurogarden.protocol.messages import (
    PROTOCOL_VERSION,
    Error,
    ErrorMessage,
    ProtocolError,
    decode_client,
    encode,
)

from .ports import RemotePort
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


def health_check(connection, request):
    """Plain HTTP on the WebSocket port, so a load balancer can ask if we are alive."""
    if request.path == "/healthz":
        return connection.respond(HTTPStatus.OK, "OK\n")
    return None


class Gateway:
    def __init__(self, runner: WorldRunner, token: str, hello_timeout: float = 5.0) -> None:
        self.runner = runner
        self.token = token
        self.hello_timeout = hello_timeout

    async def _fail(self, connection, code: str, message: str) -> None:
        payload = Error(code=code, message=message, fatal=True)
        try:
            await connection.send(encode(ErrorMessage(payload=payload)))
        except ConnectionClosed:
            return
        await connection.close(_CLOSE_CODES[code], message)

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
        port = RemotePort(connection, hello.owner, hello.role)
        await connection.send(encode(self.runner.welcome_for(hello.owner, hello.role)))
        log.info("%s connected as %s (%s)", hello.owner, hello.role, hello.client or "?")
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
            log.info("%s disconnected", port.owner)

    async def _agent_loop(self, connection, port: RemotePort) -> None:
        async for text in connection:
            try:
                message = decode_client(text)
            except ProtocolError as err:
                await self._fail(connection, err.code, str(err))
                return
            if message is None:
                continue
            if message.type == "join":
                self.runner.attach(port)  # a join after leave re-attaches the same connection
                self.runner.request_join(port, message.payload.body)
            elif message.type == "action":
                self.runner.submit_action(port, message.payload.tick, message.payload.action)
            elif message.type == "say":
                self.runner.say(port, message.payload.text)
            elif message.type == "leave":
                self.runner.detach(port)
            elif message.type == "hello":
                port.deliver(_soft_error("already_connected", "hello was already sent"))

    async def _spectator_loop(self, connection, port: RemotePort) -> None:
        async for text in connection:
            try:
                message = decode_client(text)
            except ProtocolError as err:
                await self._fail(connection, err.code, str(err))
                return
            if message is not None and message.type != "hello":
                port.deliver(_soft_error("spectator", f"spectators cannot {message.type}"))


def _soft_error(code: str, message: str) -> ErrorMessage:
    return ErrorMessage(payload=Error(code=code, message=message, fatal=False))
