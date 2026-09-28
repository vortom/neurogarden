"""Ports: how the runner reaches a brain or a spectator without knowing what it is.

A remote port wraps a WebSocket; a local port wraps a Brain living in the server process.
Hosted "NPC" brains are ordinary clients that happen to run in-process (architecture §2.2).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Protocol

from websockets.exceptions import ConnectionClosed

from neurogarden.brains.base import THOUGHT_EVERY, Brain, brain_seed, think
from neurogarden.protocol.codec import decode_channels
from neurogarden.protocol.mailbox import Mailbox
from neurogarden.protocol.messages import BodyInfo, encode

log = logging.getLogger("neurogarden.server")
CLOSE_REASON_MAX = 100  # a WebSocket close reason must fit in 123 bytes


class Port(Protocol):
    owner: str
    role: str
    closing: bool

    def deliver(self, message: Any) -> None: ...  # never blocks

    def close(self, code: int, reason: str) -> None: ...


class RemotePort:
    """A connected client, agent or spectator; messages leave through a latest-wins mailbox."""

    def __init__(self, connection, owner: str, role: str) -> None:
        self.connection = connection
        self.owner = owner
        self.role = role
        self.mailbox = Mailbox()
        self.closing = False
        self._closer: asyncio.Task | None = None

    def deliver(self, message) -> None:
        if not self.closing:
            self.mailbox.put(message)

    def close(self, code: int, reason: str) -> None:
        if self.closing:
            return
        self.closing = True
        self.mailbox.close()
        self._closer = asyncio.get_running_loop().create_task(
            self.connection.close(code, reason[:CLOSE_REASON_MAX])
        )

    async def shutdown(self) -> None:
        """Let a close started by `close()` finish, so no task is left pending."""
        if self._closer is not None:
            closer, self._closer = self._closer, None
            with contextlib.suppress(ConnectionClosed):
                await closer

    async def pump(self) -> None:
        """Send queued messages until the mailbox closes or the socket goes away."""
        try:
            while (message := await self.mailbox.get()) is not None:
                await self.connection.send(encode(message))
        except ConnectionClosed:
            return  # the reader notices too; nothing to report from here


class LocalPort:
    """A brain in the server process. Acts the moment an observation is delivered."""

    role = "agent"
    closing = False

    def __init__(
        self, brain: Brain, owner: str, runner, body: BodyInfo, rejoin: bool = True
    ) -> None:
        self.brain = brain
        self.owner = owner
        self.runner = runner
        self.body = body
        self.rejoin = rejoin
        self.lives = 0

    def deliver(self, message) -> None:
        if message.type == "joined":
            self.lives += 1
            self.brain.reset(brain_seed(message.payload.lineage))
        elif message.type == "observation":
            channels = decode_channels(message.payload.channels, self.body)
            try:
                action = int(self.brain.act(channels))
            except Exception:  # a broken hosted brain idles; the world keeps turning
                log.exception("hosted brain of %s failed to act; idling this tick", self.owner)
                return
            self.runner.submit_action(self, message.payload.tick, action)
            if message.payload.tick % THOUGHT_EVERY == 0:
                thought = think(self.brain)
                if thought is not None:
                    self.runner.say(self, thought)
        elif message.type == "died" and self.rejoin:
            self.runner.request_join(self)

    def close(self, code: int, reason: str) -> None:
        log.info("local port %s closed: %s %s", self.owner, code, reason)
