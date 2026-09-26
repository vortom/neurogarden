"""A message queue where the newest observation or frame replaces an undelivered older one."""

from __future__ import annotations

import asyncio
from collections import deque

LATEST_WINS = frozenset({"observation", "frame"})


class Mailbox:
    """FIFO for ordered messages; latest-wins for the types a slow reader should skip."""

    def __init__(self) -> None:
        self._queue: deque = deque()
        self._wakeup = asyncio.Event()
        self._closed: BaseException | None = None

    def put(self, message) -> None:
        if message.type in LATEST_WINS:
            for index, queued in enumerate(self._queue):
                if queued.type == message.type:
                    del self._queue[index]
                    break
        self._queue.append(message)
        self._wakeup.set()

    def close(self, error: BaseException | None = None) -> None:
        """Wake readers; get() raises `error` (or returns None) once the queue is drained."""
        self._closed = error if error is not None else _Closed()
        self._wakeup.set()

    async def get(self):
        while not self._queue:
            if self._closed is not None:
                if isinstance(self._closed, _Closed):
                    return None
                raise self._closed
            self._wakeup.clear()
            await self._wakeup.wait()
        return self._queue.popleft()

    def __len__(self) -> int:
        return len(self._queue)


class _Closed(Exception):
    pass
