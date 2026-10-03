"""Session: the blocking SDK for ordinary brains. A background thread runs the asyncio client."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

from neurogarden.brains.base import Brain, Mouth, brain_seed
from neurogarden.dojo.stats import EpisodeStats
from neurogarden.protocol.messages import Died, Joined, Welcome

from .client import DEFAULT_URL, AsyncClient, Observation


class Session:
    """`with Session(url, owner="alice") as session:` — connect, join, act, leave."""

    def __init__(
        self, url: str = DEFAULT_URL, *, owner: str, token: str = "dev", client: str | None = None
    ) -> None:
        kwargs = {"owner": owner, "token": token}
        if client is not None:
            kwargs["client"] = client
        self._client = AsyncClient(url, **kwargs)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def __enter__(self) -> Session:
        self._thread.start()
        try:
            self._run(self._client.__aenter__())
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._loop.is_closed():
            return
        if not self._thread.is_alive():  # never entered, or already unwound: nothing to wait for
            self._loop.close()
            return
        try:
            self._run(self._client.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()

    @property
    def welcome(self) -> Welcome:
        assert self._client.welcome is not None
        return self._client.welcome

    def join(self, body: str = "fly") -> Fly:
        joined = self._run(self._client.join(body))
        return Fly(self, joined)

    def leave(self) -> None:
        self._run(self._client.leave())


class Fly:
    """One life. Iterate observations, answer with actions; `stats` is set after death."""

    def __init__(self, session: Session, joined: Joined) -> None:
        self._session = session
        self.joined = joined
        self.died: Died | None = None

    @property
    def name(self) -> str:
        return self.joined.name

    @property
    def lineage(self) -> int:
        return self.joined.lineage

    @property
    def stats(self) -> EpisodeStats | None:
        return None if self.died is None else stats_of(self.died)

    def observations(self) -> Iterator[Observation]:
        client = self._session._client
        while True:
            observation = self._session._run(client.next_observation())
            if observation is None:
                self.died = client.last_died
                return
            yield observation

    def act(self, tick: int, action: int) -> None:
        self._session._run(self._session._client.act(tick, action))

    def say(self, text: str) -> None:
        self._session._run(self._session._client.say(text))


def stats_of(died) -> EpisodeStats:
    """A `died` message's stats as the dojo's EpisodeStats: one type for both worlds."""
    fields = died.stats.model_dump()
    fields["death_causes"] = tuple(fields["death_causes"])
    return EpisodeStats(**fields)


def run_brain(
    url: str,
    brain: Brain,
    *,
    owner: str,
    token: str = "dev",
    lives: int | None = None,
    on_life=None,
) -> list[EpisodeStats]:
    """Join, feed the brain every observation, rejoin after death: `lives` times, or forever.

    `on_life(fly)` is called after each death with the finished `Fly` (stats attached).
    Returns the stats of every completed life; a lost connection raises ConnectionLost.
    """
    finished: list[EpisodeStats] = []
    mouth = Mouth(brain)  # a brain with thoughts shows them, like a hosted one
    with Session(url, owner=owner, token=token) as session:
        while lives is None or len(finished) < lives:
            fly = session.join()
            brain.reset(brain_seed(fly.lineage))
            for observation in fly.observations():
                fly.act(observation.tick, int(brain.act(observation.channels)))
                if (thought := mouth.speak(observation.tick)) is not None:
                    fly.say(thought)
            finished.append(fly.stats)
            if on_life is not None:
                on_life(fly)
    return finished
