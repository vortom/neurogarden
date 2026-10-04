"""A flock: several brains in one process, each its own owner in the same world.

Ten connectome flies share one wiring in memory and are stepped together; to the world
they are ten ordinary clients, each with its own lives, name and place in the hall of flies.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass

from neurogarden.brains.base import Brain, Mouth, brain_seed
from neurogarden.brains.connectome import ConnectomeBrain, ConnectomeFlock
from neurogarden.dojo.stats import EpisodeStats

from .client import DEFAULT_URL, AsyncClient
from .session import stats_of


@dataclass
class FlockLife:
    """One finished life of one of the flock's flies."""

    owner: str
    name: str
    lineage: int
    stats: EpisodeStats


def _life_seed(fly: int, lineage: int) -> int:
    """Every fly its own stream of chance, every life another: flies that share weights must
    not also share their draws, or the flock is one fly ten times."""
    return brain_seed((fly << 32) ^ lineage)


def _together(brains: list[Brain]) -> ConnectomeFlock | None:
    """Connectome brains on one wiring think in one multiply; anything else, one by one."""
    if len(brains) < 2 or not all(isinstance(brain, ConnectomeBrain) for brain in brains):
        return None
    try:
        return ConnectomeFlock(brains)
    except ValueError:  # different graphs or dynamics: each on its own
        return None


async def fly_flock(
    url: str,
    brains: list[Brain],
    owners: list[str],
    *,
    token: str = "dev",
    lives: int | None = None,
    on_life: Callable[[FlockLife], None] | None = None,
    seeds: list[int] | None = None,
) -> list[FlockLife]:
    """Join every brain as its own owner and fly them all, tick by tick.

    Each fly rejoins after death until it has lived `lives` lives (forever when None).
    `seeds` gives every fly its own stream of chance (0, 1, 2… when not given).
    Returns every finished life; a lost connection raises ConnectionLost.
    """
    if len(brains) != len(owners) or not brains:
        raise ValueError("a flock needs as many owners as brains, and at least one")
    if len(set(owners)) != len(owners):
        raise ValueError("every fly of a flock needs its own owner name")
    seeds = list(range(len(brains))) if seeds is None else seeds
    if len(seeds) != len(brains):
        raise ValueError("a flock needs as many seeds as brains")
    if lives is not None and lives <= 0:
        return []  # nothing to live: like run_brain, without joining anyone
    together = _together(brains)
    mouths = [Mouth(brain) for brain in brains]
    count = len(brains)
    finished: list[FlockLife] = []
    lived = [0] * count
    flying = [False] * count

    async with contextlib.AsyncExitStack() as stack:
        clients = [
            await stack.enter_async_context(AsyncClient(url, owner=owner, token=token))
            for owner in owners
        ]

        async def hatch(index: int) -> None:
            joined = await clients[index].join()
            brains[index].reset(_life_seed(seeds[index], joined.lineage))
            flying[index] = True

        def think(channels: list) -> list[int | None]:
            if together is not None:
                return together.act(channels)
            return [
                None if seen is None else int(brain.act(seen))
                for brain, seen in zip(brains, channels, strict=True)
            ]

        await asyncio.gather(*(hatch(index) for index in range(count)))
        while any(flying):
            active = [index for index in range(count) if flying[index]]
            seen = await asyncio.gather(*(clients[index].next_observation() for index in active))
            observations = [None] * count
            reborn = []
            for index, observation in zip(active, seen, strict=True):
                if observation is not None:
                    observations[index] = observation
                    continue
                flying[index] = False  # it died: its obituary is on the client
                lived[index] += 1
                died = clients[index].last_died
                life = FlockLife(owners[index], died.name, died.lineage, stats_of(died))
                finished.append(life)
                if on_life is not None:
                    on_life(life)
                if lives is None or lived[index] < lives:
                    reborn.append(index)
            channels = [None if seen is None else seen.channels for seen in observations]
            # Off the loop: a big graph thinks for a good part of a tick, and the sockets
            # must be read meanwhile.
            actions = await asyncio.to_thread(think, channels)
            sends = []
            for index, observation in enumerate(observations):
                if observation is None:
                    continue
                sends.append(clients[index].act(observation.tick, actions[index]))
                if (thought := mouths[index].speak(observation.tick)) is not None:
                    sends.append(clients[index].say(thought))
            await asyncio.gather(*sends, *(hatch(index) for index in reborn))
    return finished


def run_flock(
    url: str = DEFAULT_URL,
    brains: list[Brain] | None = None,
    owners: list[str] | None = None,
    **options,
) -> list[FlockLife]:
    """`fly_flock` from ordinary code: blocks until every fly has lived its lives."""
    return asyncio.run(fly_flock(url, brains or [], owners or [], **options))
