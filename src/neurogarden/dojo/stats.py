"""Episode statistics, fitness and the public score. World-side: built from full events."""

from __future__ import annotations

from dataclasses import dataclass

from neurogarden.engine.events import Event

from .rewards import MAX_DRIVE, BodyState, drive


@dataclass
class EpisodeStats:
    """Counts only, never coordinates."""

    lifespan: int = 0  # ticks lived; the public score
    days: int = 0  # full days lived
    death_causes: tuple[str, ...] = ()
    bites: int = 0
    drinks: int = 0
    rest_ticks: int = 0
    bumps: int = 0
    tiles_explored: int = 1
    mean_wellbeing: float = 0.0

    @property
    def score(self) -> int:
        return self.lifespan


class StatsTracker:
    _COUNTERS = {"ate": "bites", "drank": "drinks", "rested": "rest_ticks", "bumped": "bumps"}

    def __init__(self, agent_id: int, start: tuple[int, int], day_length: int) -> None:
        self.stats = EpisodeStats()
        self._agent_id = agent_id
        self._day_length = day_length
        self._visited = {start}
        self._wellbeing_sum = 0.0
        self._updates = 0

    def update(self, events: list[Event], body: BodyState) -> None:
        stats = self.stats
        for event in events:
            if event.agent_id != self._agent_id:
                continue
            if event.type in self._COUNTERS:
                name = self._COUNTERS[event.type]
                setattr(stats, name, getattr(stats, name) + 1)
            elif event.type == "moved":
                self._visited.add(tuple(event.data["to"]))
            elif event.type == "died":
                stats.death_causes = tuple(event.data["causes"])
        stats.lifespan = body.age
        stats.days = body.age // self._day_length
        stats.tiles_explored = len(self._visited)
        self._updates += 1
        self._wellbeing_sum += 1.0 - drive(body) / MAX_DRIVE
        stats.mean_wellbeing = self._wellbeing_sum / self._updates


def fitness_lifespan(stats: EpisodeStats) -> float:
    """Default fitness for evolution: how long the fly lived."""
    return float(stats.lifespan)
