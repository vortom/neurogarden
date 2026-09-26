"""The naturalist's log: plain sentences the server writes from what happened."""

from __future__ import annotations

from dataclasses import dataclass

from neurogarden.engine.clock import day_number
from neurogarden.engine.config import Config
from neurogarden.engine.events import Event

_SECTORS = (
    ("the north-west", "the north", "the north-east"),
    ("the west", "the centre", "the east"),
    ("the south-west", "the south", "the south-east"),
)
_MEAL_GAP = 50  # ticks: one "finds fruit" line per meal, not per bite


@dataclass(frozen=True)
class Subject:
    owner: str
    name: str
    x: int
    y: int


def sector(x: int, y: int, width: int, height: int) -> str:
    """Rough compass position of a tile: 'the north-east', 'the centre'."""
    column = min(2, x * 3 // max(1, width))
    row = min(2, y * 3 // max(1, height))
    return _SECTORS[row][column]


def time_of_day(tick: int, config: Config) -> str:
    phase = tick % config.day_length
    if phase < config.dawn_end:
        return "dawn"
    if phase < config.dusk_start:
        return "day"
    if phase < config.night_start:
        return "dusk"
    return "night"


def stamp(tick: int, config: Config) -> str:
    return f"Day {day_number(tick, config)}, {time_of_day(tick, config)}"


class Chronicler:
    """Turns what happened into at most a few lines per tick; quiet ticks say nothing."""

    def __init__(self, config: Config, width: int, height: int) -> None:
        self._config = config
        self._width = width
        self._height = height
        self._last_meal: dict[int, int] = {}

    def _where(self, subject: Subject) -> str:
        return sector(subject.x, subject.y, self._width, self._height)

    def born(self, tick: int, subject: Subject, lineage: int) -> str:
        life = "" if lineage == 1 else f" (life #{lineage})"
        when = stamp(tick, self._config)
        return f"{when}: {subject.owner}'s {subject.name}{life} hatches at the nest."

    def reattached(self, tick: int, subject: Subject) -> str:
        when = stamp(tick, self._config)
        return f"{when}: {subject.owner} is back at the controls of {subject.name}."

    def away(self, tick: int, subject: Subject) -> str:
        when = stamp(tick, self._config)
        return f"{when}: {subject.owner} wandered off; {subject.name} sits still."

    def lines(self, tick: int, events: list[Event], subjects: dict[int, Subject]) -> list[str]:
        """subjects: agent_id -> Subject for every agent an event may mention."""
        out: list[str] = []
        when = stamp(tick, self._config)
        for event in events:
            subject = subjects.get(event.agent_id) if event.agent_id is not None else None
            if subject is None:
                continue
            who = f"{subject.name} ({subject.owner})"
            if event.type == "ate":
                if tick - self._last_meal.get(event.agent_id, -_MEAL_GAP) < _MEAL_GAP:
                    continue
                self._last_meal[event.agent_id] = tick
                out.append(f"{when}: {who} finds fruit in {self._where(subject)}.")
            elif event.type == "need_depleted":
                need = event.data.get("need", "something")
                out.append(f"{when}: {who} has run out of {need}.")
            elif event.type == "died":
                causes = ", ".join(event.data.get("causes", [])) or "unknown causes"
                out.append(f"{when}: {who} dies of {causes} in {self._where(subject)}.")
        return out
