"""Events: what happened during a tick."""

from __future__ import annotations

from dataclasses import dataclass, field

EVENT_TYPES = frozenset(
    {
        "moved",
        "bumped",
        "ate",
        "drank",
        "consume_failed",
        "rested",
        "invalid_action",
        "need_depleted",
        "damaged",
        "died",
        "fruit_spawned",
        "fruit_rotted",
    }
)

# Keys that carry world coordinates. Brains must never see them.
_COORDINATE_KEYS = ("x", "y", "from", "to")


@dataclass(frozen=True)
class Event:
    tick: int
    type: str
    agent_id: int | None = None
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.type!r}")


def for_agent(event: Event) -> Event:
    """Copy of the event with world coordinates stripped (agent stream)."""
    data = {key: value for key, value in event.data.items() if key not in _COORDINATE_KEYS}
    return Event(event.tick, event.type, event.agent_id, data)
