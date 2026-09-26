"""Who owns which fly: one live fly per owner, lineage counters, ports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .names import fly_name


@dataclass
class OwnerState:
    owner: str
    lineage: int = 0  # lives so far; the live fly is Fly #lineage
    agent_id: int | None = None
    name: str = ""
    port: Any = None  # the attached AgentPort, or None while the brain is away
    best_lifespan: int = 0
    lifespans: list[int] = field(default_factory=list)


class Roster:
    def __init__(self) -> None:
        self.owners: dict[str, OwnerState] = {}
        self.agent_owner: dict[int, str] = {}

    def state(self, owner: str) -> OwnerState:
        return self.owners.setdefault(owner, OwnerState(owner))

    def attach(self, port) -> Any:
        """Attach a port to its owner; returns the port it superseded, if any."""
        state = self.state(port.owner)
        superseded = state.port if state.port is not None and state.port is not port else None
        state.port = port
        return superseded

    def detach(self, port) -> None:
        state = self.owners.get(port.owner)
        if state is not None and state.port is port:
            state.port = None

    def live_agent(self, owner: str) -> int | None:
        return self.state(owner).agent_id

    def born(self, owner: str, agent_id: int) -> tuple[int, str]:
        """Record a new fly; returns (lineage, name)."""
        state = self.state(owner)
        state.lineage += 1
        state.agent_id = agent_id
        state.name = fly_name(owner, state.lineage)
        self.agent_owner[agent_id] = owner
        return state.lineage, state.name

    def died(self, agent_id: int, lifespan: int) -> OwnerState:
        state = self.state(self.agent_owner[agent_id])
        state.agent_id = None
        state.lifespans.append(lifespan)
        state.best_lifespan = max(state.best_lifespan, lifespan)
        return state

    def owner_of(self, agent_id: int) -> str:
        return self.agent_owner[agent_id]

    def port_for(self, agent_id: int) -> Any:
        state = self.owners.get(self.agent_owner.get(agent_id, ""))
        return None if state is None else state.port

    def connected(self, agent_id: int) -> bool:
        return self.port_for(agent_id) is not None

    def scores(self) -> list[OwnerState]:
        """Owners ordered for a leaderboard: best lifespan first, then name."""
        return sorted(self.owners.values(), key=lambda s: (-s.best_lifespan, s.owner))
