"""Who owns which fly: one live fly per owner, lineage counters, ports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .names import fly_name


@dataclass(frozen=True)
class AgentRecord:
    """Who a fly was: kept after the owner has moved on, so a corpse still has a name."""

    owner: str
    lineage: int
    name: str


@dataclass
class OwnerState:
    owner: str
    lineage: int = 0  # lives so far; the live fly is Fly #lineage
    agent_id: int | None = None
    name: str = ""
    port: Any = None  # the attached AgentPort, or None while the brain is away
    best_lifespan: int = 0
    best_lineage: int = 0  # which life it was; 0 until a first life has ended
    lifespans: list[int] = field(default_factory=list)


class Roster:
    def __init__(self) -> None:
        self.owners: dict[str, OwnerState] = {}
        self.agents: dict[int, AgentRecord] = {}

    def state(self, owner: str) -> OwnerState:
        return self.owners.setdefault(owner, OwnerState(owner))

    def attach(self, port) -> Any:
        """Attach a port to its owner; returns the port it superseded, if any."""
        state = self.state(port.owner)
        superseded = state.port if state.port is not None and state.port is not port else None
        state.port = port
        return superseded

    def detach(self, port) -> None:
        """Forget the port; an owner with no fly and no lives is forgotten with it."""
        state = self.owners.get(port.owner)
        if state is None or state.port is not port:
            return
        state.port = None
        if state.agent_id is None and state.lineage == 0:
            del self.owners[port.owner]  # a hello that never hatched leaves nothing behind

    def live_agent(self, owner: str) -> int | None:
        state = self.owners.get(owner)
        return None if state is None else state.agent_id

    def born(self, owner: str, agent_id: int) -> tuple[int, str]:
        """Record a new fly; returns (lineage, name)."""
        state = self.state(owner)
        state.lineage += 1
        state.agent_id = agent_id
        state.name = fly_name(owner, state.lineage)
        self.agents[agent_id] = AgentRecord(owner, state.lineage, state.name)
        return state.lineage, state.name

    def died(self, agent_id: int, lifespan: int) -> OwnerState:
        record = self.agents[agent_id]
        state = self.state(record.owner)
        state.agent_id = None
        self._ended(state, record.lineage, lifespan)
        return state

    @staticmethod
    def _ended(state: OwnerState, lineage: int, lifespan: int) -> None:
        state.lifespans.append(lifespan)
        if state.best_lineage == 0 or lifespan > state.best_lifespan:  # a tie keeps the first
            state.best_lifespan, state.best_lineage = lifespan, lineage

    def restore(self, lives) -> None:
        """Rebuild owners and records from archived lives (`archive.Life`), oldest first."""
        for life in sorted(lives, key=lambda life: (life.owner, life.lineage)):
            state = self.state(life.owner)
            state.lineage = max(state.lineage, life.lineage)
            state.name = life.name  # the latest fly's, alive or not, as `born` leaves it
            self.agents[life.agent_id] = AgentRecord(life.owner, life.lineage, life.name)
            if life.died_tick is None:
                state.agent_id = life.agent_id
            else:
                self._ended(state, life.lineage, life.lifespan or 0)

    def record(self, agent_id: int) -> AgentRecord | None:
        return self.agents.get(agent_id)

    def knows(self, agent_id: int) -> bool:
        """False for a fly the runner never hatched — a world may arrive with agents in it."""
        return agent_id in self.agents

    def owner_of(self, agent_id: int) -> str:
        return self.agents[agent_id].owner

    def port_for(self, agent_id: int) -> Any:
        """The port steering this fly; a corpse has none, even if its owner flies again."""
        record = self.agents.get(agent_id)
        state = self.owners.get(record.owner) if record is not None else None
        if state is None or state.agent_id != agent_id:
            return None
        return state.port

    def connected(self, agent_id: int) -> bool:
        return self.port_for(agent_id) is not None

    def scores(self) -> list[OwnerState]:
        """Owners who have hatched at least one fly: best lifespan first, then name."""
        lived = [state for state in self.owners.values() if state.lineage > 0]
        return sorted(lived, key=lambda s: (-s.best_lifespan, s.owner))


class GhostRoster(Roster):
    """Names for archived flies. Every ghost counts as connected; the hall of flies stays empty."""

    def __init__(self, lives) -> None:
        super().__init__()
        self.restore(lives)

    def connected(self, agent_id: int) -> bool:
        return True

    def scores(self) -> list[OwnerState]:
        return []
