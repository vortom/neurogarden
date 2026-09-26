"""Needs, health and death (tick order, step 4)."""

from __future__ import annotations

from .body import MOVE_DIRECTION, Action
from .clock import is_night
from .config import NEED_MAX
from .events import Event
from .state import Agent, WorldState
from .tiles import Terrain

# need attribute -> cause of death when it sits at zero
_CAUSES = (("satiety", "starvation"), ("hydration", "dehydration"), ("energy", "exhaustion"))


def _clamp(value: int) -> int:
    return max(0, min(NEED_MAX, value))


def energy_delta(state: WorldState, agent: Agent, action: Action) -> int:
    cfg = state.config
    if action in MOVE_DIRECTION:
        return cfg.energy_move_night if is_night(state.tick, cfg) else cfg.energy_move
    if action == Action.CONSUME:
        return cfg.energy_consume
    if action == Action.REST:
        on_nest = state.terrain[agent.y, agent.x] == Terrain.NEST
        return cfg.energy_rest_nest if on_nest else cfg.energy_rest
    return cfg.energy_idle


def metabolise(state: WorldState, agent: Agent, action: Action, events: list[Event]) -> None:
    cfg = state.config
    before = (agent.satiety, agent.hydration, agent.energy)
    agent.satiety = _clamp(agent.satiety - cfg.satiety_drain)
    agent.hydration = _clamp(agent.hydration - cfg.hydration_drain)
    agent.energy = _clamp(agent.energy + energy_delta(state, agent, action))

    depleted: list[str] = []
    for (need, cause), old in zip(_CAUSES, before, strict=True):
        if getattr(agent, need) == 0:
            depleted.append(cause)
            if old > 0:
                events.append(Event(state.tick, "need_depleted", agent.id, {"need": need}))

    if depleted:
        amount = min(agent.health, cfg.starve_damage * len(depleted))
        agent.health -= amount
        if amount > 0:
            data = {"amount": amount, "causes": list(depleted)}
            events.append(Event(state.tick, "damaged", agent.id, data))
    elif min(agent.satiety, agent.hydration, agent.energy) >= cfg.regen_threshold:
        agent.health = min(NEED_MAX, agent.health + cfg.regen_amount)

    agent.age += 1
    health_dead = agent.health == 0
    old_age_dead = cfg.max_age is not None and agent.age >= cfg.max_age
    if health_dead or old_age_dead:
        causes = list(depleted)
        if old_age_dead:
            causes.append("old_age")
        _die(state, agent, causes, events)


def _die(state: WorldState, agent: Agent, causes: list[str], events: list[Event]) -> None:
    agent.alive = False
    state.occupant[agent.y, agent.x] = 0
    events.append(Event(state.tick, "died", agent.id, {"causes": list(causes)}))
