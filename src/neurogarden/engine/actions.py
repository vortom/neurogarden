"""Apply one agent's action to the world (tick order, step 3)."""

from __future__ import annotations

from .body import DIRECTIONS, MOVE_DIRECTION, Action
from .config import NEED_MAX
from .events import Event
from .fruit import remove_fruit
from .state import Agent, WorldState
from .tiles import Resource, Terrain


def water_adjacent(state: WorldState, x: int, y: int) -> bool:
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if state.in_bounds(nx, ny) and state.terrain[ny, nx] == Terrain.WATER:
            return True
    return False


def apply_action(state: WorldState, agent: Agent, action: int, events: list[Event]) -> Action:
    """Apply the action and return the effective one (IDLE for an invalid id)."""
    agent.bumped = False
    try:
        act = Action(action)
    except ValueError:
        events.append(Event(state.tick, "invalid_action", agent.id, {"action": str(action)}))
        return Action.IDLE
    if act in MOVE_DIRECTION:
        _move(state, agent, MOVE_DIRECTION[act], events)
    elif act == Action.CONSUME:
        _consume(state, agent, events)
    elif act == Action.REST:
        on_nest = bool(state.terrain[agent.y, agent.x] == Terrain.NEST)
        events.append(Event(state.tick, "rested", agent.id, {"on_nest": on_nest}))
    return act


def _move(state: WorldState, agent: Agent, direction: int, events: list[Event]) -> None:
    dx, dy = DIRECTIONS[direction]
    agent.facing = direction
    tx, ty = agent.x + dx, agent.y + dy
    if state.walkable(tx, ty) and state.occupant[ty, tx] == 0:
        state.occupant[agent.y, agent.x] = 0
        state.occupant[ty, tx] = agent.id
        data = {"from": (agent.x, agent.y), "to": (tx, ty), "direction": direction}
        events.append(Event(state.tick, "moved", agent.id, data))
        agent.x, agent.y = tx, ty
    else:
        agent.bumped = True
        events.append(Event(state.tick, "bumped", agent.id, {"direction": direction}))


def _consume(state: WorldState, agent: Agent, events: list[Event]) -> None:
    cfg = state.config
    x, y = agent.x, agent.y
    can_eat = state.resource_kind[y, x] == Resource.FRUIT and state.resource_amount[y, x] > 0
    can_drink = water_adjacent(state, x, y)
    if can_eat and (not can_drink or agent.satiety <= agent.hydration):
        agent.satiety = min(NEED_MAX, agent.satiety + cfg.fruit_bite_satiety)
        state.resource_amount[y, x] -= 1
        bites_left = int(state.resource_amount[y, x])
        if bites_left == 0:
            remove_fruit(state, x, y)
        events.append(Event(state.tick, "ate", agent.id, {"bites_left": bites_left}))
    elif can_drink:
        agent.hydration = min(NEED_MAX, agent.hydration + cfg.drink_hydration)
        events.append(Event(state.tick, "drank", agent.id))
    else:
        events.append(Event(state.tick, "consume_failed", agent.id))
