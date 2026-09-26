"""World: the engine's public face. Pure: no I/O, no clock, no reward."""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

from . import snapshot as _snapshot
from .actions import apply_action
from .body import Action, get_body
from .config import Config
from .events import Event, for_agent
from .fruit import seed_initial_fruit, update_fruit
from .metabolism import metabolise
from .senses import ScentFields, observe
from .state import Agent, WorldState, new_state
from .tiles import Terrain, find_tiles, parse_map

FACING_SOUTH = 2


@dataclass(frozen=True)
class StepResult:
    tick: int  # the world's tick after this step
    observations: dict[int, dict[str, np.ndarray]]
    events: list[Event]  # full events with world coordinates: spectators, storage, stats
    agent_events: dict[int, list[Event]]  # own events, coordinates stripped: brains, rewards


class World:
    def __init__(self, state: WorldState, map_text: str | None = None) -> None:
        self._state = state
        self._scents = ScentFields(state)
        self.map_text = map_text

    @classmethod
    def from_map(cls, map_text: str, config: Config | None = None, seed: int = 0) -> World:
        parsed = parse_map(map_text)
        state = new_state(parsed, config or Config(), seed)
        seed_initial_fruit(state)
        return cls(state, parsed.text)

    @classmethod
    def restore(cls, data: dict) -> World:
        return cls(_snapshot.restore(data))

    @property
    def state(self) -> WorldState:
        return self._state

    @property
    def config(self) -> Config:
        return self._state.config

    @property
    def tick(self) -> int:
        return self._state.tick

    def spawn(self, body: str = "fly", at: tuple[int, int] | None = None) -> int:
        state = self._state
        get_body(body)  # raises ValueError for an unknown body
        if at is None:
            at = self._default_spawn_tile()
        x, y = at
        for coordinate in (x, y):
            if isinstance(coordinate, bool) or not isinstance(coordinate, numbers.Integral):
                raise ValueError(f"spawn coordinates must be integers, got {at!r}")
        x, y = int(x), int(y)  # numpy ints must never reach the snapshot
        if not state.walkable(x, y) or state.occupant[y, x] != 0:
            raise ValueError(f"cannot spawn at {at}: tile is not walkable or is occupied")
        cfg = state.config
        agent = Agent(
            id=state.next_agent_id,
            body=body,
            x=x,
            y=y,
            facing=FACING_SOUTH,
            satiety=cfg.initial_satiety,
            hydration=cfg.initial_hydration,
            energy=cfg.initial_energy,
            health=cfg.initial_health,
        )
        state.agents[agent.id] = agent
        state.occupant[y, x] = agent.id
        state.next_agent_id += 1
        return agent.id

    def _default_spawn_tile(self) -> tuple[int, int]:
        state = self._state
        for x, y in find_tiles(state.terrain, Terrain.NEST):
            if state.occupant[y, x] == 0:
                return x, y
        for y in range(state.height):
            for x in range(state.width):
                if state.walkable(x, y) and state.occupant[y, x] == 0:
                    return x, y
        raise ValueError("no free walkable tile to spawn on")

    def observe(self, agent_id: int) -> dict[str, np.ndarray]:
        agent = self._agent(agent_id)
        if not agent.alive:
            raise ValueError(f"agent {agent_id} is dead")
        return observe(self._state, self._scents, agent)

    def step(self, actions: dict[int, int]) -> StepResult:
        state = self._state
        for agent_id in actions:
            self._agent(agent_id)
        living = state.living()
        order = [agent.id for agent in living]
        state.rng.shuffle(order)

        events: list[Event] = []
        effective: dict[int, Action] = {}
        for agent_id in order:
            action = actions.get(agent_id, Action.IDLE)
            effective[agent_id] = apply_action(state, state.agents[agent_id], action, events)
        for agent in living:
            metabolise(state, agent, effective[agent.id], events)
        update_fruit(state, events)
        state.tick += 1

        return StepResult(
            tick=state.tick,
            observations={a.id: observe(state, self._scents, a) for a in living},
            events=events,
            agent_events={
                a.id: [for_agent(e) for e in events if e.agent_id == a.id] for a in living
            },
        )

    def snapshot(self) -> dict:
        return _snapshot.snapshot(self._state)

    def state_hash(self) -> str:
        return _snapshot.state_hash(self._state)

    def _agent(self, agent_id: int) -> Agent:
        try:
            return self._state.agents[agent_id]
        except KeyError:
            raise ValueError(f"unknown agent id {agent_id!r}") from None
