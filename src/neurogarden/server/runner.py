"""WorldRunner: the only place the live world changes, one tick at a time."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable

from neurogarden.dojo.rewards import BodyState
from neurogarden.dojo.stats import StatsTracker
from neurogarden.engine.body import Action
from neurogarden.engine.world import World
from neurogarden.protocol.catalog import build_catalog

from . import frames
from .chronicle import Chronicler, Subject
from .roster import Roster

log = logging.getLogger("neurogarden.server")
RECENT_CHRONICLE = 30


class WorldRunner:
    def __init__(
        self,
        world: World,
        *,
        tps: float = 5.0,
        map_name: str = "drosoville",
        motd: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if tps <= 0:
            raise ValueError("tps must be positive")
        self.world = world
        self.tps = tps
        self.map_name = map_name
        self.motd = motd
        self.clock = clock
        self.catalog = build_catalog()
        self.roster = Roster()
        self.spectators: list = []
        self.spawn_log: list[tuple[int, int, str, int]] = []  # (tick, agent_id, owner, lineage)
        self.action_log: list[dict[int, int]] = []  # actions applied per tick, idle included
        self.chronicle: deque[tuple[int, str]] = deque(maxlen=RECENT_CHRONICLE)
        self._chronicler = Chronicler(world.config, world.state.width, world.state.height)
        self._queued_joins: list = []
        self._pending: dict[int, tuple[int, int]] = {}
        self._last_sent: dict[int, int] = {}
        self.missed: dict[int, int] = {}
        self._trackers: dict[int, StatsTracker] = {}
        self._says: dict[int, str] = {}
        self._next_tick_at: float | None = None

    # --- what clients ask --------------------------------------------------------------

    def welcome_for(self, owner: str, role: str):
        return frames.welcome_message(
            owner, role, self.world, self.map_name, self.tps, self.catalog, self.motd
        )

    def attach(self, port) -> None:
        """Bind an agent port to its owner; a previous connection of the owner is superseded."""
        superseded = self.roster.attach(port)
        if superseded is not None:
            superseded.close(4004, "superseded by a newer connection")
            log.info("owner %s superseded a connection", port.owner)
        agent_id = self.roster.live_agent(port.owner)
        if agent_id is not None and superseded is None:
            self._note(self._chronicler.reattached(self.world.tick, self._subject(agent_id)))

    def detach(self, port) -> None:
        agent_id = self.roster.live_agent(port.owner)
        was_attached = self.roster.port_for(agent_id) is port if agent_id is not None else False
        self.roster.detach(port)
        if was_attached:
            self._note(self._chronicler.away(self.world.tick, self._subject(agent_id)))

    def request_join(self, port, body: str = "fly") -> None:
        self._queued_joins.append((port, body))

    def submit_action(self, port, tick: int, action: int) -> bool:
        """Accept the action for `tick` if this port controls a live fly and the tick is current."""
        agent_id = self.roster.live_agent(port.owner)
        if agent_id is None or self.roster.port_for(agent_id) is not port:
            return False
        if self._last_sent.get(agent_id) != tick:
            return False
        self._pending[agent_id] = (tick, action)
        return True

    def say(self, port, text: str) -> None:
        agent_id = self.roster.live_agent(port.owner)
        if agent_id is not None and self.roster.port_for(agent_id) is port:
            self._says[agent_id] = text

    def add_spectator(self, port) -> None:
        self.spectators.append(port)
        port.deliver(frames.world_message(self.world, self.map_name))
        for tick, text in self.chronicle:
            port.deliver(frames.chronicle_message(tick, text))

    def remove_spectator(self, port) -> None:
        if port in self.spectators:
            self.spectators.remove(port)

    def deadline_ms(self) -> int:
        if self._next_tick_at is None:
            return int(1000 / self.tps)
        return max(0, int((self._next_tick_at - self.clock()) * 1000))

    # --- the tick ---------------------------------------------------------------------

    def tick(self) -> None:
        tick = self.world.tick
        self._apply_joins(tick)
        actions = self._collect_actions()
        result = self.world.step(actions)
        self.action_log.append(actions)

        for agent_id, tracker in self._trackers.items():
            if agent_id in result.observations:
                body = BodyState.from_observation(result.observations[agent_id])
                tracker.update(result.events, body)

        subjects = {agent_id: self._subject(agent_id) for agent_id in result.observations}
        died = {e.agent_id: e for e in result.events if e.type == "died"}
        deadline = self.deadline_ms()
        for agent_id in result.observations:
            port = self.roster.port_for(agent_id)
            if port is None:
                if agent_id in died:
                    self._bury(agent_id, tick, died[agent_id].data["causes"], None)
                continue
            message = frames.observation_message(
                result, agent_id, tick, deadline, self.missed.get(agent_id, 0)
            )
            if agent_id in died:
                port.deliver(message)  # the final observation, then the obituary
                self._bury(agent_id, tick, died[agent_id].data["causes"], port)
            else:
                self._last_sent[agent_id] = tick  # before delivery: a local brain answers at once
                port.deliver(message)

        if self.spectators:
            frame = frames.frame_message(self.world, self.roster, result.events, tick, self._says)
            for spectator in self.spectators:
                spectator.deliver(frame)
        for text in self._chronicler.lines(tick, result.events, subjects):
            self._note(text, tick)
        log.debug(
            "tick %d: %d living, %d watching", tick, len(result.observations), len(self.spectators)
        )

    def _apply_joins(self, tick: int) -> None:
        queued, self._queued_joins = self._queued_joins, []
        for port, body in queued:
            state = self.roster.state(port.owner)
            if state.port is not port:
                continue  # superseded or gone before the tick
            if state.agent_id is not None:
                port.deliver(
                    frames.joined_message(state.agent_id, state.lineage, state.name, tick, True)
                )
                continue
            agent_id = self.world.spawn(body)
            lineage, name = self.roster.born(port.owner, agent_id)
            agent = self.world.state.agents[agent_id]
            self.spawn_log.append((tick, agent_id, port.owner, lineage))
            self._trackers[agent_id] = StatsTracker(
                agent_id, (agent.x, agent.y), self.world.config.day_length
            )
            self.missed[agent_id] = 0
            port.deliver(frames.joined_message(agent_id, lineage, name, tick, False))
            self._note(self._chronicler.born(tick, self._subject(agent_id), lineage), tick)
            log.info("%s joined as %s (#%d, agent %d)", port.owner, name, lineage, agent_id)

    def _collect_actions(self) -> dict[int, int]:
        actions: dict[int, int] = {}
        for agent in self.world.state.living():
            port = self.roster.port_for(agent.id)
            sent = self._last_sent.get(agent.id)
            if port is None or sent is None:
                actions[agent.id] = int(Action.IDLE)
                continue
            pending = self._pending.pop(agent.id, None)
            if pending is not None and pending[0] == sent:
                actions[agent.id] = pending[1]
            else:
                actions[agent.id] = int(Action.IDLE)
                self.missed[agent.id] = self.missed.get(agent.id, 0) + 1
        return actions

    def _bury(self, agent_id: int, tick: int, causes: list[str], port) -> None:
        stats = self._trackers.pop(agent_id).stats
        owner_state = self.roster.state(self.roster.owner_of(agent_id))
        lineage, name = owner_state.lineage, owner_state.name
        self.roster.died(agent_id, stats.lifespan)
        for table in (self._pending, self._last_sent, self.missed, self._says):
            table.pop(agent_id, None)
        if port is not None:
            port.deliver(frames.died_message(agent_id, lineage, name, tick, causes, stats))
        log.info("%s's %s died at tick %d: %s", owner_state.owner, name, tick, ", ".join(causes))

    def _subject(self, agent_id: int) -> Subject:
        agent = self.world.state.agents[agent_id]
        owner = self.roster.owner_of(agent_id)
        state = self.roster.state(owner)
        name = state.name if state.agent_id == agent_id else f"fly {agent_id}"
        return Subject(owner, name, agent.x, agent.y)

    def _note(self, text: str, tick: int | None = None) -> None:
        tick = self.world.tick if tick is None else tick
        self.chronicle.append((tick, text))
        message = frames.chronicle_message(tick, text)
        for spectator in self.spectators:
            spectator.deliver(message)

    # --- the clock --------------------------------------------------------------------

    async def run(self, stop: asyncio.Event) -> None:
        """Tick on absolute deadlines; an overrun starts the next tick at once, no bursts."""
        period = 1.0 / self.tps
        self._next_tick_at = self.clock() + period
        while not stop.is_set():
            delay = self._next_tick_at - self.clock()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                    break
                except TimeoutError:
                    pass
            self.tick()
            now = self.clock()
            self._next_tick_at = max(self._next_tick_at + period, now)
