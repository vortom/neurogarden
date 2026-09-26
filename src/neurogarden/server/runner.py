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
from .ports import LocalPort
from .roster import Roster

log = logging.getLogger("neurogarden.server")
RECENT_CHRONICLE = 30
SAY_TTL = 150  # ticks a speech bubble stays over a fly


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
        self.spawn_log: list[tuple[int, int, str, int, str]] = []  # tick, agent, owner, life, body
        self.action_log: list[dict[int, int]] = []  # actions applied per tick, idle included
        self.chronicle: deque[tuple[int, str]] = deque(maxlen=RECENT_CHRONICLE)
        self._chronicler = Chronicler(world.config, world.state.width, world.state.height)
        self._queued_joins: list = []
        self._pending: dict[int, tuple[int, int]] = {}
        self._last_sent: dict[int, int] = {}
        self.missed: dict[int, int] = {}
        self._trackers: dict[int, StatsTracker] = {}
        self._says: dict[int, tuple[int, str]] = {}  # agent_id -> (tick said, text)
        self._next_tick_at: float | None = None

    # --- what clients ask --------------------------------------------------------------

    def welcome_for(self, owner: str, role: str):
        return frames.welcome_message(
            owner, role, self.world, self.map_name, self.tps, self.catalog, self.motd
        )

    def is_reserved(self, owner: str) -> bool:
        """True for an owner flown by a hosted brain: no remote connection may take it over."""
        state = self.roster.owners.get(owner)
        return state is not None and isinstance(state.port, LocalPort)

    def attach(self, port) -> None:
        """Bind an agent port to its owner; a previous connection of the owner is superseded."""
        if port.closing:
            return
        superseded = self.roster.attach(port)
        if superseded is not None:
            superseded.close(4004, "superseded by a newer connection")
            log.info("owner %s superseded a connection", port.owner)
        agent_id = self.roster.live_agent(port.owner)
        if agent_id is not None and superseded is None:
            self._note(self._chronicler.reattached(self.world.tick, self._subject(agent_id)))

    def detach(self, port) -> None:
        # A leave cancels a join the same connection queued in this tick: no fly is hatched
        # for a client that changed its mind before the world advanced.
        self._queued_joins = [entry for entry in self._queued_joins if entry[0] is not port]
        agent_id = self.roster.live_agent(port.owner)
        was_attached = self.roster.port_for(agent_id) is port if agent_id is not None else False
        self.roster.detach(port)
        if was_attached:
            self._pending.pop(agent_id, None)  # no stale action replayed on reconnect
            self._last_sent.pop(agent_id, None)  # and no missed charged for the silence
            self._note(self._chronicler.away(self.world.tick, self._subject(agent_id)))

    def request_join(self, port, body: str = "fly") -> None:
        """Queue a join for the next tick; one queued join per port, the newest wins."""
        self._queued_joins = [entry for entry in self._queued_joins if entry[0] is not port]
        self._queued_joins.append((port, body))

    def controls_a_fly(self, port) -> bool:
        agent_id = self.roster.live_agent(port.owner)
        return agent_id is not None and self.roster.port_for(agent_id) is port

    def submit_action(self, port, tick: int, action: int) -> bool:
        """Accept the action for `tick` if this port controls a live fly and the tick is current."""
        if not self.controls_a_fly(port):
            return False
        agent_id = self.roster.live_agent(port.owner)
        if self._last_sent.get(agent_id) != tick:
            return False
        self._pending[agent_id] = (tick, action)
        return True

    def say(self, port, text: str) -> None:
        if not self.controls_a_fly(port):
            return
        agent_id = self.roster.live_agent(port.owner)
        if text:
            self._says[agent_id] = (self.world.tick, text)
        else:
            self._says.pop(agent_id, None)  # an empty say clears the bubble at once

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

        ours = [agent_id for agent_id in result.observations if self.roster.knows(agent_id)]
        subjects = {agent_id: self._subject(agent_id) for agent_id in ours}
        died = {e.agent_id: e for e in result.events if e.type == "died"}
        deadline = self.deadline_ms()
        for agent_id in ours:
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

        says = self._bubbles(self.world.tick)
        if self.spectators:
            frame = frames.frame_message(self.world, self.roster, result.events, tick, says)
            for spectator in self.spectators:
                spectator.deliver(frame)
        for text in self._chronicler.lines(tick, result.events, subjects):
            self._note(text, tick)
        log.debug("tick %d: %d living, %d watching", tick, len(ours), len(self.spectators))

    def _apply_joins(self, tick: int) -> None:
        queued, self._queued_joins = self._queued_joins, []
        for port, body in queued:
            if port.closing:
                continue  # the connection went away before the tick
            state = self.roster.state(port.owner)
            if state.port is not port:
                if state.port is not None:
                    continue  # a superseded connection cannot take the fly back
                self.attach(port)  # a join after leave or a disconnect re-attaches this one
            if state.agent_id is not None:
                port.deliver(
                    frames.joined_message(state.agent_id, state.lineage, state.name, tick, True)
                )
                continue
            try:
                agent_id = self.world.spawn(body)
            except ValueError as err:  # the map is full: the world lives on without this fly
                log.warning("%s could not hatch: %s", port.owner, err)
                port.deliver(frames.error_message("world_full", str(err), fatal=False))
                continue
            lineage, name = self.roster.born(port.owner, agent_id)
            agent = self.world.state.agents[agent_id]
            self.spawn_log.append((tick, agent_id, port.owner, lineage, body))
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

    def _bubbles(self, tick: int) -> dict[int, str]:
        """The speech bubbles still worth drawing; old ones fade away."""
        self._says = {
            agent_id: said for agent_id, said in self._says.items() if tick - said[0] < SAY_TTL
        }
        return {agent_id: text for agent_id, (_, text) in self._says.items()}

    def _bury(self, agent_id: int, tick: int, causes: list[str], port) -> None:
        stats = self._trackers.pop(agent_id).stats
        record = self.roster.record(agent_id)
        self.roster.died(agent_id, stats.lifespan)
        for table in (self._pending, self._last_sent, self.missed, self._says):
            table.pop(agent_id, None)
        if port is not None:
            port.deliver(
                frames.died_message(agent_id, record.lineage, record.name, tick, causes, stats)
            )
        log.info("%s's %s died at tick %d: %s", record.owner, record.name, tick, ", ".join(causes))

    def _subject(self, agent_id: int) -> Subject:
        agent = self.world.state.agents[agent_id]
        record = self.roster.record(agent_id)
        if record is None:
            return Subject("", f"fly {agent_id}", agent.x, agent.y)
        return Subject(record.owner, record.name, agent.x, agent.y)

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
            self._next_tick_at = max(self._next_tick_at + period, self.clock())
            try:
                self.tick()  # rebased first, so deadline_ms() inside the tick is the time left
            except Exception:  # a broken tick stops the world instead of freezing it
                log.exception("tick %d failed; stopping the world", self.world.tick)
                stop.set()
                return
