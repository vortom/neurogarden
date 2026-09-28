"""Builders for the messages the server sends. Pure functions over engine state."""

from __future__ import annotations

from neurogarden.dojo.stats import EpisodeStats
from neurogarden.engine.clock import day_number, light_at
from neurogarden.engine.config import RULES_VERSION
from neurogarden.engine.events import Event
from neurogarden.engine.tiles import Resource
from neurogarden.engine.world import StepResult, World
from neurogarden.protocol.codec import encode_channels
from neurogarden.protocol.messages import (
    PROTOCOL_VERSION,
    AgentEvent,
    AgentView,
    Catalog,
    Chronicle,
    ChronicleMessage,
    Died,
    DiedMessage,
    Error,
    ErrorMessage,
    Frame,
    FrameMessage,
    Joined,
    JoinedMessage,
    Observation,
    ObservationMessage,
    OwnerScore,
    ReplayInfo,
    ReplayInfoMessage,
    ResourceView,
    Stats,
    Welcome,
    WelcomeMessage,
    WorldEvent,
    WorldInfo,
    WorldMap,
    WorldMessage,
)

from .names import mood
from .roster import Roster


def stats_model(stats: EpisodeStats) -> Stats:
    return Stats(
        lifespan=stats.lifespan,
        days=stats.days,
        death_causes=list(stats.death_causes),
        bites=stats.bites,
        drinks=stats.drinks,
        rest_ticks=stats.rest_ticks,
        bumps=stats.bumps,
        tiles_explored=stats.tiles_explored,
        mean_wellbeing=stats.mean_wellbeing,
    )


def welcome_message(
    owner: str, role: str, world: World, map_name: str, tps: float, catalog: Catalog, motd: str
) -> WelcomeMessage:
    info = WorldInfo(
        name=map_name,
        width=world.state.width,
        height=world.state.height,
        tps=tps,
        day_length=world.config.day_length,
        dawn_end=world.config.dawn_end,
        dusk_start=world.config.dusk_start,
        night_start=world.config.night_start,
        rules_version=RULES_VERSION,
    )
    payload = Welcome(
        protocol=PROTOCOL_VERSION, owner=owner, role=role, world=info, catalog=catalog, motd=motd
    )
    return WelcomeMessage(payload=payload)


def world_message(world: World, map_name: str) -> WorldMessage:
    state = world.state
    payload = WorldMap(
        map_name=map_name, width=state.width, height=state.height, terrain=state.terrain.tolist()
    )
    return WorldMessage(payload=payload)


def joined_message(agent_id: int, lineage: int, name: str, tick: int, reattached: bool):
    payload = Joined(
        agent_id=agent_id, lineage=lineage, name=name, tick=tick, reattached=reattached
    )
    return JoinedMessage(payload=payload)


def observation_message(
    result: StepResult, agent_id: int, tick: int, deadline_ms: int, missed: int
) -> ObservationMessage:
    events = [AgentEvent(type=e.type, data=e.data) for e in result.agent_events.get(agent_id, [])]
    payload = Observation(
        tick=tick,
        deadline_ms=deadline_ms,
        channels=encode_channels(result.observations[agent_id]),
        events=events,
        missed=missed,
    )
    return ObservationMessage(payload=payload)


def died_message(
    agent_id: int, lineage: int, name: str, tick: int, causes: list[str], stats: EpisodeStats
) -> DiedMessage:
    payload = Died(
        agent_id=agent_id,
        lineage=lineage,
        name=name,
        tick=tick,
        causes=list(causes),
        stats=stats_model(stats),
    )
    return DiedMessage(payload=payload)


def chronicle_message(tick: int, text: str) -> ChronicleMessage:
    return ChronicleMessage(payload=Chronicle(tick=tick, text=text))


def error_message(code: str, message: str, fatal: bool = False) -> ErrorMessage:
    return ErrorMessage(payload=Error(code=code, message=message, fatal=fatal))


def replay_message(life, speed: float, done: bool) -> ReplayInfoMessage:
    """Who a ghost is (an `archive.Life`), sent before its first frame and after its last."""
    payload = ReplayInfo(
        owner=life.owner,
        lineage=life.lineage,
        name=life.name,
        born_tick=life.born_tick,
        died_tick=life.died_tick,
        lifespan=life.lifespan,
        causes=list(life.causes),
        speed=speed,
        done=done,
    )
    return ReplayInfoMessage(payload=payload)


def frame_message(
    world: World, roster: Roster, events: list[Event], tick: int, says: dict[int, str]
) -> FrameMessage:
    state = world.state
    ys, xs = (state.resource_kind == Resource.FRUIT).nonzero()
    resources = [
        ResourceView(
            x=int(x),
            y=int(y),
            kind=int(state.resource_kind[y, x]),
            amount=int(state.resource_amount[y, x]),
        )
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    ]
    just_died = {event.agent_id for event in events if event.type == "died"}
    agents = []
    for agent_id in sorted(state.agents):
        agent = state.agents[agent_id]
        record = roster.record(agent_id)
        if record is None:
            continue
        if not agent.alive and agent_id not in just_died:
            continue  # a corpse appears in the frame of the tick it died, then no more
        agents.append(
            AgentView(
                agent_id=agent_id,
                owner=record.owner,
                lineage=record.lineage,
                name=record.name,
                x=agent.x,
                y=agent.y,
                facing=agent.facing,
                satiety=agent.satiety,
                hydration=agent.hydration,
                energy=agent.energy,
                health=agent.health,
                age=agent.age,
                alive=agent.alive,
                connected=agent.alive and roster.connected(agent_id),
                mood=mood(agent.satiety, agent.hydration, agent.energy, agent.health, agent.alive),
                say=says.get(agent_id, ""),
            )
        )
    scores = [
        OwnerScore(
            owner=s.owner,
            lives=s.lineage,
            best_lifespan=s.best_lifespan,
            best_lineage=s.best_lineage,
            alive=s.agent_id is not None,
        )
        for s in roster.scores()
    ]
    payload = Frame(
        tick=tick,
        day=day_number(tick, world.config),
        light=light_at(tick, world.config),
        resources=resources,
        agents=agents,
        events=[WorldEvent(type=e.type, agent_id=e.agent_id, data=e.data) for e in events],
        scores=scores,
    )
    return FrameMessage(payload=payload)
