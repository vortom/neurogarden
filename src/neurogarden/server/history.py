"""Reading the archive back through the engine: rebuild a world, play a stretch of it, verify.

Every function here builds its own `World` from a snapshot and feeds it the recorded
inputs; the live world is never touched. The engine's determinism is what makes the
archive a faithful record: same inputs, same world, same hashes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from neurogarden.dojo.rewards import BodyState
from neurogarden.dojo.stats import StatsTracker
from neurogarden.engine.world import StepResult, World

from .archive import Archive, ArchiveError, TickInputs


@dataclass
class Rebuilt:
    world: World
    trackers: dict[int, StatsTracker]
    last_meal: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Moment:
    """One recorded tick, re-lived: `world` is the state after it, `result` what happened."""

    tick: int
    world: World
    result: StepResult
    inputs: TickInputs


def apply_inputs(
    world: World, inputs: TickInputs, trackers: dict[int, StatsTracker] | None = None
) -> StepResult:
    """Tell the world what the archive says it was told at this tick, then step it.

    With `trackers`, keep the stats of every fly the way the runner does: a tracker is
    born with a spawn, updated every tick, and dropped once its fly has died.
    """
    if world.tick != inputs.tick:
        raise ArchiveError(f"world is at tick {world.tick}, inputs are for tick {inputs.tick}")
    for agent_id in inputs.despawns:
        world.despawn(agent_id)
    for agent_id, body, x, y in inputs.spawns:
        hatched = world.spawn(body, at=(x, y))
        if hatched != agent_id:
            raise ArchiveError(
                f"tick {inputs.tick}: the archive hatched agent {agent_id}, the engine {hatched}"
            )
        if trackers is not None:
            trackers[agent_id] = StatsTracker(agent_id, (x, y), world.config.day_length)
    result = world.step(inputs.actions)
    if trackers is not None:
        for agent_id, tracker in trackers.items():
            if agent_id in result.observations:
                body = BodyState.from_observation(result.observations[agent_id])
                tracker.update(result.events, body)
        for event in result.events:
            if event.type == "died":
                trackers.pop(event.agent_id, None)
    return result


def world_at(archive: Archive, tick: int) -> Rebuilt:
    """The world as it stood before tick `tick` ran (so `world.tick == tick`)."""
    info = archive.world_info
    if info is None:
        raise ArchiveError(f"{archive.path} holds no world")
    found = archive.latest_snapshot(tick)
    if found is None:
        raise ArchiveError(f"{archive.path} has no snapshot at or before tick {tick}")
    snapshot_tick, state, extras = found
    world = World.restore(state, info.map_text)
    trackers = {
        int(agent_id): StatsTracker.from_dict(data)
        for agent_id, data in extras.get("trackers", {}).items()
    }
    last_meal = {int(agent_id): int(at) for agent_id, at in extras.get("last_meal", {}).items()}
    for inputs in archive.inputs(snapshot_tick, tick):
        apply_inputs(world, inputs, trackers)
    if world.tick != tick:
        raise ArchiveError(f"{archive.path} ends at tick {world.tick}; tick {tick} was asked for")
    return Rebuilt(world, trackers, last_meal)


def rebuild(archive: Archive) -> Rebuilt:
    """The world where the archive left off, ready to run its next tick."""
    return world_at(archive, archive.next_tick())


def playback(archive: Archive, from_tick: int, to_tick: int | None = None) -> Iterator[Moment]:
    """Re-live the recorded ticks in [from_tick, to_tick) on a world of our own."""
    world = world_at(archive, from_tick).world
    for inputs in archive.inputs(from_tick, to_tick):
        result = apply_inputs(world, inputs)
        yield Moment(inputs.tick, world, result, inputs)


@dataclass(frozen=True)
class Verified:
    ticks: int
    checkpoints: int
    snapshots: int


def verify(archive: Archive) -> Verified:
    """Re-run the whole history from its first snapshot and compare every stored hash.

    Raises ArchiveError at the first checkpoint or snapshot the re-run does not reproduce.
    """
    ticks = archive.snapshot_ticks()
    if not ticks:
        raise ArchiveError(f"{archive.path} has no snapshot to start from")
    first = ticks[0]
    snapshots = {tick: archive.latest_snapshot(tick) for tick in ticks}
    checkpoints = archive.checkpoints()
    world = World.restore(snapshots[first][1])
    stepped = checked = matched = 0

    def compare(tick: int) -> None:
        nonlocal checked, matched
        actual = world.state_hash()
        if tick in checkpoints:
            checked += 1
            if checkpoints[tick] != actual:
                raise ArchiveError(
                    f"checkpoint at tick {tick} does not reproduce: "
                    f"stored {checkpoints[tick]}, re-run {actual}"
                )
        if tick in snapshots and tick != first:
            matched += 1
            stored = World.restore(snapshots[tick][1]).state_hash()
            if stored != actual:
                raise ArchiveError(
                    f"snapshot at tick {tick} does not reproduce: stored {stored}, re-run {actual}"
                )

    compare(first)
    for inputs in archive.inputs(first):
        apply_inputs(world, inputs)
        stepped += 1
        compare(world.tick)
    return Verified(stepped, checked, matched)
