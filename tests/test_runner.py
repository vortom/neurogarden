"""The runner without sockets: fake ports, ticks by hand."""

import asyncio
import json

from neurogarden.brains import ScriptedBrain
from neurogarden.engine import Action, Config, World, maps
from neurogarden.protocol.messages import encode
from neurogarden.server.ports import LocalPort
from neurogarden.server.runner import WorldRunner

TINY = """
#######
#..F..#
#.N..~#
#######
"""


class FakePort:
    def __init__(self, owner, role="agent"):
        self.owner = owner
        self.role = role
        self.inbox = []
        self.closed = None

    def deliver(self, message):
        self.inbox.append(message)

    def close(self, code, reason):
        self.closed = (code, reason)

    def of(self, kind):
        return [m.payload for m in self.inbox if m.type == kind]

    def last(self, kind):
        return self.of(kind)[-1]


def make_runner(**overrides):
    config = overrides.pop("config", Config())
    world = World.from_map(overrides.pop("map", TINY), config, seed=1)
    return WorldRunner(world, tps=overrides.pop("tps", 10.0), map_name="tiny")


def joined_agent(runner, owner="alice"):
    port = FakePort(owner)
    runner.attach(port)
    runner.request_join(port)
    runner.tick()
    return port


def test_join_is_queued_until_the_tick_then_spawns_and_observes():
    runner = make_runner()
    port = FakePort("alice")
    runner.attach(port)
    runner.request_join(port)
    assert port.inbox == [] and runner.world.state.agents == {}
    runner.tick()
    joined = port.last("joined")
    assert (joined.agent_id, joined.lineage, joined.tick, joined.reattached) == (1, 1, 0, False)
    assert joined.name
    observation = port.last("observation")
    assert observation.tick == 0 and observation.missed == 0
    assert observation.channels.body[4] == 1  # the fly idled through its birth tick
    assert runner.spawn_log == [(0, 1, "alice", 1)]
    assert runner.action_log == [{1: int(Action.IDLE)}]
    assert json.loads(encode(port.inbox[-1]))["type"] == "observation"


def test_action_for_the_current_tick_is_applied_and_late_ones_count_as_missed():
    runner = make_runner()
    port = joined_agent(runner)
    assert runner.submit_action(port, tick=0, action=int(Action.MOVE_E))
    runner.tick()
    assert runner.action_log[-1] == {1: int(Action.MOVE_E)}
    assert runner.world.state.agents[1].x == 3
    assert port.last("observation").missed == 0
    assert not runner.submit_action(port, tick=0, action=1)  # stale
    assert not runner.submit_action(port, tick=5, action=1)  # from the future
    runner.tick()
    assert runner.action_log[-1] == {1: int(Action.IDLE)}
    assert port.last("observation").missed == 1
    assert runner.submit_action(port, tick=2, action=int(Action.REST))
    assert runner.submit_action(port, tick=2, action=int(Action.MOVE_W))  # latest wins
    runner.tick()
    assert runner.action_log[-1] == {1: int(Action.MOVE_W)}


def test_detached_fly_idles_without_counting_and_reattach_resumes():
    runner = make_runner()
    port = joined_agent(runner)
    runner.detach(port)
    runner.tick()
    runner.tick()
    assert port.of("observation")[-1].tick == 0  # nothing new arrived while away
    assert runner.missed[1] == 0
    back = FakePort("alice")
    runner.attach(back)
    runner.request_join(back)
    runner.tick()
    assert back.last("joined").reattached and back.last("joined").agent_id == 1
    assert back.last("observation").tick == 3
    assert any("wandered off" in text for _, text in runner.chronicle)
    assert any("back at the controls" in text for _, text in runner.chronicle)


def test_supersede_closes_the_old_port_and_moves_the_fly():
    runner = make_runner()
    old = joined_agent(runner)
    new = FakePort("alice")
    runner.attach(new)
    assert old.closed == (4004, "superseded by a newer connection")
    runner.request_join(new)
    runner.tick()
    assert new.last("joined").reattached
    assert runner.submit_action(new, tick=1, action=0)
    assert not runner.submit_action(old, tick=1, action=0)


def test_death_sends_final_observation_then_died_and_frees_the_slot():
    runner = make_runner(config=Config(initial_satiety=2, initial_health=5))
    port = joined_agent(runner)
    runner.tick()  # satiety 2 -> 1 -> 0: starve damage 5 kills a fly with 5 health
    assert [m.type for m in port.inbox[-2:]] == ["observation", "died"]
    died = port.last("died")
    assert (died.agent_id, died.tick, died.causes) == (1, 1, ["starvation"])
    assert died.stats.lifespan == 2 and died.stats.death_causes == ["starvation"]
    assert runner.roster.live_agent("alice") is None
    assert 1 not in runner.missed
    runner.request_join(port)
    runner.tick()
    assert (port.last("joined").agent_id, port.last("joined").lineage) == (2, 2)
    assert runner.roster.state("alice").best_lifespan == 2
    assert any("dies of starvation" in text for _, text in runner.chronicle)


def test_spectators_get_the_world_then_a_frame_per_tick_with_moods_and_scores():
    runner = make_runner()
    joined_agent(runner)
    watcher = FakePort("watcher", role="spectator")
    runner.add_spectator(watcher)
    assert watcher.inbox[0].type == "world"
    assert watcher.inbox[0].payload.terrain[2][2] == 5  # the nest
    assert any(m.type == "chronicle" for m in watcher.inbox)  # the hatching, replayed
    runner.tick()
    frame = watcher.last("frame")
    assert frame.tick == 1 and len(frame.agents) == 1
    fly = frame.agents[0]
    assert (fly.owner, fly.lineage, fly.connected, fly.mood) == ("alice", 1, True, "content")
    assert frame.scores[0].owner == "alice" and frame.scores[0].alive
    assert [r[2] for r in frame.resources] == [1]  # the map's F tile
    runner.remove_spectator(watcher)
    runner.tick()
    assert watcher.last("frame").tick == 1


def test_say_shows_in_the_frame_for_the_fly_that_said_it():
    runner = make_runner()
    port = joined_agent(runner)
    watcher = FakePort("w", role="spectator")
    runner.add_spectator(watcher)
    runner.say(port, "fruit?")
    runner.tick()
    assert watcher.last("frame").agents[0].say == "fruit?"


def test_local_port_plays_a_brain_and_rejoins_after_death():
    runner = make_runner(config=Config(initial_satiety=3, initial_health=5))
    port = LocalPort(ScriptedBrain(), "npc-scripted-1", runner, runner.catalog.bodies["fly"])
    runner.attach(port)
    runner.request_join(port)
    for _ in range(12):
        runner.tick()
    assert port.lives >= 2
    assert runner.roster.state("npc-scripted-1").lineage >= 2
    played = [action for tick in runner.action_log for action in tick.values()]
    assert any(action != int(Action.IDLE) for action in played)  # the brain acted, in-process


def test_deadline_is_the_period_when_not_running_and_the_remaining_time_when_running():
    runner = make_runner(tps=4.0)
    assert runner.deadline_ms() == 250

    async def run_briefly():
        stop = asyncio.Event()
        task = asyncio.create_task(runner.run(stop))
        await asyncio.sleep(0.05)
        during = runner.deadline_ms()
        stop.set()
        await task
        return during

    during = asyncio.run(run_briefly())
    assert 0 <= during <= 250


def test_run_ticks_on_the_wall_clock_and_stops_cleanly():
    runner = make_runner(tps=50.0)

    async def run_for(seconds):
        stop = asyncio.Event()
        task = asyncio.create_task(runner.run(stop))
        await asyncio.sleep(seconds)
        stop.set()
        await task

    asyncio.run(run_for(0.25))
    assert 6 <= runner.world.tick <= 16


def test_tick_order_is_deterministic_from_the_logs():
    runner = make_runner(map=maps.load("drosoville"))
    alice, bob = joined_agent(runner, "alice"), joined_agent(runner, "bob")
    for step in range(30):
        runner.submit_action(alice, alice.last("observation").tick, (step % 4) + 1)
        runner.submit_action(bob, bob.last("observation").tick, int(Action.REST))
        runner.tick()
    replay = World.from_map(maps.load("drosoville"), Config(), seed=1)
    for tick, actions in enumerate(runner.action_log):
        for spawn_tick, agent_id, _, _ in runner.spawn_log:
            if spawn_tick == tick:
                assert replay.spawn() == agent_id
        replay.step(actions)
    assert replay.state_hash() == runner.world.state_hash()
