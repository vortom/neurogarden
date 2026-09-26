"""The runner without sockets: fake ports, ticks by hand."""

import asyncio
import json

from websockets.exceptions import ConnectionClosedError

from neurogarden.brains import ScriptedBrain
from neurogarden.engine import Action, Config, World, maps
from neurogarden.protocol.messages import encode
from neurogarden.server import frames
from neurogarden.server.ports import LocalPort, RemotePort
from neurogarden.server.runner import SAY_TTL, WorldRunner

TINY = """
#######
#..F..#
#.N..~#
#######
"""
CRAMPED = """
###
#N#
###
"""


class FakePort:
    def __init__(self, owner, role="agent"):
        self.owner = owner
        self.role = role
        self.inbox = []
        self.closed = None
        self.closing = False

    def deliver(self, message):
        self.inbox.append(message)

    def close(self, code, reason):
        self.closed = (code, reason)
        self.closing = True

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
    assert runner.spawn_log == [(0, 1, "alice", 1, "fly")]
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


def test_reattach_does_not_charge_a_missed_tick_for_the_silence():
    runner = make_runner()
    port = joined_agent(runner)
    runner.detach(port)
    runner.tick()
    back = FakePort("alice")
    runner.attach(back)
    runner.request_join(back)
    runner.tick()
    assert back.last("observation").missed == 0


def test_reattach_does_not_replay_an_action_from_before_the_disconnect():
    runner = make_runner()
    port = joined_agent(runner)
    assert runner.submit_action(port, tick=0, action=int(Action.MOVE_E))
    x_before = runner.world.state.agents[1].x
    runner.detach(port)  # the action never reached a tick
    runner.tick()
    back = FakePort("alice")
    runner.attach(back)
    runner.request_join(back)
    runner.tick()
    assert runner.world.state.agents[1].x == x_before


def test_leave_then_join_reattaches_the_same_port_and_notes_it_once():
    runner = make_runner()
    port = joined_agent(runner)
    runner.detach(port)  # leave
    runner.request_join(port)
    runner.tick()
    assert len(port.of("joined")) == 2 and port.last("joined").reattached
    notes = [text for _, text in runner.chronicle if "back at the controls" in text]
    assert len(notes) == 1


def test_a_leave_cancels_a_join_queued_in_the_same_tick():
    runner = make_runner()
    port = FakePort("alice")
    runner.attach(port)
    runner.request_join(port)
    runner.detach(port)  # the client changed its mind before the world advanced
    runner.tick()
    assert port.of("joined") == [] and runner.roster.live_agent("alice") is None
    assert runner.world.state.agents == {}


def test_a_superseded_port_cannot_take_the_fly_back_with_a_queued_join():
    runner = make_runner()
    old = joined_agent(runner)
    new = FakePort("alice")
    runner.attach(new)
    runner.request_join(old)  # in flight when the newer connection arrived
    runner.request_join(new)
    runner.tick()
    assert runner.roster.state("alice").port is new
    assert len(old.of("joined")) == 1  # only the first life, nothing new
    assert new.last("joined").reattached and new.of("observation")


def test_only_one_join_stays_queued_per_port():
    runner = make_runner()
    port = FakePort("alice")
    runner.attach(port)
    for _ in range(50):
        runner.request_join(port)
    assert len(runner._queued_joins) == 1
    runner.tick()
    assert len(port.of("joined")) == 1


def test_an_owner_who_never_hatched_scores_nothing_and_is_forgotten_on_detach():
    runner = make_runner()
    watcher = FakePort("w", role="spectator")
    runner.add_spectator(watcher)
    curious = FakePort("curious")
    runner.attach(curious)
    runner.tick()
    assert [s.owner for s in watcher.last("frame").scores] == []
    runner.detach(curious)
    assert "curious" not in runner.roster.owners


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
    assert [(r.x, r.y, r.kind) for r in frame.resources] == [(3, 1, 1)]  # the map's F tile
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


def test_a_speech_bubble_fades_and_an_empty_say_clears_it_at_once():
    runner = make_runner()
    port = joined_agent(runner)
    watcher = FakePort("w", role="spectator")
    runner.add_spectator(watcher)
    runner.say(port, "fruit?")
    runner.tick()
    runner.say(port, "")
    runner.tick()
    assert watcher.last("frame").agents[0].say == ""
    runner.say(port, "still here")
    runner.tick()
    assert watcher.last("frame").agents[0].say == "still here"
    for _ in range(SAY_TTL):
        runner.tick()
    assert watcher.last("frame").agents[0].say == ""


def test_a_dead_fly_shows_in_the_frame_of_its_last_tick_and_never_again():
    runner = make_runner(config=Config(initial_satiety=2, initial_health=5))
    port = joined_agent(runner)
    watcher = FakePort("w", role="spectator")
    runner.add_spectator(watcher)
    runner.tick()  # the fly starves
    corpse = watcher.last("frame").agents[0]
    assert (corpse.alive, corpse.lineage, corpse.name) == (False, 1, port.last("joined").name)
    runner.request_join(port)
    runner.tick()
    listed = watcher.last("frame").agents
    assert [(a.agent_id, a.lineage) for a in listed] == [(2, 2)]  # only the new life


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


class DeadConnection:
    """A socket that has already gone away: every send and close raises."""

    def __init__(self):
        self.closed = False

    async def send(self, text):
        raise ConnectionClosedError(None, None)

    async def close(self, code, reason):
        self.closed = (code, reason)


def test_a_pump_whose_socket_died_finishes_without_an_unretrieved_exception():
    async def scenario():
        port = RemotePort(DeadConnection(), "alice", "agent")
        pump = asyncio.create_task(port.pump())
        port.deliver(frames.chronicle_message(0, "the cable was yanked"))
        await asyncio.wait_for(pump, timeout=1)
        return pump.exception()

    assert asyncio.run(scenario()) is None


def test_closing_a_port_truncates_the_reason_and_leaves_no_pending_task():
    async def scenario():
        connection = DeadConnection()
        port = RemotePort(connection, "alice", "agent")
        port.close(4004, "n" * 300)
        await port.shutdown()
        return connection.closed, port._closer

    closed, closer = asyncio.run(scenario())
    assert closed == (4004, "n" * 100) and closer is None


class BrokenBrain:
    def reset(self, seed=None):
        pass

    def act(self, observation):
        raise RuntimeError("no idea what to do")


def test_a_hosted_brain_that_raises_idles_and_keeps_its_fly(caplog):
    runner = make_runner()
    port = LocalPort(BrokenBrain(), "npc-broken-1", runner, runner.catalog.bodies["fly"])
    runner.attach(port)
    runner.request_join(port)
    runner.tick()
    runner.tick()
    assert runner.roster.live_agent("npc-broken-1") == 1
    assert runner.action_log[-1] == {1: int(Action.IDLE)}
    assert runner.missed[1] == 1
    assert "failed to act" in caplog.text


def test_a_join_that_cannot_hatch_answers_world_full_without_stopping_the_tick():
    runner = make_runner(map=CRAMPED)
    alice, bob = FakePort("alice"), FakePort("bob")
    runner.attach(alice)
    runner.attach(bob)
    runner.request_join(alice)
    runner.request_join(bob)
    runner.tick()
    assert alice.of("joined") and not bob.of("joined")
    refusal = bob.last("error")
    assert (refusal.code, refusal.fatal) == ("world_full", False)
    assert runner.roster.live_agent("bob") is None
    assert len(runner.action_log) == 1  # the tick finished


def test_a_world_that_arrives_with_flies_the_runner_never_hatched_keeps_ticking():
    config = Config(initial_satiety=2, initial_health=5)
    world = World.from_map(TINY, config, seed=1)
    stranger = world.spawn()
    runner = WorldRunner(world, tps=10.0, map_name="tiny")
    watcher = FakePort("w", role="spectator")
    runner.add_spectator(watcher)
    port = joined_agent(runner)
    assert [a.agent_id for a in watcher.last("frame").agents] == [2]  # the stranger is not ours
    for _ in range(3):
        runner.tick()  # the stranger starves: no roster entry, no observation, no KeyError
    assert not world.state.agents[stranger].alive
    assert runner.action_log[0][stranger] == int(Action.IDLE)
    assert not any(m.payload.agent_id == stranger for m in port.inbox if m.type == "died")


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
    assert 3 <= runner.world.tick <= 20  # a loaded machine ticks slower, never faster


def test_observations_from_a_running_world_carry_the_time_left_in_the_tick():
    runner = make_runner(tps=20.0)
    port = FakePort("alice")
    runner.attach(port)
    runner.request_join(port)

    async def run_briefly():
        stop = asyncio.Event()
        task = asyncio.create_task(runner.run(stop))
        await asyncio.sleep(0.2)
        stop.set()
        await task

    asyncio.run(run_briefly())
    deadlines = [observation.deadline_ms for observation in port.of("observation")]
    assert len(deadlines) >= 2
    assert all(0 < deadline <= 50 for deadline in deadlines), deadlines


def test_a_tick_that_raises_stops_the_world_instead_of_freezing_it(caplog):
    runner = make_runner(tps=100.0)
    ticks = []

    def exploding_tick():
        ticks.append(1)
        if len(ticks) == 3:
            raise RuntimeError("the sky fell")

    runner.tick = exploding_tick

    async def run_until_it_gives_up():
        stop = asyncio.Event()
        await asyncio.wait_for(runner.run(stop), timeout=2)
        return stop.is_set()

    assert asyncio.run(run_until_it_gives_up())
    assert len(ticks) == 3 and "the sky fell" in caplog.text


def test_tick_order_is_deterministic_from_the_logs():
    runner = make_runner(map=maps.load("drosoville"))
    alice, bob = joined_agent(runner, "alice"), joined_agent(runner, "bob")
    for step in range(30):
        runner.submit_action(alice, alice.last("observation").tick, (step % 4) + 1)
        runner.submit_action(bob, bob.last("observation").tick, int(Action.REST))
        runner.tick()
    replay = World.from_map(maps.load("drosoville"), Config(), seed=1)
    for tick, actions in enumerate(runner.action_log):
        for spawn_tick, agent_id, *_ in runner.spawn_log:
            if spawn_tick == tick:
                assert replay.spawn() == agent_id
        replay.step(actions)
    assert replay.state_hash() == runner.world.state_hash()
