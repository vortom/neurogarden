"""Integration: a real server on an ephemeral port, real SDK clients, real sockets."""

import asyncio
import contextlib
import gc
import json
import logging
import urllib.request

import pytest
from websockets.asyncio.client import connect

from neurogarden.brains import RandomBrain, ScriptedBrain
from neurogarden.engine import Config, World, maps
from neurogarden.protocol import build_catalog
from neurogarden.sdk import AsyncClient, ConnectionLost, ServerError, Session, run_brain
from neurogarden.server import Server, ServerConfig
from neurogarden.server import frames as server_frames

FAST = dict(port=0, tps=50.0, npcs=[], hello_timeout=0.5)
JOIN_FRAME = '{"v": 1, "type": "join", "payload": {}}'


def run(coro):
    return asyncio.run(coro)


def hello_frame(owner, role="agent", token="dev"):
    payload = {"protocol": 1, "token": token, "owner": owner, "role": role}
    return json.dumps({"v": 1, "type": "hello", "payload": payload})


def test_the_welcome_describes_the_shape_of_a_day():
    """A client names dawn, day, dusk and night from these, so the world has to say them."""
    config = Config(day_length=400, dawn_end=40, dusk_start=300, night_start=340)
    world = World.from_map(maps.load("drosoville"), config, seed=0)
    message = server_frames.welcome_message(
        "alice", "spectator", world, "drosoville", 5.0, build_catalog(), "hello"
    )
    day = message.payload.world
    assert (day.day_length, day.dawn_end, day.dusk_start, day.night_start) == (400, 40, 300, 340)


def test_two_brains_and_an_npc_share_the_world_and_the_spectator_sees_them_all():
    async def scenario():
        async with Server(ServerConfig(**{**FAST, "npcs": [("scripted", 1)]})) as server:
            async with (
                AsyncClient(server.url, owner="alice") as alice,
                AsyncClient(server.url, owner="bob") as bob,
                AsyncClient(server.url, owner="w", role="spectator") as watcher,
            ):
                assert alice.welcome.world.name == "drosoville" and alice.welcome.motd
                a_joined, b_joined = await asyncio.gather(alice.join(), bob.join())
                assert {a_joined.agent_id, b_joined.agent_id} & {2, 3}
                random_brain, scripted = RandomBrain(seed=1), ScriptedBrain()
                seen, worst_missed, acted = 0, 0, 0
                async for observation in alice.observations():
                    await alice.act(observation.tick, random_brain.act(observation.channels))
                    acted += 1
                    seen += 1
                    worst_missed = max(worst_missed, observation.missed)
                    if seen == 20:
                        break
                # a brain answering within the tick is mostly not "missed"; a loaded
                # machine may drop a few, but not one in four
                assert seen == 20 and acted == 20 and worst_missed <= 5
                observation = await bob.next_observation()
                await bob.act(observation.tick, scripted.act(observation.channels))
                await bob.say("hello garden")
                frame = None
                for _ in range(10):
                    frame = await watcher.next_frame()
                    if any(a.say for a in frame.agents):
                        break
                owners = {a.owner for a in frame.agents}
                assert owners == {"alice", "bob", "npc-scripted-1"}
                assert {a.mood for a in frame.agents} <= {"content", "hungry", "thirsty", "sleepy"}
                assert next(a for a in frame.agents if a.owner == "bob").say == "hello garden"
                assert watcher.spectacle.world is not None
                assert any("hatches" in text for _, text in watcher.spectacle.chronicle)

    run(scenario())


def test_disconnect_idles_the_fly_and_a_reconnect_reattaches_it():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            async with AsyncClient(server.url, owner="alice") as first:
                joined = await first.join()
                await first.next_observation()
            await asyncio.sleep(0.1)
            async with AsyncClient(server.url, owner="alice") as again:
                back = await again.join()
                assert back.reattached and back.agent_id == joined.agent_id
                observation = await again.next_observation()
                assert observation.tick > 2
            async with AsyncClient(server.url, owner="alice") as third:
                await third.join()
                await third.leave()
                fourth = await third.join()
                assert fourth.reattached
            back = [t for _, t in server.runner.chronicle if "back at the controls" in t]
            assert len(back) == 3  # two reconnects and one rejoin after leave, noted once each
            return server.runner.roster.state("alice").lineage

    assert run(scenario()) == 1


def test_a_newer_connection_supersedes_the_older_one():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            old = await AsyncClient(server.url, owner="alice").__aenter__()
            await old.join()
            async with AsyncClient(server.url, owner="alice") as new:
                joined = await new.join()
                assert joined.reattached
                with pytest.raises(ConnectionLost) as lost:
                    for _ in range(5):
                        await old.next_observation()
                assert lost.value.code == 4004
            await old.close()

    run(scenario())


def test_a_superseded_connection_cannot_take_the_fly_back_by_joining():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            async with connect(server.url) as stale:
                await stale.send(hello_frame("alice"))
                await stale.recv()  # welcome
                await stale.send(JOIN_FRAME)
                async with AsyncClient(server.url, owner="alice") as fresh:
                    joined = await fresh.join()
                    for _ in range(10):  # the stale connection keeps asking for the fly
                        with contextlib.suppress(Exception):
                            await stale.send(JOIN_FRAME)
                        await asyncio.sleep(0.005)
                    seen = [await fresh.next_observation() for _ in range(5)]
                    assert all(observation is not None for observation in seen)
                    assert server.runner.roster.live_agent("alice") == joined.agent_id

    run(scenario())


def test_a_hosted_npc_owner_cannot_be_taken_over_by_a_remote_client():
    async def scenario():
        async with Server(ServerConfig(**{**FAST, "npcs": [("scripted", 1)]})) as server:
            with pytest.raises(ServerError) as err:
                async with AsyncClient(server.url, owner="npc-scripted-1"):
                    pass
            assert err.value.code == "unauthorized"
            async with AsyncClient(server.url, owner="w", role="spectator") as watcher:
                frame = await watcher.next_frame()
                npc = next(a for a in frame.agents if a.owner == "npc-scripted-1")
                assert npc.connected  # the hosted brain still has its fly

    run(scenario())


def test_an_action_before_joining_answers_no_fly_without_closing():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            async with connect(server.url) as ws:
                await ws.send(hello_frame("alice"))
                await ws.recv()  # welcome
                await ws.send('{"v": 1, "type": "action", "payload": {"tick": 0, "action": 1}}')
                return json.loads(await ws.recv())

    message = run(scenario())
    assert message["type"] == "error"
    assert (message["payload"]["code"], message["payload"]["fatal"]) == ("no_fly", False)


def test_a_close_reason_too_long_for_a_frame_is_truncated():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            async with connect(server.url) as ws:
                await ws.send(json.dumps({"v": "x" * 300, "type": "hello", "payload": {}}))
                error = json.loads(await ws.recv())
                try:
                    await ws.recv()
                except Exception as closed:  # noqa: BLE001 - the close is what we test
                    return error, closed.rcvd.code, closed.rcvd.reason

    error, code, reason = run(scenario())
    assert code == 4001 and len(reason.encode()) <= 123
    assert len(error["payload"]["message"]) > 123  # the whole story stays in the payload


def test_an_abrupt_disconnect_leaves_no_task_exception_behind(caplog):
    caplog.set_level(logging.ERROR, logger="asyncio")

    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            client = await AsyncClient(server.url, owner="alice").__aenter__()
            await client.join()
            await client.next_observation()
            client._connection.transport.abort()  # yanked cable, no close handshake
            await asyncio.sleep(0.2)

    run(scenario())
    gc.collect()
    assert not [r for r in caplog.records if "never retrieved" in r.getMessage()]


def test_a_failing_tick_stops_the_world_and_still_closes_the_listener(caplog):
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            real_tick, ticks = server.runner.tick, []

            def exploding_tick():
                ticks.append(1)
                if len(ticks) == 3:
                    raise RuntimeError("the sky fell")
                real_tick()

            server.runner.tick = exploding_tick
            await asyncio.wait_for(server.stop.wait(), timeout=2)
            url = server.url
        with pytest.raises(ConnectionLost):  # the listener went down with the world
            await AsyncClient(url, owner="late").__aenter__()

    run(scenario())
    assert "the sky fell" in caplog.text


def test_a_slow_spectator_gets_the_latest_frame_not_a_backlog():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            async with AsyncClient(server.url, owner="w", role="spectator") as watcher:
                await watcher.next_frame()
                await asyncio.sleep(0.2)  # ~10 frames pile up at 50 tps
                frame = await watcher.next_frame()
                return frame.tick, server.runner.world.tick

    shown, now = run(scenario())
    assert now - shown <= 2  # latest-wins over the wire, not a queue of stale frames


def test_death_then_rejoin_gives_the_next_lineage_and_the_stats():
    config = Config(initial_satiety=3, initial_health=5)

    async def scenario():
        async with Server(ServerConfig(**FAST, config=config)) as server:
            async with AsyncClient(server.url, owner="alice") as alice:
                first = await alice.join()
                async for _ in alice.observations():
                    pass
                assert alice.last_died.lineage == 1 and alice.last_died.causes == ["starvation"]
                assert alice.last_died.stats.lifespan == 3
                second = await alice.join()
                assert (second.lineage, second.reattached) == (2, False)
                assert second.name != first.name
                return server.runner.roster.state("alice").best_lifespan

    assert run(scenario()) == 3


def test_the_sync_session_and_run_brain_work_from_a_plain_thread():
    config = Config(initial_satiety=3, initial_health=5)
    lives = []

    async def scenario():
        # 20 tps: a blocking round trip per tick has room even on a busy machine
        async with Server(ServerConfig(**{**FAST, "tps": 20.0}, config=config)) as server:
            stats = await asyncio.to_thread(
                run_brain,
                server.url,
                ScriptedBrain(),
                owner="alice",
                lives=2,
                on_life=lambda fly: lives.append((fly.name, fly.lineage)),
            )
            with_session = await asyncio.to_thread(session_scenario, server.url)
            return stats, with_session

    def session_scenario(url):
        with Session(url, owner="carol") as session:
            fly = session.join()
            steps = 0
            for observation in fly.observations():
                fly.act(observation.tick, 6)
                steps += 1
                if steps == 3:
                    fly.say("zzz")
            return fly.lineage, fly.stats.death_causes, steps

    stats, with_session = run(scenario())
    assert [s.lifespan for s in stats] == [3, 3]
    assert [lineage for _, lineage in lives] == [1, 2]
    assert with_session[0] == 1 and with_session[1] == ("starvation",) and with_session[2] >= 1


@pytest.mark.parametrize(
    "frame, code",
    [
        ('{"v": 2, "type": "hello", "payload": {}}', 4001),
        ('{"v": 1, "type": "join", "payload": {}}', 4003),
        ("nonsense", 4000),
        (
            '{"v": 1, "type": "hello", "payload": {"protocol": 1, "token": "wrong", '
            '"owner": "x", "role": "agent"}}',
            4002,
        ),
        (
            '{"v": 1, "type": "hello", "payload": {"protocol": 9, "token": "dev", '
            '"owner": "x", "role": "agent"}}',
            4001,
        ),
    ],
)
def test_bad_handshakes_close_with_the_documented_codes(frame, code):
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            async with connect(server.url) as ws:
                await ws.send(frame)
                messages = []
                try:
                    while True:
                        messages.append(json.loads(await ws.recv()))
                except Exception as closed:  # noqa: BLE001 - the close code is what we test
                    return messages, getattr(getattr(closed, "rcvd", None), "code", None)

    messages, close_code = run(scenario())
    assert close_code == code
    if code != 4003:
        assert messages[-1]["type"] == "error" and messages[-1]["payload"]["fatal"]


def test_silence_after_connecting_is_closed_with_hello_required():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            async with connect(server.url) as ws:
                try:
                    await ws.recv()
                except Exception as closed:  # noqa: BLE001
                    return closed.rcvd.code

    assert run(scenario()) == 4003


def test_wrong_token_raises_server_error_in_the_sdk_and_leaves_no_socket_open():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            client = AsyncClient(server.url, owner="alice", token="nope")
            with pytest.raises(ServerError) as err:
                async with client:
                    pass
            assert err.value.code == "unauthorized"
            assert client._connection is None

    run(scenario())


def test_closing_the_client_wakes_whoever_waits_for_an_observation():
    async def scenario():
        async with Server(ServerConfig(**{**FAST, "tps": 2.0})) as server:
            client = await AsyncClient(server.url, owner="alice").__aenter__()
            await client.join()
            await client.next_observation()  # the birth tick; the next is half a second away
            waiting = asyncio.create_task(client.next_observation())
            await asyncio.sleep(0.05)
            await client.close()
            with pytest.raises(ConnectionLost):
                await asyncio.wait_for(waiting, timeout=1)

    run(scenario())


def test_a_spectator_keeps_only_the_recent_chronicle():
    from neurogarden.sdk.client import CHRONICLE_MAX, Spectacle

    spectacle = Spectacle()
    for tick in range(CHRONICLE_MAX + 50):
        spectacle.chronicle.append((tick, f"line {tick}"))
    assert len(spectacle.chronicle) == CHRONICLE_MAX
    assert spectacle.chronicle[-1] == (CHRONICLE_MAX + 49, f"line {CHRONICLE_MAX + 49}")


def test_a_session_closed_before_it_was_entered_does_not_block():
    session = Session("ws://127.0.0.1:1", owner="alice")
    session.close()  # no thread was ever started: nothing to wait for
    session.close()  # and closing twice is harmless


def test_health_endpoint_answers_plain_http():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            url = f"http://127.0.0.1:{server.port}/healthz"
            return await asyncio.to_thread(lambda: urllib.request.urlopen(url).read())

    assert run(scenario()) == b"OK\n"


@pytest.mark.parametrize("host", ["0.0.0.0", "", "::", "192.168.1.10"])
def test_a_bind_beyond_loopback_with_the_default_token_is_refused(host):
    with pytest.raises(ValueError):
        Server(ServerConfig(host=host))


def test_server_actions_replay_through_the_engine_alone():
    async def scenario():
        settings = {**FAST, "seed": 11, "npcs": [("random", 1)]}
        async with Server(ServerConfig(**settings)) as server:
            async with AsyncClient(server.url, owner="alice") as alice:
                await alice.join()
                brain = ScriptedBrain()
                seen = 0
                async for observation in alice.observations():
                    await alice.act(observation.tick, brain.act(observation.channels))
                    seen += 1
                    if seen == 40:
                        break
            server.stop.set()
            await server._ticker
            runner = server.runner
            return runner.spawn_log, list(runner.action_log), runner.world.state_hash()

    spawn_log, action_log, expected = run(scenario())
    replay = World.from_map(maps.load("drosoville"), Config(), seed=11)
    for tick, actions in enumerate(action_log):
        for spawn_tick, agent_id, *_ in spawn_log:  # spawns of this tick, in join order
            if spawn_tick == tick:
                assert replay.spawn() == agent_id
        replay.step(actions)
    assert replay.state_hash() == expected
