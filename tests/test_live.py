"""Integration: a real server on an ephemeral port, real SDK clients, real sockets."""

import asyncio
import json
import urllib.request

import pytest
from websockets.asyncio.client import connect

from neurogarden.brains import RandomBrain, ScriptedBrain
from neurogarden.engine import Config, World, maps
from neurogarden.sdk import AsyncClient, ConnectionLost, ServerError, Session, run_brain
from neurogarden.server import Server, ServerConfig

FAST = dict(port=0, tps=50.0, npcs=[], hello_timeout=0.5)


def run(coro):
    return asyncio.run(coro)


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
                seen, worst_missed = 0, 0
                async for observation in alice.observations():
                    await alice.act(observation.tick, random_brain.act(observation.channels))
                    seen += 1
                    worst_missed = max(worst_missed, observation.missed)
                    if seen == 20:
                        break
                assert worst_missed <= 3  # a brain answering within the tick is not "missed"
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
        async with Server(ServerConfig(**FAST, config=config)) as server:
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
    assert with_session[0] == 1 and with_session[1] == ("starvation",) and with_session[2] >= 3


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


def test_wrong_token_raises_server_error_in_the_sdk():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            with pytest.raises(ServerError) as err:
                async with AsyncClient(server.url, owner="alice", token="nope"):
                    pass
            assert err.value.code == "unauthorized"

    run(scenario())


def test_health_endpoint_answers_plain_http():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            url = f"http://127.0.0.1:{server.port}/healthz"
            return await asyncio.to_thread(lambda: urllib.request.urlopen(url).read())

    assert run(scenario()) == b"OK\n"


def test_non_loopback_bind_with_the_default_token_is_refused():
    with pytest.raises(ValueError):
        Server(ServerConfig(host="0.0.0.0"))


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
        for spawn_tick, agent_id, _, _ in spawn_log:  # spawns of this tick, in join order
            if spawn_tick == tick:
                assert replay.spawn() == agent_id
        replay.step(actions)
    assert replay.state_hash() == expected
