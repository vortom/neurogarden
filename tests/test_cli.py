import asyncio
import json

import pytest

from neurogarden import cli
from neurogarden.protocol import schema_text
from neurogarden.server import Server, ServerConfig, parse_npc, serve


def test_schema_command_prints_the_artifact(capsys):
    assert cli.main(["schema"]) == 0
    out = capsys.readouterr().out
    assert out == schema_text()
    assert json.loads(out)["title"].startswith("NeuroGarden protocol")


def test_parse_npc():
    assert parse_npc("scripted:2") == ("scripted", 2)
    assert parse_npc("random") == ("random", 1)
    assert parse_npc("none") == ("none", 0)
    with pytest.raises(ValueError):
        parse_npc("oracle:1")
    with pytest.raises(ValueError):
        parse_npc("random:-1")


def test_serve_parser_defaults_and_npc_flag():
    args = cli.build_parser().parse_args(["serve", "--npc", "scripted:2", "--npc", "random"])
    assert args.tps == 5.0 and args.host == "127.0.0.1" and args.npc == ["scripted:2", "random"]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["join"])  # --owner is required


def test_join_and_watch_against_a_live_server(capsys):
    from neurogarden.engine import Config

    config = ServerConfig(
        port=0,
        tps=50.0,
        npcs=[("scripted", 1)],
        hello_timeout=1.0,
        config=Config(initial_satiety=3, initial_health=5),  # a short life, so join returns
    )

    async def scenario():
        async with Server(config) as server:
            join = asyncio.to_thread(
                cli.main,
                [
                    "join",
                    "--brain",
                    "random",
                    "--owner",
                    "alice",
                    "--url",
                    server.url,
                    "--lives",
                    "1",
                ],
            )
            watch = asyncio.to_thread(
                cli.main,
                ["watch", "--url", server.url, "--frames", "3", "--ascii", "--follow", "alice"],
            )
            return await asyncio.gather(join, watch)

    assert asyncio.run(scenario()) == [0, 0]
    out = capsys.readouterr().out
    assert "Day 1" in out and "best:" in out and "npc-scripted-1" in out
    assert "alice's" in out and "starvation" in out  # the brain really flew a fly


def test_join_with_the_wrong_token_fails_with_one_line(capsys):
    config = ServerConfig(port=0, tps=50.0, npcs=[], hello_timeout=1.0, token="secret")

    async def scenario():
        async with Server(config) as server:
            argv = ["join", "--owner", "bob", "--url", server.url, "--token", "guess"]
            return await asyncio.to_thread(cli.main, argv)

    assert asyncio.run(scenario()) == 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == ["neurogarden: unauthorized: bad token"]


def test_watch_without_a_server_fails_with_one_line(capsys):
    assert cli.main(["watch", "--url", "ws://127.0.0.1:1", "--frames", "1"]) == 1
    assert capsys.readouterr().err.startswith("neurogarden: connection closed")


def test_serve_with_an_unknown_npc_fails_before_binding(capsys):
    assert cli.main(["serve", "--npc", "oracle:2", "--port", "0"]) == 1
    assert "unknown brain 'oracle'" in capsys.readouterr().err


def test_serve_announces_the_port_it_really_bound(capsys):
    async def scenario():
        bound = []

        def announce(server):
            bound.append(server.port)
            cli.banner(server)
            server.stop.set()  # one banner is all this test wants

        config = ServerConfig(port=0, tps=50.0, npcs=[])
        await asyncio.wait_for(serve(config, on_ready=announce), timeout=5)
        return bound[0]

    port = asyncio.run(scenario())
    out = capsys.readouterr().out
    assert port != 0 and f"on ws://127.0.0.1:{port}" in out
    assert f"--url ws://127.0.0.1:{port}" in out


def test_join_reports_each_life(capsys):
    from neurogarden.engine import Config

    config = ServerConfig(
        port=0,
        tps=50.0,
        npcs=[],
        hello_timeout=1.0,
        config=Config(initial_satiety=3, initial_health=5),
    )

    async def scenario():
        async with Server(config) as server:
            argv = [
                "join",
                "--brain",
                "scripted",
                "--owner",
                "bob",
                "--url",
                server.url,
                "--lives",
                "2",
            ]
            return await asyncio.to_thread(cli.main, argv)

    assert asyncio.run(scenario()) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("bob's")]
    assert len(lines) == 2 and "(#2)" in lines[1] and "starvation" in lines[1]


def test_join_without_a_server_fails_cleanly(capsys):
    assert cli.main(["join", "--owner", "x", "--url", "ws://127.0.0.1:1", "--lives", "1"]) == 1
