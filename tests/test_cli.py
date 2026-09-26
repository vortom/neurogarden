import asyncio
import json

import pytest

from neurogarden import cli
from neurogarden.protocol import schema_text
from neurogarden.server import Server, ServerConfig, parse_npc


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
    config = ServerConfig(port=0, tps=50.0, npcs=[("scripted", 1)], hello_timeout=1.0)

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
                    "0",
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
