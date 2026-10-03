import asyncio
import json
import re

import pytest

from neurogarden import cli
from neurogarden.protocol import schema_text
from neurogarden.server import Server, ServerConfig, app, parse_npc, serve


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
    assert "play:  open http://127.0.0.1:" in out  # the committed bundle is there to play


def test_the_banner_offers_no_page_when_no_bundle_was_built(capsys, monkeypatch):
    monkeypatch.setattr(cli, "bundle_present", lambda: False)

    class FakeServer:
        config = ServerConfig(map="drosoville", seed=0, port=8765, web=True)
        port = 8765
        url = "ws://127.0.0.1:8765"
        resumed = False

    cli.banner(FakeServer())
    out = capsys.readouterr().out
    assert "watch:" in out and "play:" not in out


def test_the_server_warns_when_it_is_told_to_serve_a_page_it_does_not_have(caplog, monkeypatch):
    monkeypatch.setattr(app, "bundle_present", lambda: False)

    async def scenario():
        async with Server(ServerConfig(port=0, tps=50.0, npcs=[])):
            pass

    asyncio.run(scenario())
    assert "npm --prefix web run build" in caplog.text


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


def test_serve_refuses_a_port_that_is_already_taken(capsys):
    import socket

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        argv = ["serve", "--port", str(port), "--npc", "none", "--archive", ":memory:"]
        assert cli.main(argv) == 1
    assert "address already in use" in capsys.readouterr().err


# --- the archive commands ---------------------------------------------------------------------


@pytest.fixture
def garden(tmp_path):
    """An archive with one short, finished life of alice's, and the npc still going."""
    import numpy as np  # noqa: F401  (the export test needs numpy; keep the import local)

    from neurogarden.engine import Config

    path = str(tmp_path / "garden.db")
    config = ServerConfig(
        port=0,
        tps=50.0,
        npcs=[("scripted", 1)],
        hello_timeout=1.0,
        seed=3,
        config=Config(initial_satiety=8, initial_health=5),
        archive=path,
    )

    async def scenario():
        async with Server(config) as server:
            argv = ["join", "--brain", "random", "--owner", "alice", "--url", server.url]
            await asyncio.to_thread(cli.main, [*argv, "--lives", "1"])
        return server.runner.world.tick

    ticks = asyncio.run(scenario())
    return path, ticks


def test_lives_prints_the_hall_of_flies(garden, capsys):
    path, _ = garden
    assert cli.main(["lives", "--archive", path]) == 0
    out = capsys.readouterr().out
    lives = int(re.search(r"(\d+) lives", out).group(1))
    assert "drosoville (seed 3)" in out and lives >= 2  # the npc may have rejoined a few times
    assert "alice" in out and "npc-scripted-1" in out and "starvation / 0 / 0" in out
    assert cli.main(["lives", "--archive", path, "--owner", "alice"]) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("alice")]
    assert len(rows) == 1 and " 1 " in rows[0]


def test_lives_marks_a_fly_that_is_still_going(tmp_path, capsys):
    from neurogarden.engine import World, maps
    from neurogarden.server import Archive, WorldRunner
    from test_runner import joined_agent

    path = str(tmp_path / "live.db")
    runner = WorldRunner(World.from_map(maps.load("drosoville")), archive=Archive.open(path))
    joined_agent(runner, "carol")
    for _ in range(4):
        runner.tick()
    runner.archive.close()
    assert cli.main(["lives", "--archive", path]) == 0
    out = capsys.readouterr().out
    assert "1 lives" in out and "still alive" in out and "5*" in out


def test_replay_shows_a_life_frame_by_frame(garden, capsys):
    path, _ = garden
    argv = ["replay", "--archive", path, "--owner", "alice", "--life", "1", "--ascii"]
    assert cli.main([*argv, "--frames", "3", "--fps", "1000"]) == 0
    out = capsys.readouterr().out
    assert out.count("ghost: alice's") == 3
    shown = [(int(tick), int(born)) for tick, born in re.findall(r"tick (\d+) of (\d+)–", out)]
    born = shown[0][1]  # whenever she hatched: a busy machine joins her a tick or two late
    assert shown == [(born, born), (born + 1, born), (born + 2, born)]
    assert cli.main([*argv, "--fps", "1000"]) == 0  # the whole life, to its last frame
    assert "dies of starvation" in capsys.readouterr().out
    assert cli.main(["replay", "--archive", path, "--owner", "alice", "--life", "7"]) == 1
    assert "no life #7" in capsys.readouterr().err


def test_export_writes_observation_action_pairs(garden, tmp_path, capsys):
    import numpy as np

    path, _ = garden
    out = str(tmp_path / "alice-1.npz")
    argv = ["export", "--archive", path, "--owner", "alice", "--life", "1", "--out", out]
    assert cli.main(argv) == 0
    assert "wrote" in capsys.readouterr().out
    with np.load(out) as data:
        meta = json.loads(str(data["meta"]))
        steps = len(data["actions"])
        assert meta["owner"] == "alice" and meta["causes"] == ["starvation"]
        lifespan = meta["died_tick"] - meta["born_tick"] + 1
        assert steps == lifespan - 1  # the hatching idle and the final observation are left out
        assert data["ticks"][0] == meta["born_tick"] and data["ticks"][-1] == meta["died_tick"] - 1
        assert data["vision"].shape[0] == steps and data["body"].shape == (steps, 5)
        assert data["body"][0][4] == 1  # age 1 after the hatching tick
        assert list(meta["catalog"]["actions"])[0] == "idle"


def test_verify_checks_the_whole_history(garden, capsys):
    path, _ = garden
    assert cli.main(["verify", "--archive", path]) == 0
    assert capsys.readouterr().out.startswith("ok:")
    assert cli.main(["verify", "--archive", path + ".missing"]) == 1
    assert "no archive" in capsys.readouterr().err


def test_serve_resumes_an_archive_and_refuses_another_seed(garden, capsys, monkeypatch):
    path, ticks = garden
    seen = []
    archive = cli._open_archive(path)
    lives = len(archive.lives())  # alice's, and as many as the npc got through meanwhile
    archive.close()

    def announce(server):
        cli.banner(server)
        seen.append((server.resumed, server.runner.world.tick))
        server.stop.set()  # one banner is all this test wants

    async def stopping_serve(config, on_ready=None):
        await serve(config, on_ready=announce)

    monkeypatch.setattr(cli, "serve", stopping_serve)
    # no --map/--seed: the CLI takes them from the archive and resumes
    assert cli.main(["serve", "--archive", path, "--port", "0", "--npc", "none"]) == 0
    assert cli.main(["serve", "--archive", path, "--port", "0", "--seed", "9"]) == 1
    out, err = capsys.readouterr()
    assert seen == [(True, ticks)]
    assert "drosoville (seed 3)" in out and f"resumed at tick {ticks}" in out
    assert f"{lives} lives so far" in out and f"archive: {path}" in out
    assert "seed 3), not drosoville (seed 9)" in err


def test_replay_refuses_a_non_positive_fps(garden, capsys):
    path, _ = garden
    argv = ["replay", "--archive", path, "--owner", "alice", "--life", "1", "--fps", "0"]
    assert cli.main(argv) == 1
    assert "fps must be positive" in capsys.readouterr().err


def test_archive_commands_refuse_files_that_are_not_archives(tmp_path, capsys):
    import sqlite3

    notes = tmp_path / "notes.db"
    notes.write_text("not sqlite at all")
    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()
    for path in (notes, empty, tmp_path):
        assert cli.main(["lives", "--archive", str(path)]) == 1
    errors = capsys.readouterr().err.splitlines()
    assert len(errors) == 3 and all(line.startswith("neurogarden: ") for line in errors)
    assert "not a NeuroGarden archive" in errors[0] and "holds no world" in errors[1]
    assert (
        cli.main(["serve", "--archive", str(tmp_path / "no" / "dir" / "g.db"), "--port", "0"]) == 1
    )
    assert "cannot open" in capsys.readouterr().err


def test_export_refuses_an_unwritable_output_path(garden, tmp_path, capsys):
    path, _ = garden
    out = str(tmp_path / "no" / "such" / "dir" / "life.npz")
    argv = ["export", "--archive", path, "--owner", "alice", "--life", "1", "--out", out]
    assert cli.main(argv) == 1
    assert capsys.readouterr().err.startswith("neurogarden: ")


# --- the evolved brain -----------------------------------------------------------------------


def test_evolve_writes_weights_that_join_can_fly(tmp_path, capsys):
    out = str(tmp_path / "tiny.npz")
    argv = [
        "evolve", "--out", out, "--generations", "1", "--population", "2", "--episodes", "1",
        "--max-steps", "20", "--hidden", "3", "--workers", "1", "--seed", "4",
    ]  # fmt: skip
    assert cli.main(argv) == 0
    printed = capsys.readouterr().out
    assert "evolving a 3-neuron brain" in printed and "gen   1/1" in printed and "wrote" in printed
    again = str(tmp_path / "again.npz")
    assert cli.main([*argv, "--start", out, "--out", again, "--sigma", "0.3"]) == 0
    assert "sigma 0.3, learning rate 0.05" in capsys.readouterr().out

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
            join = ["join", "--brain", "evolved", "--weights", out, "--owner", "eve"]
            return await asyncio.to_thread(cli.main, [*join, "--url", server.url, "--lives", "1"])

    assert asyncio.run(scenario()) == 0
    assert "eve's" in capsys.readouterr().out
    assert cli.main(["join", "--brain", "random", "--weights", out, "--owner", "x"]) == 1
    assert "for --brain evolved" in capsys.readouterr().err
    assert cli.main(["join", "--weights", str(tmp_path / "nope.npz"), "--owner", "x"]) == 1
    assert "No such file" in capsys.readouterr().err  # --weights alone means evolved
    assert (
        cli.main(
            [
                "evolve",
                "--start",
                again,
                "--out",
                out,
                "--generations",
                "1",
                "--population",
                "2",
                "--episodes",
                "1",
                "--max-steps",
                "20",
                "--workers",
                "1",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out  # the shape and the step size: inherited from --start
    assert "evolving a 3-neuron brain" in printed and "sigma 0.3" in printed
    assert (
        cli.main(["evolve", "--start", again, "--out", out, "--hidden", "5", "--workers", "1"]) == 1
    )
    assert "hidden neurons" in capsys.readouterr().err
    assert cli.main(["evolve", "--out", out, "--population", "3", "--workers", "1"]) == 1
    assert "even" in capsys.readouterr().err
    assert cli.main(["evolve", "--out", out, "--graph", "full", "--control", "1"]) == 1
    assert "--graph, --control: only for --brain connectome" in capsys.readouterr().err


def test_an_evolve_cut_short_keeps_its_last_finished_generation(tmp_path, capsys, monkeypatch):
    from neurogarden.brains import Genome

    real = cli.evolve

    def cut_short(config, checkpoint=None, **rest):
        def keep(result):
            checkpoint(result)
            if len(result.history) == 2:
                raise KeyboardInterrupt

        return real(config, checkpoint=keep, **rest)

    monkeypatch.setattr(cli, "evolve", cut_short)
    argv = [
        "evolve", "--out", str(tmp_path / "cut"), "--generations", "5", "--population", "2",
        "--episodes", "1", "--max-steps", "20", "--hidden", "3", "--workers", "1", "--seed", "4",
    ]  # fmt: skip
    assert cli.main(argv) == 1
    assert "cut.npz holds generation 2" in capsys.readouterr().err
    _, meta = Genome.load(tmp_path / "cut.npz")
    assert meta["generations"] == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cut.npz"]  # no half-written file
