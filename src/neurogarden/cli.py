"""`neurogarden` — serve a world, join it with a brain, watch it, print the protocol schema."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import numpy as np

from neurogarden.brains import BRAINS
from neurogarden.dojo.render_ansi import AgentGlimpse, View, render_view
from neurogarden.protocol import ProtocolError, schema_text
from neurogarden.sdk import DEFAULT_URL, AsyncClient, ConnectionLost, ServerError, run_brain
from neurogarden.server import ServerConfig, parse_npc, serve

_HOME_AND_CLEAR = "\x1b[H\x1b[2J"
# Everything that means "this world would not have us": one line on stderr, exit 1.
_REFUSALS = (ServerError, ProtocolError, ConnectionLost, ValueError)


def _refuse(err: Exception) -> int:
    print(f"neurogarden: {err}", file=sys.stderr)
    return 1


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--token", default="dev")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurogarden", description="Small worlds. Strange minds.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    serve_cmd = commands.add_parser("serve", help="run a live world")
    serve_cmd.add_argument("--map", default="drosoville")
    serve_cmd.add_argument("--seed", type=int, default=0)
    serve_cmd.add_argument("--tps", type=float, default=5.0, help="ticks per second")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8765)
    serve_cmd.add_argument("--token", default="dev")
    serve_cmd.add_argument(
        "--npc",
        action="append",
        default=None,
        metavar="KIND[:N]",
        help="hosted brains kept alive in the world (default scripted:1; 'none' for none)",
    )
    serve_cmd.add_argument("--hello-timeout", type=float, default=5.0)
    serve_cmd.add_argument("--no-web", action="store_true", help="do not serve the browser page")

    join_cmd = commands.add_parser("join", help="connect a brain to a live world")
    join_cmd.add_argument("--brain", choices=sorted(BRAINS), default="scripted")
    join_cmd.add_argument("--owner", required=True)
    join_cmd.add_argument("--seed", type=int, default=0)
    join_cmd.add_argument("--lives", type=int, default=None, help="stop after N lives")
    _add_connection_args(join_cmd)

    watch_cmd = commands.add_parser("watch", help="spectate a live world in the terminal")
    watch_cmd.add_argument("--owner", default="watcher")
    watch_cmd.add_argument("--follow", default=None, help="highlight this owner's fly")
    watch_cmd.add_argument("--ascii", action="store_true")
    watch_cmd.add_argument("--frames", type=int, default=None, help="stop after N frames")
    _add_connection_args(watch_cmd)

    commands.add_parser("schema", help="print the protocol JSON Schema")
    return parser


def banner(server) -> None:
    """Printed once the world is up, so `--port 0` announces the port it really bound."""
    url = server.url
    print(f"NeuroGarden — {server.config.map} (seed {server.config.seed}) on {url}")
    print(f"join:  neurogarden join --owner you --brain scripted --url {url}")
    print(f"watch: neurogarden watch --url {url}")
    if server.config.web:
        print(f"play:  open http://{server.config.host}:{server.port}/ in a browser")


def cmd_serve(args) -> int:
    try:
        npcs = [parse_npc(spec) for spec in (args.npc or ["scripted:1"])]
        config = ServerConfig(
            map=args.map,
            seed=args.seed,
            tps=args.tps,
            host=args.host,
            port=args.port,
            token=args.token,
            npcs=[npc for npc in npcs if npc[0] != "none"],
            hello_timeout=args.hello_timeout,
            web=not args.no_web,
        )
        asyncio.run(serve(config, on_ready=banner))
    except ValueError as err:
        return _refuse(err)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_join(args) -> int:
    brain = BRAINS[args.brain](seed=args.seed)

    def report(fly) -> None:
        stats = fly.stats
        causes = ", ".join(stats.death_causes) or "unknown"
        print(
            f"{args.owner}'s {fly.name} (#{fly.lineage}) lived {stats.lifespan} ticks "
            f"({stats.days} days) | {causes} | bites {stats.bites} drinks {stats.drinks}"
        )

    try:
        run_brain(
            args.url, brain, owner=args.owner, token=args.token, lives=args.lives, on_life=report
        )
    except _REFUSALS as err:
        return _refuse(err)
    except KeyboardInterrupt:
        pass
    return 0


def frame_view(world, frame, chronicle, lines: int = 6) -> View:
    """A renderer View from what a spectator was sent."""
    agents = [
        AgentGlimpse(
            agent_id=a.agent_id,
            x=a.x,
            y=a.y,
            satiety=a.satiety,
            hydration=a.hydration,
            energy=a.energy,
            health=a.health,
            alive=a.alive,
            owner=a.owner,
            name=a.name,
            lineage=a.lineage,
            connected=a.connected,
            mood=a.mood,
            say=a.say,
        )
        for a in frame.agents
    ]
    return View(
        terrain=np.array(world.terrain, dtype=np.uint8),
        resources=[(r.x, r.y, r.kind, r.amount) for r in frame.resources],
        agents=agents,
        tick=frame.tick,
        day=frame.day,
        light=frame.light,
        chronicle=[text for _, text in list(chronicle)[-lines:]],
    )


def scoreboard(frame, limit: int = 6) -> str:
    return "  ".join(
        f"{s.owner}:{s.best_lifespan}{'*' if s.alive else ''}" for s in frame.scores[:limit]
    )


async def _watch(args) -> int:
    client = AsyncClient(args.url, owner=args.owner, token=args.token, role="spectator")
    async with client:
        shown = 0
        async for frame in client.frames():
            if client.spectacle.world is None:
                continue
            view = frame_view(client.spectacle.world, frame, client.spectacle.chronicle)
            focus = next(
                (a.agent_id for a in frame.agents if a.owner == args.follow and a.alive), None
            )
            text = render_view(view, focus=focus, ascii=args.ascii, roster=True)
            sys.stdout.write(f"{_HOME_AND_CLEAR}{text}\n  best: {scoreboard(frame)}\n")
            sys.stdout.flush()
            shown += 1
            if args.frames is not None and shown >= args.frames:
                break
    return 0


def cmd_watch(args) -> int:
    try:
        return asyncio.run(_watch(args))
    except _REFUSALS as err:
        return _refuse(err)
    except KeyboardInterrupt:
        return 0


def cmd_schema(args) -> int:
    sys.stdout.write(schema_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return {"serve": cmd_serve, "join": cmd_join, "watch": cmd_watch, "schema": cmd_schema}[
        args.command
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
