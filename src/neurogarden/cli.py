"""`neurogarden` — serve a world, join it with a brain, watch it, print the protocol schema,
and read the archive back: the hall of flies, a life replayed, a life exported, the whole
history verified."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from neurogarden.brains import BRAINS, WEIGHTED, ConnectomeGenome, Genome
from neurogarden.brains.evolved import weights_kind
from neurogarden.connectome import data as connectome_data
from neurogarden.connectome.data import VARIANTS
from neurogarden.connectome.model import POOLED
from neurogarden.dojo.distil import DistilConfig, distil
from neurogarden.dojo.evolve import BRAINS as BREEDABLE
from neurogarden.dojo.evolve import EvolveConfig, evolve
from neurogarden.dojo.render_ansi import AgentGlimpse, View, render_view
from neurogarden.dojo.stats import FITNESSES
from neurogarden.engine import RULES_VERSION
from neurogarden.engine.clock import day_number
from neurogarden.protocol import ProtocolError, build_catalog, schema_text
from neurogarden.sdk import (
    DEFAULT_URL,
    AsyncClient,
    ConnectionLost,
    ServerError,
    run_brain,
    run_flock,
)
from neurogarden.server import Archive, ServerConfig, history, parse_npc, serve
from neurogarden.server import frames as server_frames
from neurogarden.server.archive import MEMORY, ArchiveError, Life
from neurogarden.server.roster import GhostRoster
from neurogarden.server.static_files import bundle_present

_HOME_AND_CLEAR = "\x1b[H\x1b[2J"
DEFAULT_ARCHIVE = "neurogarden.db"
# Everything that means "this world would not have us": one line on stderr, exit 1.
_REFUSALS = (ServerError, ProtocolError, ConnectionLost, ValueError)


def _refuse(err: Exception) -> int:
    print(f"neurogarden: {err}", file=sys.stderr)
    return 1


def _add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--token", default="dev")


def _add_archive_arg(parser: argparse.ArgumentParser, what: str) -> None:
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE, metavar="PATH", help=what)


def _add_life_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--owner", required=True)
    parser.add_argument("--life", type=int, required=True, metavar="N", help="the owner's Nth fly")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurogarden", description="Small worlds. Strange minds.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    serve_cmd = commands.add_parser("serve", help="run a live world")
    serve_cmd.add_argument("--map", default=None, help="a new world's map (default drosoville)")
    serve_cmd.add_argument("--seed", type=int, default=None, help="a new world's seed (default 0)")
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
    _add_archive_arg(
        serve_cmd,
        "the world's file: resumed if it exists, created if not; "
        f"{MEMORY} for a world that ends with the process",
    )

    join_cmd = commands.add_parser("join", help="connect a brain to a live world")
    join_cmd.add_argument(
        "--brain",
        choices=sorted(BRAINS),
        default=None,
        help="default scripted; evolved with --weights",
    )
    join_cmd.add_argument("--owner", required=True)
    join_cmd.add_argument("--seed", type=int, default=0)
    join_cmd.add_argument("--lives", type=int, default=None, help="stop after N lives")
    join_cmd.add_argument(
        "--weights", default=None, metavar="FILE.npz", help="an evolved brain's own weights"
    )
    _add_connection_args(join_cmd)

    watch_cmd = commands.add_parser("watch", help="spectate a live world in the terminal")
    watch_cmd.add_argument("--owner", default="watcher")
    watch_cmd.add_argument("--follow", default=None, help="highlight this owner's fly")
    watch_cmd.add_argument("--ascii", action="store_true")
    watch_cmd.add_argument("--frames", type=int, default=None, help="stop after N frames")
    _add_connection_args(watch_cmd)

    commands.add_parser("schema", help="print the protocol JSON Schema")

    flock_cmd = commands.add_parser(
        "flock", help="connect several brains at once, each its own owner"
    )
    flock_cmd.add_argument("--brain", choices=sorted(BRAINS), default=None)
    flock_cmd.add_argument("--count", type=int, default=5, help="how many flies")
    flock_cmd.add_argument(
        "--owner-prefix", default=None, help="owners are PREFIX-1 … PREFIX-N (default: the brain)"
    )
    flock_cmd.add_argument("--seed", type=int, default=0)
    flock_cmd.add_argument("--lives", type=int, default=None, help="lives per fly, then stop")
    flock_cmd.add_argument(
        "--weights", default=None, metavar="FILE.npz", help="the brains' own weights"
    )
    _add_connection_args(flock_cmd)

    connectome_cmd = commands.add_parser(
        "connectome", help="the MaleCNS wiring: fetch the files, build a graph, see what is cached"
    )
    connectome_cmd.add_argument("action", choices=("fetch", "build", "info"))
    connectome_cmd.add_argument("--graph", choices=VARIANTS, default="central")
    connectome_cmd.add_argument(
        "--min-synapses", type=int, default=connectome_data.DEFAULT_MIN_SYNAPSES
    )

    evolve_cmd = commands.add_parser("evolve", help="evolve a brain in the dojo")
    evolve_cmd.add_argument(
        "--brain", choices=BREEDABLE, default=None, help="what to breed (default evolved)"
    )
    evolve_cmd.add_argument(
        "--graph", choices=VARIANTS, default=None, help="connectome: which graph"
    )
    evolve_cmd.add_argument("--min-synapses", type=int, default=None, help="connectome only")
    evolve_cmd.add_argument(
        "--control", type=int, default=None, metavar="SEED",
        help="connectome: breed on the row-shuffled graph (the control)",
    )  # fmt: skip
    evolve_cmd.add_argument("--substeps", type=int, default=None, help="connectome only")
    evolve_cmd.add_argument(
        "--pooled", type=int, default=None, help="connectome: features the readout sees"
    )
    evolve_cmd.add_argument("--out", required=True, metavar="FILE.npz", help="where the weights go")
    evolve_cmd.add_argument("--generations", type=int, default=EvolveConfig.generations)
    evolve_cmd.add_argument("--population", type=int, default=EvolveConfig.population)
    evolve_cmd.add_argument("--episodes", type=int, default=EvolveConfig.episodes)
    evolve_cmd.add_argument("--max-steps", type=int, default=EvolveConfig.max_steps)
    evolve_cmd.add_argument("--sigma", type=float, default=EvolveConfig.sigma)
    evolve_cmd.add_argument("--learning-rate", type=float, default=EvolveConfig.learning_rate)
    evolve_cmd.add_argument(
        "--hidden", type=int, default=None, help=f"neurons (default {EvolveConfig.hidden})"
    )
    evolve_cmd.add_argument(
        "--fitness", choices=sorted(FITNESSES), default=None, help=f"default {EvolveConfig.fitness}"
    )
    evolve_cmd.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"softmax temperature the brains act at; 0 = always the highest score "
        f"(default {EvolveConfig.temperature})",
    )
    evolve_cmd.add_argument("--seed", type=int, default=EvolveConfig.seed)
    evolve_cmd.add_argument("--map", default=EvolveConfig.map, help="the world to breed in")
    evolve_cmd.add_argument(
        "--workers", type=int, default=None, help="evaluation processes (default: all cores)"
    )
    evolve_cmd.add_argument(
        "--start", default=None, metavar="FILE.npz", help="carry on from these weights"
    )

    distil_cmd = commands.add_parser(
        "distil", help="teach a connectome brain its readout from the evolved brain"
    )
    distil_cmd.add_argument("--out", required=True, metavar="FILE.npz", help="where the weights go")
    distil_cmd.add_argument(
        "--teacher", default=None, metavar="FILE.npz", help="an evolved brain (default: shipped)"
    )
    distil_cmd.add_argument("--graph", choices=VARIANTS, default=DistilConfig.graph)
    distil_cmd.add_argument("--min-synapses", type=int, default=DistilConfig.min_synapses)
    distil_cmd.add_argument(
        "--control", type=int, default=None, metavar="SEED",
        help="learn on the row-shuffled graph (the control)",
    )  # fmt: skip
    distil_cmd.add_argument("--substeps", type=int, default=DistilConfig.substeps)
    distil_cmd.add_argument("--pooled", type=int, default=DistilConfig.pooled)
    distil_cmd.add_argument(
        "--gain", type=float, default=DistilConfig.gain, help="the network gain (not learned here)"
    )
    distil_cmd.add_argument(
        "--input-gain", type=float, default=DistilConfig.input_gain,
        help="how hard the senses drive their neurons (not learned here)",
    )  # fmt: skip
    distil_cmd.add_argument(
        "--rounds", type=int, default=DistilConfig.rounds,
        help="the teacher flies the first, the student the others",
    )  # fmt: skip
    distil_cmd.add_argument("--lives", type=int, default=DistilConfig.lives, help="per round")
    distil_cmd.add_argument("--max-steps", type=int, default=DistilConfig.max_steps)
    distil_cmd.add_argument("--temperature", type=float, default=DistilConfig.temperature)
    distil_cmd.add_argument("--seed", type=int, default=DistilConfig.seed)
    distil_cmd.add_argument("--map", default=DistilConfig.map, help="the world to learn in")
    distil_cmd.add_argument(
        "--workers", type=int, default=None, help="processes flying the lives (default: all cores)"
    )

    lives_cmd = commands.add_parser("lives", help="the hall of flies: every life in an archive")
    _add_archive_arg(lives_cmd, "the world's file")
    lives_cmd.add_argument("--owner", default=None, help="only this owner's lives")

    replay_cmd = commands.add_parser("replay", help="watch an archived life again, in the terminal")
    _add_archive_arg(replay_cmd, "the world's file")
    _add_life_args(replay_cmd)
    replay_cmd.add_argument("--fps", type=float, default=20.0, help="frames per second")
    replay_cmd.add_argument("--ascii", action="store_true")
    replay_cmd.add_argument("--frames", type=int, default=None, help="stop after N frames")

    export_cmd = commands.add_parser("export", help="a life as (observation, action) arrays")
    _add_archive_arg(export_cmd, "the world's file")
    _add_life_args(export_cmd)
    export_cmd.add_argument("--out", required=True, metavar="FILE.npz")

    verify_cmd = commands.add_parser(
        "verify", help="re-run an archive's history from its first snapshot, check every hash"
    )
    _add_archive_arg(verify_cmd, "the world's file")
    return parser


def banner(server) -> None:
    """Printed once the world is up, so `--port 0` announces the port it really bound."""
    url = server.url
    config = server.config
    print(f"NeuroGarden — {config.map} (seed {config.seed}) on {url}")
    if getattr(server, "resumed", False):
        world = server.runner.world
        day = day_number(world.tick, world.config)
        lives = len(server.archive.lives())
        print(f"resumed at tick {world.tick} (day {day}); {lives} lives so far")
    if config.archive != MEMORY:
        print(f"archive: {config.archive}")
    print(f"join:  neurogarden join --owner you --brain scripted --url {url}")
    print(f"watch: neurogarden watch --url {url}")
    if config.web and bundle_present():  # the server warns when it is missing
        print(f"play:  open http://{config.host}:{server.port}/ in a browser")


def cmd_serve(args) -> int:
    try:
        held = Archive.peek(args.archive)  # an existing world says what it is
        map_name = args.map if args.map is not None else (held.map_name if held else "drosoville")
        seed = args.seed if args.seed is not None else (held.seed if held else None)
        npcs = [parse_npc(spec) for spec in (args.npc or ["scripted:1"])]
        config = ServerConfig(
            map=map_name,
            seed=seed if seed is not None else 0,  # an archive without a seed accepts any
            tps=args.tps,
            host=args.host,
            port=args.port,
            token=args.token,
            npcs=[npc for npc in npcs if npc[0] != "none"],
            hello_timeout=args.hello_timeout,
            web=not args.no_web,
            archive=args.archive,
        )
        asyncio.run(serve(config, on_ready=banner))
    except (ValueError, OSError) as err:  # bad flags, a foreign archive, or the port is taken
        return _refuse(err)
    except KeyboardInterrupt:
        pass
    return 0


def _obituary(owner: str, name: str, lineage: int, stats) -> str:
    causes = ", ".join(stats.death_causes) or "unknown"
    return (
        f"{owner}'s {name} (#{lineage}) lived {stats.lifespan} ticks "
        f"({stats.days} days) | {causes} | bites {stats.bites} drinks {stats.drinks}"
    )


def _make_brain(args, seed: int | None = None):
    seed = args.seed if seed is None else seed
    if args.weights is None:
        return BRAINS[args.brain or "scripted"](seed=seed)
    kind = args.brain or weights_kind(args.weights)
    if kind not in WEIGHTED:
        raise ValueError("--weights is for --brain evolved or connectome")
    return WEIGHTED[kind](seed=seed, path=args.weights)


def cmd_join(args) -> int:
    try:
        brain = _make_brain(args)
    except (ValueError, OSError) as err:  # no such file, or not an evolved brain's weights
        return _refuse(err)

    def report(fly) -> None:
        print(_obituary(args.owner, fly.name, fly.lineage, fly.stats))

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


# --- the archive, read back ----------------------------------------------------------------


def _open_archive(path: str) -> Archive:
    """An archive to read from; raises ArchiveError (a ValueError) for anything else."""
    archive = Archive.open(path, readonly=True)
    if archive.world_info is None:  # a database, but not (yet) a world: nothing to read
        archive.close()
        raise ArchiveError(f"{path} holds no world")
    return archive


def _find_life(archive: Archive, owner: str, lineage: int) -> Life:
    life = archive.life(owner, lineage)
    if life is None:
        raise ValueError(f"{owner} has no life #{lineage} in {archive.path}")
    return life


def cmd_lives(args) -> int:
    try:
        archive = _open_archive(args.archive)
    except ValueError as err:
        return _refuse(err)
    try:
        info = archive.world_info
        now = archive.next_tick()
        lives = archive.lives(args.owner)
        if info is not None:
            day = day_number(now, info.config)
            print(f"{info.map_name} (seed {info.seed}) — tick {now}, day {day}, {len(lives)} lives")
        print(
            f"{'owner':<16} {'#':>3} {'name':<14} {'born':>7} {'lived':>7}  cause / bites / drinks"
        )
        for life in lives:
            stats = life.stats
            if stats is None:
                lived, detail = f"{now - life.born_tick}*", "still alive"
            else:
                causes = ", ".join(life.causes) or "unknown"
                lived = str(stats.lifespan)
                detail = f"{causes} / {stats.bites} / {stats.drinks}"
            print(
                f"{life.owner:<16} {life.lineage:>3} {life.name:<14} {life.born_tick:>7} "
                f"{lived:>7}  {detail}"
            )
    finally:
        archive.close()
    return 0


def cmd_replay(args) -> int:
    if args.fps <= 0:
        return _refuse(ValueError("--fps must be positive"))
    try:
        archive = _open_archive(args.archive)
    except ValueError as err:
        return _refuse(err)
    try:
        life = _find_life(archive, args.owner, args.life)
        info = archive.world_info
        roster = GhostRoster(archive.lives())
        end = archive.life_end(life)
        start = history.world_at(archive, life.born_tick).world
        world_map = server_frames.world_message(start, info.map_name).payload  # never changes
        told = archive.chronicle_between(life.born_tick, end)  # the lines of those days
        heard = 0  # how many of them the viewer has reached
        shown = 0
        for moment in history.playback(archive, life.born_tick, end, start):
            while heard < len(told) and told[heard][0] <= moment.tick:
                heard += 1
            frame = server_frames.frame_message(
                moment.world, roster, moment.result.events, moment.tick, {}
            ).payload
            view = frame_view(world_map, frame, told[:heard])
            text = render_view(view, focus=life.agent_id, ascii=args.ascii, roster=True)
            sys.stdout.write(
                f"{_HOME_AND_CLEAR}{text}\n  ghost: {life.owner}'s {life.name} (#{life.lineage})"
                f" · tick {moment.tick} of {life.born_tick}–{end - 1}\n"
            )
            sys.stdout.flush()
            shown += 1
            if args.frames is not None and shown >= args.frames:
                break
            time.sleep(1.0 / args.fps)
    except (ValueError, OSError) as err:  # no such life, or the terminal went away
        return _refuse(err)
    except KeyboardInterrupt:
        pass
    finally:
        archive.close()
    return 0


def export_life(archive: Archive, life: Life) -> dict[str, np.ndarray]:
    """(observation, action) pairs of one life: what the brain saw, and what it did next.

    The observation after tick t is what the brain answered with its action for tick t+1,
    so the pairs are (observation_t, action_{t+1}); the hatching tick (an enforced idle) and
    the final observation (nothing follows it) are left out.
    """
    channels: dict[str, list[np.ndarray]] = defaultdict(list)
    actions: list[int] = []
    ticks: list[int] = []
    previous = None
    for moment in history.playback(archive, life.born_tick, archive.life_end(life)):
        if previous is not None:
            for name, values in previous.items():
                channels[name].append(values)
            actions.append(moment.inputs.actions.get(life.agent_id, 0))
            ticks.append(moment.tick - 1)
        previous = moment.result.observations.get(life.agent_id)
    if not actions:
        raise ValueError(f"nothing to export: {life.owner}'s {life.name} has not acted yet")
    arrays = {name: np.stack(values) for name, values in channels.items()}
    arrays["actions"] = np.array(actions, dtype=np.int64)
    arrays["ticks"] = np.array(ticks, dtype=np.int64)
    return arrays


def cmd_export(args) -> int:
    try:
        archive = _open_archive(args.archive)
    except ValueError as err:
        return _refuse(err)
    try:
        life = _find_life(archive, args.owner, args.life)
        arrays = export_life(archive, life)
        meta = {
            "owner": life.owner,
            "lineage": life.lineage,
            "name": life.name,
            "body": life.body,
            "born_tick": life.born_tick,
            "died_tick": life.died_tick,
            "causes": list(life.causes),
            "rules_version": RULES_VERSION,
            "archive": archive.path,
            "catalog": build_catalog().bodies[life.body].model_dump(),
        }
        np.savez_compressed(args.out, meta=json.dumps(meta), **arrays)
        print(
            f"wrote {args.out}: {len(arrays['actions'])} steps of "
            f"{life.owner}'s {life.name} (#{life.lineage})"
        )
    except (ValueError, OSError) as err:  # no such life, or nowhere to write the file
        return _refuse(err)
    finally:
        archive.close()
    return 0


def cmd_verify(args) -> int:
    try:
        archive = _open_archive(args.archive)
    except ValueError as err:
        return _refuse(err)
    try:
        report = history.verify(archive)
    except ValueError as err:
        return _refuse(err)
    finally:
        archive.close()
    print(
        f"ok: {report.ticks} ticks re-run, {report.checkpoints} checkpoints and "
        f"{report.snapshots} snapshots reproduced"
    )
    return 0


def _writable(target: str) -> None:
    out_dir = Path(target).resolve().parent
    if not out_dir.is_dir():  # found out now, not after the last generation
        raise ValueError(f"cannot write to {out_dir}: no such directory")


def _on_graph(trainable) -> str:
    wiring = trainable.ready()
    control = "" if wiring.control is None else f", random control {wiring.control}"
    return (
        f"a connectome brain on the {wiring.variant} graph ({wiring.n} neurons, "
        f"{wiring.connections} connections{control})"
    )


def _keeper(target: str):
    """Where a long run's weights go and the function that writes them after each of its
    steps — beside the name, then renamed, so the file is whole or absent and a run cut
    short keeps what it had. Also the list of steps written so far."""
    out = Path(target)
    out = out if out.suffix == ".npz" else out.with_name(out.name + ".npz")
    saved: list[int] = []

    def keep(result) -> None:
        partial = result.genome.save(out.with_name(out.stem + ".part.npz"), result.meta)
        partial.replace(out)
        saved.append(len(result.history))

    return out, keep, saved


def cmd_evolve(args) -> int:
    try:
        start = parent = None
        if args.start is not None:
            loader = ConnectomeGenome if weights_kind(args.start) == "connectome" else Genome
            start, parent = loader.load(args.start)
        # Carrying on from weights keeps their shape and settings unless told otherwise.
        inherited = parent or {}

        def setting(given, key):
            return given if given is not None else inherited.get(key, getattr(EvolveConfig, key))

        config = EvolveConfig(
            brain=args.brain or inherited.get("brain") or EvolveConfig.brain,
            graph=args.graph or inherited.get("graph") or EvolveConfig.graph,
            min_synapses=setting(args.min_synapses, "min_synapses"),
            control=args.control if args.control is not None else inherited.get("control"),
            substeps=setting(args.substeps, "substeps"),
            # the start's own width, whatever its meta says or leaves out
            pooled=args.pooled if args.pooled is not None else getattr(start, "pooled", POOLED),
            generations=args.generations,
            population=args.population,
            sigma=args.sigma,
            learning_rate=args.learning_rate,
            hidden=(
                args.hidden
                if args.hidden is not None
                else inherited.get("hidden", EvolveConfig.hidden)
            ),
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            workers=args.workers,
            map=args.map,
            fitness=args.fitness or inherited.get("fitness") or EvolveConfig.fitness,
            temperature=(
                args.temperature
                if args.temperature is not None
                else inherited.get("temperature", EvolveConfig.temperature)
            ),
        )
        _writable(args.out)
        if config.brain == "connectome":
            what = _on_graph(config.trainable())  # refused here if the graph is not cached
        else:
            connectome_only = {
                "--graph": args.graph,
                "--min-synapses": args.min_synapses,
                "--control": args.control,
                "--substeps": args.substeps,
                "--pooled": args.pooled,
            }
            given = [flag for flag, value in connectome_only.items() if value is not None]
            if given:  # said out loud, not silently dropped
                raise ValueError(f"{', '.join(given)}: only for --brain connectome")
            what = f"a {config.hidden}-neuron brain"
        if start is not None:  # refused now, not after the banner
            config.trainable().check_start(start, parent)
    except (ValueError, OSError) as err:
        return _refuse(err)

    def report(generation) -> None:
        print(
            f"gen {generation.index + 1:>3}/{config.generations}: best {generation.best:7.0f}"
            f"  mean {generation.mean:7.0f}  centre {generation.centre:7.0f}"
            f"  ({generation.seconds:.1f}s)",
            flush=True,
        )

    print(
        f"evolving {what}: {config.generations} generations of "
        f"{config.population}, {config.episodes} lives each up to {config.max_steps} ticks, "
        f"fitness {config.fitness}, temperature {config.temperature}"
    )
    out, keep, saved = _keeper(args.out)
    try:
        evolve(config, start=start, on_generation=report, parent=parent, checkpoint=keep)
    except KeyboardInterrupt:
        kept = f"{out} holds generation {saved[-1]}" if saved else "nothing written"
        print(f"stopped; {kept}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as err:
        return _refuse(err)
    print(f"wrote {out}")
    return 0


def cmd_distil(args) -> int:
    try:
        config = DistilConfig(
            rounds=args.rounds,
            lives=args.lives,
            max_steps=args.max_steps,
            seed=args.seed,
            workers=args.workers,
            map=args.map,
            temperature=args.temperature,
            teacher=args.teacher,
            graph=args.graph,
            min_synapses=args.min_synapses,
            control=args.control,
            substeps=args.substeps,
            pooled=args.pooled,
            gain=args.gain,
            input_gain=args.input_gain,
        )
        _writable(args.out)
        what = _on_graph(config.trainable())  # refused here if the graph is not cached
        if args.teacher is not None:
            Genome.load(args.teacher)  # and here if the teacher is not an evolved brain
    except (ValueError, OSError) as err:
        return _refuse(err)

    def report(entry) -> None:
        lived = statistics.median(entry.lifespans)
        unsettled = "" if entry.settled else ", fit not settled"
        print(
            f"round {entry.index + 1}/{config.rounds}: the {entry.flown_by} flew, median "
            f"lifespan {lived:.0f} | {entry.ticks} moments, agreement {entry.agreement:.2f}, "
            f"divergence {entry.divergence:.3f}{unsettled}  ({entry.seconds:.1f}s)",
            flush=True,
        )

    print(
        f"teaching {what}: {config.rounds} rounds of {config.lives} lives up to "
        f"{config.max_steps} ticks, {config.pooled} pooled features, "
        f"teacher {config.teacher or 'the shipped evolved brain'}"
    )
    out, keep, saved = _keeper(args.out)
    try:
        distil(config, on_round=report, checkpoint=keep)
    except KeyboardInterrupt:
        kept = f"{out} holds round {saved[-1]}" if saved else "nothing written"
        print(f"stopped; {kept}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as err:
        return _refuse(err)
    print(f"wrote {out}")
    return 0


def cmd_flock(args) -> int:
    if args.count < 1:
        return _refuse(ValueError("--count must be at least 1"))
    try:
        kind = args.brain or (weights_kind(args.weights) if args.weights else "connectome")
        args.brain = kind
        if args.weights and kind == "connectome":  # bred on a shuffled graph: the control,
            _, bred = ConnectomeGenome.load(args.weights)  # whatever the file is called
            kind = "connectome" if bred.get("control") is None else "connectome-random"
        if args.owner_prefix:
            prefix = args.owner_prefix
        elif args.weights:  # its own owners, so it does not take over the shipped flock's flies
            prefix = re.sub(r"[^A-Za-z0-9_.-]", "-", Path(args.weights).stem)[:48] or kind
        else:
            prefix = {"connectome": "cns", "connectome-random": "rnd"}.get(kind, kind)
        seeds = [args.seed + index for index in range(args.count)]
        brains = [_make_brain(args, seed=seed) for seed in seeds]
        owners = [f"{prefix}-{index + 1}" for index in range(args.count)]

        def report(life) -> None:
            print(_obituary(life.owner, life.name, life.lineage, life.stats), flush=True)

        print(f"flying {args.count} {kind} flies as {owners[0]} … {owners[-1]}", flush=True)
        run_flock(
            args.url,
            brains,
            owners,
            token=args.token,
            lives=args.lives,
            on_life=report,
            seeds=seeds,
        )
    except (*_REFUSALS, OSError) as err:
        return _refuse(err)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_connectome(args) -> int:
    try:
        if args.action == "fetch":
            shown: dict[str, int] = {}

            def progress(name: str, done: int, total: int | None) -> None:
                tenth = done * 10 // total if total else done >> 27  # every 10%, or 128 MB
                if shown.get(name) != tenth:
                    shown[name] = tenth
                    of = f" of {total / 1e6:.0f} MB" if total else ""
                    print(f"{name}: {done / 1e6:.0f} MB{of}", flush=True)

            for path in connectome_data.fetch(on_progress=progress):
                print(f"have {path}")
        elif args.action == "build":
            wiring = connectome_data.build(args.graph, args.min_synapses, on_progress=print)
            print(f"built {wiring.variant} (>= {wiring.min_synapses} synapses): {wiring.digest}")
        else:
            report = connectome_data.info()
            print(f"cache: {report['cache']}")
            for key, size in report["sources"].items():
                print(f"  {key:<13} {'missing' if size is None else f'{size / 1e6:.0f} MB'}")
            for graph in report["graphs"]:
                print(
                    f"  graph {graph['graph']} (>= {graph['min_synapses']} synapses): "
                    f"{graph['neurons']} neurons, {graph['connections']} connections, "
                    f"{graph['digest']}"
                )
            for name in report["broken"]:
                print(f"  {name}: cannot be read; run `neurogarden connectome build` again")
            if not report["graphs"]:
                print("  no graph built yet: `neurogarden connectome build`")
    except (ValueError, OSError) as err:
        return _refuse(err)
    return 0


COMMANDS = {
    "evolve": cmd_evolve,
    "distil": cmd_distil,
    "flock": cmd_flock,
    "connectome": cmd_connectome,
    "serve": cmd_serve,
    "join": cmd_join,
    "watch": cmd_watch,
    "schema": cmd_schema,
    "lives": cmd_lives,
    "replay": cmd_replay,
    "export": cmd_export,
    "verify": cmd_verify,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
