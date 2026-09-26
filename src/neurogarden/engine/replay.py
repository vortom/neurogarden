"""Replays: seed + actions reproduce a world; checkpoints prove it.

The caller does the file I/O; the engine only builds and verifies the data.
"""

from __future__ import annotations

from .config import RULES_VERSION, Config
from .world import World


class ReplayMismatch(AssertionError):
    def __init__(self, tick: int, expected: str, actual: str) -> None:
        super().__init__(f"state hash differs at tick {tick}: expected {expected}, got {actual}")
        self.tick = tick


def _run(replay: dict, on_checkpoint) -> None:
    world = World.from_map(replay["map"], Config.from_dict(replay["config"]), replay["seed"])
    agent_ids = []
    for spawn in replay["spawns"]:
        at = tuple(spawn["at"]) if spawn["at"] is not None else None
        agent_ids.append(world.spawn(body=spawn["body"], at=at))
    every = replay["checkpoint_every"]
    for actions in replay["actions"]:
        world.step(dict(zip(agent_ids, actions, strict=True)))
        if world.tick % every == 0 or world.tick == len(replay["actions"]):
            on_checkpoint(world.tick, world.state_hash())


def record(
    map_text: str,
    seed: int,
    actions: list[list[int]],
    config: Config | None = None,
    spawns: list[dict] | None = None,
    checkpoint_every: int = 100,
) -> dict:
    """Run the actions (one list per tick, aligned with `spawns`) and return the replay."""
    replay = {
        "rules_version": RULES_VERSION,
        "config": (config or Config()).to_dict(),
        "map": map_text,
        "seed": seed,
        "spawns": spawns or [{"body": "fly", "at": None}],
        "actions": [list(tick_actions) for tick_actions in actions],
        "checkpoint_every": checkpoint_every,
        "checkpoints": {},
    }
    _run(replay, lambda tick, digest: replay["checkpoints"].__setitem__(str(tick), digest))
    return replay


def verify(replay: dict) -> None:
    """Re-run the replay; raise ReplayMismatch at the first differing checkpoint."""
    if replay["rules_version"] != RULES_VERSION:
        raise ValueError(
            f"replay has rules_version {replay['rules_version']}, engine is {RULES_VERSION}"
        )
    seen: set[str] = set()

    def check(tick: int, digest: str) -> None:
        expected = replay["checkpoints"].get(str(tick))
        if expected != digest:
            raise ReplayMismatch(tick, str(expected), digest)
        seen.add(str(tick))

    _run(replay, check)
    missing = set(replay["checkpoints"]) - seen
    if missing:
        raise ReplayMismatch(int(min(missing, key=int)), "a checkpoint", "no such tick")
