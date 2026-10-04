"""Watch a brain live in the terminal.

uv run python -m neurogarden.dojo.watch --brain scripted
"""

from __future__ import annotations

import argparse
import sys
import time

from neurogarden.brains import BRAINS
from neurogarden.brains.base import brain_seed

from .env import NeuroGardenEnv
from .render_ansi import render

_HOME_AND_CLEAR = "\x1b[H\x1b[2J"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch a brain live in Drosoville.")
    parser.add_argument("--brain", choices=sorted(BRAINS), default="scripted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tps", type=float, default=10.0, help="ticks per second; 0 = flat out")
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--ascii", action="store_true", help="plain ASCII instead of emoji")
    args = parser.parse_args(argv)

    try:
        brain = BRAINS[args.brain](seed=args.seed)
    except (ValueError, OSError) as err:  # weights or a graph it needs are not here
        print(f"neurogarden: {err}", file=sys.stderr)
        return 1
    env = NeuroGardenEnv(max_steps=args.max_steps)
    observation, info = env.reset(seed=args.seed)
    brain.reset(brain_seed(args.seed))

    def draw() -> None:
        frame = render(env.world, env.agent_id, ascii=args.ascii)
        sys.stdout.write(f"{_HOME_AND_CLEAR}{frame}\n")
        sys.stdout.flush()

    try:
        draw()
        while True:
            observation, _, terminated, truncated, info = env.step(brain.act(observation))
            draw()  # after every step, so truncation and death both show their final frame
            if terminated or truncated:
                break
            if args.tps > 0:
                time.sleep(1.0 / args.tps)
    except KeyboardInterrupt:
        pass
    stats = info["stats"]
    causes = ", ".join(stats.death_causes) or "still alive"
    print(f"\n{args.brain}: lived {stats.lifespan} ticks ({stats.days} days) | {causes}")
    print(f"bites {stats.bites} | drinks {stats.drinks} | explored {stats.tiles_explored} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
