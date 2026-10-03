"""Measurement (sub-project 6): how long a connectome brain lives, over many seeds at once.

The balance guard (`uv run pytest -m slow`) flies ten lives of up to 3000 ticks in one
process. This flies as many as asked, each in its own process, and prints every lifespan:
the yardstick the spec quotes (20 lives of up to 6000 ticks, as for the evolved brain).

    uv run python docs/superpowers/spikes/connectome_lifespans.py FILE.npz [LIVES] [TICKS] [WORKERS]

FILE.npz is a connectome brain's weights (`neurogarden distil` or `evolve` wrote it; the
shipped ones are in src/neurogarden/brains/weights/). Life i is flown on world seed i.
Needs the cached graph the weights name (`neurogarden connectome build`).
"""

import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

from neurogarden.brains import ConnectomeBrain, run_episode
from neurogarden.dojo import NeuroGardenEnv

PATH = sys.argv[1]
LIVES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TICKS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 4


def life(seed: int) -> tuple[int, int, int, str]:
    stats = run_episode(NeuroGardenEnv(max_steps=TICKS), ConnectomeBrain(path=PATH), seed=seed)
    return stats.lifespan, stats.bites, stats.drinks, ",".join(stats.death_causes) or "alive"


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        lives = list(pool.map(life, range(LIVES)))
    spans = [span for span, *_ in lives]
    alive = sum(end == "alive" for *_, end in lives)
    print(
        f"{PATH}: median {statistics.median(spans)} of {TICKS}, {alive} of {LIVES} alive at the end"
    )
    print("  lifespans", spans)
    print("  bites", [bites for _, bites, _, _ in lives])
    print("  drinks", [drinks for _, _, drinks, _ in lives])
    print("  ends", [end for *_, end in lives])
