"""Measurement (sub-project 6): how long a brain lives, over many seeds at once.

The balance guard (`uv run pytest -m slow`) flies ten lives of up to 3000 ticks in one
process. This flies as many as asked, each in its own process, and prints every lifespan:
the yardstick the spec quotes (20 lives of up to 6000 ticks, born at any hour of the day —
as flies in a live garden are).

    uv run python docs/superpowers/spikes/connectome_lifespans.py BRAIN [LIVES] [TICKS] [WORKERS]

BRAIN is a name (`connectome`, `connectome-random`, `evolved`, `scripted`, `random`) or a
connectome brain's weights file (what `neurogarden distil` or `evolve` wrote). Life i is
flown on world seed i. Add `dawn` as a last argument to hatch every fly at tick 0 instead,
the way the dojo did before lives could begin at any hour. A connectome brain needs the
cached graph its weights name (`neurogarden connectome build`).
"""

import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

from neurogarden.brains import BRAINS, ConnectomeBrain, run_episode
from neurogarden.dojo import NeuroGardenEnv

ARGS = [arg for arg in sys.argv[1:] if arg != "dawn"]
ANY_HOUR = "dawn" not in sys.argv[1:]
BRAIN = ARGS[0]
LIVES = int(ARGS[1]) if len(ARGS) > 1 else 20
TICKS = int(ARGS[2]) if len(ARGS) > 2 else 6000
WORKERS = int(ARGS[3]) if len(ARGS) > 3 else 4


def life(seed: int) -> tuple[int, int, int, str]:
    brain = BRAINS[BRAIN]() if BRAIN in BRAINS else ConnectomeBrain(path=BRAIN)
    stats = run_episode(NeuroGardenEnv(max_steps=TICKS, any_hour=ANY_HOUR), brain, seed=seed)
    return stats.lifespan, stats.bites, stats.drinks, ",".join(stats.death_causes) or "alive"


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        lives = list(pool.map(life, range(LIVES)))
    spans = [span for span, *_ in lives]
    alive = sum(end == "alive" for *_, end in lives)
    born = "any hour" if ANY_HOUR else "dawn"
    print(
        f"{BRAIN}, born at {born}: median {statistics.median(spans)} of {TICKS}, "
        f"{alive} of {LIVES} alive at the end"
    )
    print("  lifespans", spans)
    print("  bites", [bites for _, bites, _, _ in lives])
    print("  drinks", [drinks for _, _, drinks, _ in lives])
    print("  ends", [end for *_, end in lives])
