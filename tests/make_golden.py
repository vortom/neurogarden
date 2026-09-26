"""Regenerate tests/data/golden_replay.json.

Run only after an intentional rule change, together with bumping RULES_VERSION:

    uv run python tests/make_golden.py
"""

import json
from pathlib import Path

from neurogarden.engine import maps
from neurogarden.engine.replay import record
from neurogarden.engine.rng import SplitMix64

GOLDEN = Path(__file__).parent / "data" / "golden_replay.json"
TICKS = 1200
AGENTS = 2


def main() -> None:
    rng = SplitMix64(2026)
    # ids 0..8: valid actions plus a few invalid ones, to pin that path down too
    actions = [[rng.randbelow(9) for _ in range(AGENTS)] for _ in range(TICKS)]
    spawns = [{"body": "fly", "at": None}, {"body": "fly", "at": [14, 11]}]
    replay = record(maps.load("drosoville"), seed=7, actions=actions, spawns=spawns)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(replay, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN} with {len(replay['checkpoints'])} checkpoints")


if __name__ == "__main__":
    main()
