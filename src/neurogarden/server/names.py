"""Fly names and moods: the small things that make a live world worth watching.

The glyphs a mood is drawn with belong to the renderer (`dojo.render_ansi`), not here.
"""

from __future__ import annotations

from neurogarden.engine.config import NEED_MAX
from neurogarden.engine.rng import SplitMix64

ADJECTIVES = (
    "Amber", "Brisk", "Cinder", "Dusty", "Ember", "Fuzzy", "Gilded", "Hasty", "Ivory", "Jolly",
    "Keen", "Lucky", "Misty", "Nimble", "Olive", "Plucky", "Quiet", "Rusty", "Sunny", "Tiny",
    "Umber", "Velvet", "Wily", "Zesty",
)  # fmt: skip
NOUNS = (
    "Wing", "Zip", "Buzz", "Whirr", "Speck", "Dot", "Glint", "Flick", "Hum", "Drift",
    "Pip", "Mote", "Spark", "Twitch", "Flit", "Blip",
)  # fmt: skip


def fly_name(owner: str, lineage: int) -> str:
    """Deterministic and cute: the same owner's third fly is always the same 'Dusty Wing'."""
    seed = sum(ord(c) * 131**i for i, c in enumerate(owner)) + lineage * 7919
    rng = SplitMix64(seed)
    return f"{ADJECTIVES[rng.randbelow(len(ADJECTIVES))]} {NOUNS[rng.randbelow(len(NOUNS))]}"


def mood(satiety: int, hydration: int, energy: int, health: int, alive: bool) -> str:
    """One word for a thought bubble. Derived from the needs a spectator can see; never a rule."""
    if not alive:
        return "dead"
    lowest_name, lowest = min(
        (("hungry", satiety), ("thirsty", hydration), ("sleepy", energy)), key=lambda pair: pair[1]
    )
    if health < NEED_MAX // 4 or lowest < 150:
        return "dying"
    if lowest < 350:
        return "desperate"
    if lowest < 600:
        return lowest_name
    return "content"
