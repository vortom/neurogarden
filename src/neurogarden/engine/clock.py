"""Day and night as a pure function of the tick."""

from __future__ import annotations

from .config import LIGHT_MAX, NIGHT_LIGHT_THRESHOLD, Config


def light_at(tick: int, config: Config) -> int:
    phase = tick % config.day_length
    if phase < config.dawn_end:
        return phase * LIGHT_MAX // config.dawn_end
    if phase < config.dusk_start:
        return LIGHT_MAX
    if phase < config.night_start:
        return (config.night_start - phase) * LIGHT_MAX // (config.night_start - config.dusk_start)
    return 0


def is_night(tick: int, config: Config) -> bool:
    return light_at(tick, config) < NIGHT_LIGHT_THRESHOLD


def day_number(tick: int, config: Config) -> int:
    return tick // config.day_length + 1
