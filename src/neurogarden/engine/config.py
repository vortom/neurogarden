"""Every engine tunable lives in one frozen, validated dataclass."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

# Bump on any change that alters state evolution or observations for identical inputs.
RULES_VERSION = 1

NEED_MAX = 1000
LIGHT_MAX = 1000
NIGHT_LIGHT_THRESHOLD = 500
MAX_VISION_RADIUS = 3

_POSITIVE = (
    "fruit_bites",
    "fruit_lifetime",
    "tree_radius",
    "tree_max_fruit",
    "smell_range_fruit",
    "smell_range_humidity",
    "smell_range_nest",
)
_NON_NEGATIVE = (
    "satiety_drain",
    "hydration_drain",
    "energy_rest",
    "energy_rest_nest",
    "fruit_bite_satiety",
    "drink_hydration",
    "starve_damage",
    "regen_threshold",
    "regen_amount",
    "tree_initial_fruit",
)
_INITIAL = ("initial_satiety", "initial_hydration", "initial_energy", "initial_health")


@dataclass(frozen=True)
class Config:
    # needs
    initial_satiety: int = 700
    initial_hydration: int = 700
    initial_energy: int = 800
    initial_health: int = 1000
    satiety_drain: int = 1
    hydration_drain: int = 1
    # energy delta per action
    energy_idle: int = -1
    energy_move: int = -2
    energy_move_night: int = -4
    energy_consume: int = -1
    energy_rest: int = 8
    energy_rest_nest: int = 12
    # feeding
    fruit_bites: int = 4
    fruit_bite_satiety: int = 150
    drink_hydration: int = 150
    # health
    starve_damage: int = 5
    regen_threshold: int = 300
    regen_amount: int = 1
    max_age: int | None = None
    # time: dawn ramp [0, dawn_end), day, dusk ramp [dusk_start, night_start), night
    day_length: int = 1200
    dawn_end: int = 100
    dusk_start: int = 700
    night_start: int = 800
    # vision
    vision_radius_day: int = 3
    vision_radius_night: int = 1
    # smell
    smell_range_fruit: int = 12
    smell_range_humidity: int = 10
    smell_range_nest: int = 16
    # fruit
    tree_radius: int = 2
    tree_max_fruit: int = 3
    tree_initial_fruit: int = 2
    fruit_spawn_permille: int = 8
    fruit_lifetime: int = 1500

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "max_age" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{f.name} must be an int, got {value!r}")
        if self.max_age is not None and self.max_age <= 0:
            raise ValueError("max_age must be positive or None")
        for name in _INITIAL:
            if not 0 < getattr(self, name) <= NEED_MAX:
                raise ValueError(f"{name} must be in 1..{NEED_MAX}")
        for name in _POSITIVE:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if not 0 < self.dawn_end <= self.dusk_start < self.night_start <= self.day_length:
            raise ValueError("need 0 < dawn_end <= dusk_start < night_start <= day_length")
        if not 0 <= self.vision_radius_night <= self.vision_radius_day <= MAX_VISION_RADIUS:
            raise ValueError(f"need 0 <= night radius <= day radius <= {MAX_VISION_RADIUS}")
        if not 0 <= self.fruit_spawn_permille <= 1000:
            raise ValueError("fruit_spawn_permille must be in 0..1000")
        if self.tree_initial_fruit > self.tree_max_fruit:
            raise ValueError("tree_initial_fruit must not exceed tree_max_fruit")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**data)
