"""The body catalog: what every number in an observation means, built from the engine itself."""

from __future__ import annotations

from neurogarden.engine.body import BODIES, VISION_SIZE, action_names, observation_spec
from neurogarden.engine.config import LIGHT_MAX, NEED_MAX, NIGHT_LIGHT_THRESHOLD
from neurogarden.engine.events import EVENT_TYPES
from neurogarden.engine.senses import SCENTS
from neurogarden.engine.tiles import Resource, Terrain

from .messages import BodyInfo, Catalog, ChannelInfo

SMELL_COLUMNS = ("own", "n", "e", "s", "w")
VISION_LAYERS = ("terrain", "resource", "occupant")
OCCUPANT_CLASSES = {"none": 0, "self": 1, "other": 2}
TOUCH_FIELDS = ("bumped", "resource_kind", "water_adjacent", "on_nest")
BODY_FIELDS = ("satiety", "hydration", "energy", "health", "age")
ENV_FIELDS = ("light",)
# What an event may carry. A brain's own stream never sees the coordinate keys
# (`x`, `y`, `from`, `to`); a spectator's frame does.
EVENT_DATA_KEYS = {
    "moved": ("from", "to", "direction"),
    "bumped": ("direction",),
    "ate": ("bites_left",),
    "drank": (),
    "consume_failed": (),
    "rested": ("on_nest",),
    "invalid_action": ("action",),
    "need_depleted": ("need",),
    "damaged": ("amount", "causes"),
    "died": ("causes",),
    "fruit_spawned": ("x", "y"),
    "fruit_rotted": ("x", "y"),
}


def _enum_table(enum) -> dict[str, int]:
    return {member.name.lower(): int(member) for member in enum}


def _axes(name: str) -> dict[str, list[str]]:
    if name == "smell":
        return {"rows": list(SCENTS), "cols": list(SMELL_COLUMNS)}
    if name == "vision":
        return {"layers": list(VISION_LAYERS)}
    if name == "touch":
        return {"fields": list(TOUCH_FIELDS)}
    if name == "body":
        return {"fields": list(BODY_FIELDS)}
    if name == "env":
        return {"fields": list(ENV_FIELDS)}
    raise ValueError(f"no axis names for channel {name!r}")


def build_body(body: str) -> BodyInfo:
    channels: dict[str, ChannelInfo] = {}
    for name, spec in observation_spec(body).items():
        info = ChannelInfo(
            shape=list(spec.shape), dtype=spec.dtype, low=spec.low, high=spec.high, axes=_axes(name)
        )
        if name == "vision":
            info.tables = {
                "terrain": _enum_table(Terrain),
                "resource": _enum_table(Resource),
                "occupant": dict(OCCUPANT_CLASSES),
            }
            info.centre = [VISION_SIZE // 2, VISION_SIZE // 2]
            info.north_up = True
        channels[name] = info
    return BodyInfo(frame=BODIES[body].frame, actions=list(action_names(body)), channels=channels)


def build_events() -> dict[str, list[str]]:
    """Every event type the engine can emit, with the data keys it carries."""
    unlisted = EVENT_TYPES - set(EVENT_DATA_KEYS)
    if unlisted:  # a new engine event must say what it carries
        raise ValueError(f"no data keys listed for {sorted(unlisted)}")
    return {name: list(keys) for name, keys in sorted(EVENT_DATA_KEYS.items())}


def build_catalog() -> Catalog:
    return Catalog(
        bodies={name: build_body(name) for name in BODIES},
        constants={
            "need_max": NEED_MAX,
            "light_max": LIGHT_MAX,
            "night_light_threshold": NIGHT_LIGHT_THRESHOLD,
        },
        events=build_events(),
    )
