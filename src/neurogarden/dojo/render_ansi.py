"""Terminal renderer: emoji by default, plain ASCII as a fallback."""

from __future__ import annotations

from neurogarden.engine.clock import day_number, is_night, light_at
from neurogarden.engine.tiles import Resource, Terrain
from neurogarden.engine.world import World

_EMOJI = {
    Terrain.GROUND: "· ",
    Terrain.ROCK: "🪨",
    Terrain.WATER: "🟦",
    Terrain.TREE: "🌳",
    Terrain.NEST: "🏠",
    "fruit": "🍎",
    "fly": "🪰",
    "bar": ("█", "░"),
    "day": "☀️ ",
    "night": "🌙",
}
_ASCII = {
    Terrain.GROUND: ".",
    Terrain.ROCK: "#",
    Terrain.WATER: "~",
    Terrain.TREE: "T",
    Terrain.NEST: "N",
    "fruit": "f",
    "fly": "@",
    "bar": ("#", "-"),
    "day": "day",
    "night": "night",
}
_BAR_WIDTH = 10
_NEEDS = ("satiety", "hydration", "energy", "health")


def _bar(value: int, glyphs: tuple[str, str]) -> str:
    filled = (value * _BAR_WIDTH + 999) // 1000
    return glyphs[0] * filled + glyphs[1] * (_BAR_WIDTH - filled)


def render(world: World, agent_id: int | None = None, ascii: bool = False) -> str:
    """The whole map plus a HUD for one agent. Spectator view: not for brains."""
    glyphs = _ASCII if ascii else _EMOJI
    state = world.state
    rows = []
    for y in range(state.height):
        cells = []
        for x in range(state.width):
            if state.occupant[y, x] != 0:
                cells.append(glyphs["fly"])
            elif state.resource_kind[y, x] == Resource.FRUIT:
                cells.append(glyphs["fruit"])
            else:
                cells.append(glyphs[Terrain(int(state.terrain[y, x]))])
        rows.append("".join(cells))

    sky = glyphs["night"] if is_night(state.tick, state.config) else glyphs["day"]
    light = light_at(state.tick, state.config)
    rows.append(f"Day {day_number(state.tick, state.config)} | tick {state.tick} | {sky} {light}")
    agent = state.agents.get(agent_id) if agent_id is not None else None
    if agent is not None:
        for need in _NEEDS:
            value = getattr(agent, need)
            rows.append(f"{need:<10}{_bar(value, glyphs['bar'])} {value:>4}")
        if not agent.alive:
            rows.append("the fly has died")
    return "\n".join(rows)
