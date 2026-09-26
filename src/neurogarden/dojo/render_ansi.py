"""Terminal renderer: emoji by default, plain ASCII as a fallback.

Renders a `View` — plain data either side can build: the dojo from a `World`, the spectator
from the server's `world` + `frame` messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurogarden.engine.clock import day_number, light_at
from neurogarden.engine.config import NIGHT_LIGHT_THRESHOLD
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
MOOD_GLYPHS = {
    "content": "✨",
    "hungry": "🍎",
    "thirsty": "💧",
    "sleepy": "💤",
    "desperate": "❗",
    "dying": "☠️",
    "dead": "✝",
}
MOOD_ASCII = {
    "content": "~",
    "hungry": "f",
    "thirsty": "w",
    "sleepy": "z",
    "desperate": "!",
    "dying": "x",
    "dead": "+",
}
_BAR_WIDTH = 10
_NEEDS = ("satiety", "hydration", "energy", "health")


@dataclass
class AgentGlimpse:
    """What a spectator may know about a fly."""

    agent_id: int
    x: int
    y: int
    satiety: int
    hydration: int
    energy: int
    health: int
    alive: bool = True
    owner: str = ""
    name: str = ""
    lineage: int = 0
    connected: bool = True
    mood: str = ""
    say: str = ""


@dataclass
class View:
    terrain: np.ndarray  # (H, W) uint8
    resources: list[tuple[int, int, int, int]]  # (x, y, kind, amount)
    agents: list[AgentGlimpse]
    tick: int
    day: int
    light: int
    chronicle: list[str] = field(default_factory=list)


def view_of(world: World) -> View:
    state = world.state
    ys, xs = (state.resource_kind == Resource.FRUIT).nonzero()
    resources = [
        (int(x), int(y), int(state.resource_kind[y, x]), int(state.resource_amount[y, x]))
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    ]
    agents = [
        AgentGlimpse(a.id, a.x, a.y, a.satiety, a.hydration, a.energy, a.health, a.alive)
        for a in state.agents.values()
    ]
    return View(
        terrain=state.terrain,
        resources=resources,
        agents=agents,
        tick=state.tick,
        day=day_number(state.tick, state.config),
        light=light_at(state.tick, state.config),
    )


def _bar(value: int, glyphs: tuple[str, str]) -> str:
    filled = (value * _BAR_WIDTH + 999) // 1000
    return glyphs[0] * filled + glyphs[1] * (_BAR_WIDTH - filled)


def _roster_line(agent: AgentGlimpse, focus: int | None, glyphs: dict, ascii: bool) -> str:
    marker = ">" if agent.agent_id == focus else " "
    link = "" if agent.connected else " (away)"
    bars = " ".join(
        f"{need[0].upper()}{_bar(getattr(agent, need), glyphs['bar'])}" for need in _NEEDS
    )
    bubble = f' "{agent.say}"' if agent.say else ""
    glyph = (MOOD_ASCII if ascii else MOOD_GLYPHS).get(agent.mood, "?")
    who = f"{marker}{agent.owner:<15} {agent.name:<14} #{agent.lineage:<2}"
    return f"{who} {bars} {glyph}{link}{bubble}"


def render_view(
    view: View, focus: int | None = None, ascii: bool = False, roster: bool = False
) -> str:
    """The map, a status line, then one agent's need bars (dojo) or the roster (live world)."""
    glyphs = _ASCII if ascii else _EMOJI
    height, width = view.terrain.shape
    occupied = {(a.x, a.y) for a in view.agents if a.alive}
    fruit = {(x, y) for x, y, kind, _ in view.resources if kind == Resource.FRUIT}
    rows = []
    for y in range(height):
        cells = []
        for x in range(width):
            if (x, y) in occupied:
                cells.append(glyphs["fly"])
            elif (x, y) in fruit:
                cells.append(glyphs["fruit"])
            else:
                cells.append(glyphs[Terrain(int(view.terrain[y, x]))])
        rows.append("".join(cells))

    sky = glyphs["night"] if view.light < NIGHT_LIGHT_THRESHOLD else glyphs["day"]
    rows.append(f"Day {view.day} | tick {view.tick} | {sky} {view.light}")

    if roster:
        alive = [a for a in view.agents if a.alive]
        for agent in sorted(alive, key=lambda a: (a.owner, a.agent_id)):
            rows.append(_roster_line(agent, focus, glyphs, ascii))
        rows.extend(f"  {line}" for line in view.chronicle)
        return "\n".join(rows)

    focused = next((a for a in view.agents if a.agent_id == focus), None)
    if focused is not None:
        for need in _NEEDS:
            value = getattr(focused, need)
            rows.append(f"{need:<10}{_bar(value, glyphs['bar'])} {value:>4}")
        if not focused.alive:
            rows.append("the fly has died")
    return "\n".join(rows)


def render(world: World, agent_id: int | None = None, ascii: bool = False) -> str:
    """The whole map plus a HUD for one agent. Spectator view: not for brains."""
    return render_view(view_of(world), focus=agent_id, ascii=ascii)
