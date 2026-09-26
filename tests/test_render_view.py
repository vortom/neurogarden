import numpy as np

from neurogarden.dojo.render_ansi import AgentGlimpse, View, render, render_view, view_of
from neurogarden.engine import Config, World
from neurogarden.engine.tiles import parse_map

MAP = """
#####
#FN~#
#T..#
#####
"""


def test_view_of_a_world_renders_exactly_as_before():
    world = World.from_map(MAP, Config(tree_initial_fruit=0))
    fly = world.spawn()
    view = view_of(world)
    assert view.resources == [(1, 1, 1, 4)] and view.agents[0].agent_id == fly
    assert render(world, fly, ascii=True).splitlines()[:5] == [
        "#####",
        "#f@~#",
        "#T..#",
        "#####",
        "Day 1 | tick 0 | night 0",
    ]


def test_roster_mode_lists_living_flies_with_bars_moods_and_chronicle():
    terrain = parse_map(MAP).terrain
    agents = [
        AgentGlimpse(1, 2, 1, 700, 650, 800, 1000, True, "alice", "Dusty Wing", 1, True, "content"),
        AgentGlimpse(
            2, 3, 2, 300, 900, 900, 1000, True, "bob", "Amber Zip", 3, False, "desperate", "help"
        ),
        AgentGlimpse(3, 1, 1, 0, 0, 0, 0, False, "carol", "Tiny Dot", 2, False, "dead"),
    ]
    view = View(terrain, [], agents, 5, 1, 1000, ["Day 1, dawn: alice's Dusty Wing hatches."])
    lines = render_view(view, focus=2, ascii=True, roster=True).splitlines()
    assert lines[:4] == ["#####", "#.@~#", "#T.@#", "#####"]  # the dead fly is not drawn
    assert lines[4] == "Day 1 | tick 5 | day 1000"
    assert lines[5].startswith(" alice           Dusty Wing     #1  S#######--- H#######--- ")
    assert lines[5].endswith(" ~")
    assert lines[6].startswith(">bob             Amber Zip      #3 ")
    assert lines[6].endswith(' ! (away) "help"')
    assert lines[7] == "  Day 1, dawn: alice's Dusty Wing hatches."
    assert len(lines) == 8


def test_a_speech_bubble_cannot_smuggle_escape_codes_into_the_terminal():
    terrain = np.full((1, 1), 1, dtype=np.uint8)
    shouty = AgentGlimpse(1, 0, 0, 900, 900, 900, 1000, True, "a", "B C", 1, True, "content")
    shouty.say = "\x1b[2Jrun\x07"
    line = render_view(View(terrain, [], [shouty], 0, 1, 1000), roster=True).splitlines()[-1]
    assert line.endswith(' "[2Jrun"') and "\x1b" not in line and "\x07" not in line


def test_emoji_roster_line_uses_mood_glyphs():
    terrain = np.full((1, 1), 1, dtype=np.uint8)
    agent = AgentGlimpse(1, 0, 0, 500, 900, 900, 1000, True, "a", "B C", 1, True, "hungry")
    line = render_view(View(terrain, [], [agent], 0, 1, 1000), roster=True).splitlines()[-1]
    assert line.endswith("🍎")
