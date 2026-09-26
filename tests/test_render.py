from neurogarden.dojo.render_ansi import render
from neurogarden.engine import Config, World

MAP = """
#####
#FN~#
#T..#
#####
"""


def make(**overrides):
    world = World.from_map(MAP, Config(tree_initial_fruit=0, **overrides))
    return world, world.spawn()


def test_ascii_frame_shows_tiles_fruit_and_fly():
    world, fly = make()
    lines = render(world, fly, ascii=True).splitlines()
    assert lines[:4] == ["#####", "#f@~#", "#T..#", "#####"]
    assert lines[4] == "Day 1 | tick 0 | night 0"
    assert lines[5] == "satiety   #######---  700"
    assert lines[7] == "energy    ########--  800"
    assert lines[8] == "health    ########## 1000"


def test_emoji_frame_uses_one_glyph_per_tile():
    world, fly = make()
    assert render(world, fly).splitlines()[1] == "🪨🍎🪰🟦🪨"


def test_frame_without_an_agent_has_no_need_bars():
    world, _ = make()
    assert len(render(world, None, ascii=True).splitlines()) == 5


def test_dead_fly_is_reported():
    world, fly = make(initial_satiety=1, initial_health=5)
    world.step({})
    assert render(world, fly, ascii=True).splitlines()[-1] == "the fly has died"
