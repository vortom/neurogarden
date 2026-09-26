import numpy as np
import pytest

from neurogarden.engine.tiles import MapError, Terrain, find_tiles, parse_map

SMALL = """
#####
#.F~#
#TN.#
#####
"""


def test_parse_terrain_and_fruit():
    parsed = parse_map(SMALL)
    assert parsed.terrain.shape == (4, 5)
    assert parsed.terrain.dtype == np.uint8
    assert parsed.terrain[1, 1] == Terrain.GROUND
    assert parsed.terrain[1, 2] == Terrain.GROUND  # 'F' is ground
    assert parsed.terrain[1, 3] == Terrain.WATER
    assert parsed.terrain[2, 1] == Terrain.TREE
    assert parsed.terrain[2, 2] == Terrain.NEST
    assert parsed.terrain[0, 0] == Terrain.ROCK
    assert parsed.fruit == ((2, 1),)


def test_text_is_normalised():
    assert parse_map(SMALL).text == "#####\n#.F~#\n#TN.#\n#####"


def test_void_never_appears_in_a_map():
    assert not (parse_map(SMALL).terrain == Terrain.VOID).any()


def test_empty_map_is_an_error():
    with pytest.raises(MapError) as err:
        parse_map("  \n ")
    assert (err.value.line, err.value.column) == (1, 1)


def test_ragged_row_reports_line_and_column():
    with pytest.raises(MapError) as err:
        parse_map("###\n##\n###")
    assert (err.value.line, err.value.column) == (2, 3)


def test_unknown_character_reports_line_and_column():
    with pytest.raises(MapError) as err:
        parse_map("###\n#?#\n###")
    assert (err.value.line, err.value.column) == (2, 2)


def test_find_tiles_is_row_major_xy():
    terrain = parse_map("T.T\n.T.").terrain
    assert find_tiles(terrain, Terrain.TREE) == ((0, 0), (2, 0), (1, 1))
