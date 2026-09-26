import dataclasses

import pytest

from neurogarden.engine.config import RULES_VERSION, Config


def test_defaults_are_valid_and_match_spec():
    cfg = Config()
    assert (cfg.initial_satiety, cfg.initial_hydration, cfg.initial_energy) == (700, 700, 800)
    assert (cfg.energy_move, cfg.energy_move_night) == (-2, -4)
    assert cfg.max_age is None
    assert cfg.day_length == 1200
    assert RULES_VERSION == 1


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().day_length = 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"initial_satiety": 0},
        {"initial_health": 1001},
        {"fruit_bites": 0},
        {"starve_damage": -1},
        {"max_age": 0},
        {"dawn_end": 0},
        {"dusk_start": 900},
        {"night_start": 1300},
        {"vision_radius_day": 4},
        {"vision_radius_night": 3, "vision_radius_day": 2},
        {"fruit_spawn_permille": 1001},
        {"tree_initial_fruit": 4},
        {"satiety_drain": 1.5},
        {"fruit_bites": True},
    ],
)
def test_nonsense_is_rejected(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


def test_dict_round_trip():
    cfg = Config(max_age=5000, fruit_bites=2)
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError):
        Config.from_dict({"wings": 2})
