import numpy as np

from neurogarden.brains import BRAINS, RandomBrain, ScriptedBrain, run_episode
from neurogarden.brains.base import brain_seed
from neurogarden.dojo import NeuroGardenEnv
from neurogarden.engine import Action
from neurogarden.engine.rng import SplitMix64
from neurogarden.engine.tiles import Terrain


def observation(
    body=(700, 700, 800, 1000, 0),
    touch=(0, 0, 0, 0),
    light=1000,
    smell=None,
    blocked=(),
    terrain_default=Terrain.GROUND,
):
    """A hand-built observation: open `terrain_default` all around unless `blocked` directions."""
    vision = np.zeros((7, 7, 3), dtype=np.uint8)
    vision[:, :, 0] = terrain_default
    vision[3, 3, 2] = 1
    for direction in blocked:
        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[direction]
        vision[3 + dy, 3 + dx, 0] = Terrain.ROCK
    return {
        "smell": np.array(smell or [[0] * 5] * 3, dtype=np.int16),
        "vision": vision,
        "touch": np.array(touch, dtype=np.uint8),
        "body": np.array(body, dtype=np.int32),
        "env": np.array([light], dtype=np.int16),
    }


def test_random_brain_is_seeded_and_covers_all_actions():
    a, b = RandomBrain(seed=1), RandomBrain(seed=1)
    picks = [a.act({}) for _ in range(200)]
    assert picks == [b.act({}) for _ in range(200)]
    assert set(picks) == set(range(7))
    a.reset()
    assert [a.act({}) for _ in range(200)] == picks


def test_rule_1_eats_when_hungry_on_fruit_and_drinks_when_thirsty_by_water():
    brain = ScriptedBrain()
    assert brain.act(observation(body=(500, 700, 800, 1000, 0), touch=(0, 1, 0, 0))) == 5
    assert brain.act(observation(body=(700, 500, 800, 1000, 0), touch=(0, 0, 1, 0))) == 5
    assert brain.act(observation(body=(700, 700, 800, 1000, 0), touch=(0, 1, 1, 0))) != 5


def test_rule_2_critical_energy_rests_in_place():
    brain = ScriptedBrain()
    assert brain.act(observation(body=(700, 700, 100, 1000, 0))) == Action.REST


def test_rule_3_urgent_need_forages_even_at_night():
    brain = ScriptedBrain()
    smell = [[500, 0, 600, 0, 0], [0] * 5, [0] * 5]  # fruit is stronger to the east
    night = observation(body=(200, 700, 800, 1000, 0), light=0, smell=smell)
    assert brain.act(night) == Action.MOVE_E


def test_rule_4_night_goes_home_and_rests_on_the_nest():
    brain = ScriptedBrain()
    smell = [[0] * 5, [0] * 5, [500, 0, 0, 0, 600]]  # nest is stronger to the west
    assert brain.act(observation(light=0, smell=smell)) == Action.MOVE_W
    assert brain.act(observation(light=0, touch=(0, 0, 0, 1))) == Action.REST
    assert brain.act(observation(light=0)) == Action.REST  # no nest scent: rest in place


def test_rule_4_tired_keeps_resting_until_rested():
    brain = ScriptedBrain()
    on_nest = (0, 0, 0, 1)
    assert brain.act(observation(body=(700, 700, 200, 1000, 0), touch=on_nest)) == Action.REST
    assert brain.act(observation(body=(700, 700, 500, 1000, 0), touch=on_nest)) == Action.REST
    assert brain.act(observation(body=(700, 700, 750, 1000, 0), touch=on_nest)) != Action.REST


def test_rules_3_and_4_treat_void_terrain_as_walkable_unknown():
    brain = ScriptedBrain()
    fruit_smell = [[500, 0, 600, 0, 0], [0] * 5, [0] * 5]  # fruit is stronger to the east
    urgent = observation(
        body=(200, 700, 800, 1000, 0), light=0, smell=fruit_smell, terrain_default=Terrain.VOID
    )
    assert brain.act(urgent) == Action.MOVE_E  # rule 3: forages through unseen tiles at night

    nest_smell = [[0] * 5, [0] * 5, [500, 0, 0, 0, 600]]  # nest is stronger to the west
    homebound = observation(light=0, smell=nest_smell, terrain_default=Terrain.VOID)
    assert brain.act(homebound) == Action.MOVE_W  # rule 4: follows the nest scent home


def test_rule_5_serves_the_lower_need_and_only_steps_onto_walkable_tiles():
    brain = ScriptedBrain()
    smell = [[100, 900, 0, 0, 0], [100, 0, 0, 0, 900], [0] * 5]
    thirsty = observation(body=(550, 500, 800, 1000, 0), smell=smell)
    assert brain.act(thirsty) == Action.MOVE_W  # water is west
    hungry = observation(body=(500, 550, 800, 1000, 0), smell=smell)
    assert brain.act(hungry) == Action.MOVE_N  # fruit is north
    walled = observation(body=(500, 550, 800, 1000, 0), smell=smell, blocked=(0,))
    assert brain.act(walled) != Action.MOVE_N


def test_rule_6_explores_without_walking_into_walls():
    brain = ScriptedBrain(seed=3)
    boxed = observation(blocked=(0, 1, 2))
    assert all(brain.act(boxed) == Action.MOVE_W for _ in range(20))
    assert ScriptedBrain().act(observation(blocked=(0, 1, 2, 3))) == Action.IDLE


def test_run_episode_returns_stats_and_is_reproducible():
    def lifespan():
        return run_episode(NeuroGardenEnv(max_steps=300), ScriptedBrain(), seed=5)

    first, second = lifespan(), lifespan()
    assert first.lifespan == 300
    assert (first.bites, first.drinks, first.tiles_explored) == (
        second.bites,
        second.drinks,
        second.tiles_explored,
    )


def test_registry_lists_every_brain():
    assert set(BRAINS) == {"random", "scripted", "evolved"}


def test_brain_seed_decorrelates_from_the_world_seed():
    assert brain_seed(None) is None
    for seed in (0, 1, 42):
        derived = brain_seed(seed)
        assert derived != seed
        world_stream = [SplitMix64(seed).next_u64() for _ in range(3)]
        brain_stream = [SplitMix64(derived).next_u64() for _ in range(3)]
        assert world_stream != brain_stream
