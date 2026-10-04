import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import neurogarden.dojo as dojo
from neurogarden.dojo import NeuroGardenEnv, TinyObservation
from neurogarden.dojo.env import birth_tick
from neurogarden.engine import Action, Config, MapError, World, maps

TINY = """
#####
#.N~#
#####
"""

CORRIDOR = """
#####
#N..#
#####
"""


def test_registered_under_the_neurogarden_namespace():
    env = gymnasium.make("NeuroGarden/Drosoville-v0")
    assert isinstance(env.unwrapped, NeuroGardenEnv)
    assert env.spec.max_episode_steps is None  # the env truncates itself
    assert isinstance(dojo.make(max_steps=10).unwrapped, NeuroGardenEnv)


def test_passes_the_gymnasium_checker():
    env = gymnasium.make("NeuroGarden/Drosoville-v0", render_mode="ansi")
    check_env(env.unwrapped)


def test_spaces_follow_the_body_spec():
    env = NeuroGardenEnv()
    assert env.action_space.n == 7
    assert env.observation_space["vision"].shape == (7, 7, 3)
    assert env.observation_space["smell"].dtype == np.int16
    observation, info = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    assert info["tick"] == 0 and info["day"] == 1 and info["events"] == []


def test_same_seed_same_episode():
    def rollout(seed):
        env = NeuroGardenEnv()
        env.reset(seed=seed)
        env.action_space.seed(0)
        for _ in range(200):
            env.step(env.action_space.sample())
        return env.world.state_hash()

    assert rollout(3) == rollout(3)
    assert rollout(3) != rollout(4)


def test_reset_seed_is_the_world_seed():
    env = NeuroGardenEnv()
    env.reset(seed=123)
    twin = World.from_map(maps.load("drosoville"), seed=123)
    twin.spawn()
    assert env.world.state_hash() == twin.state_hash()


def test_a_life_can_begin_at_any_hour_of_the_day():
    env = NeuroGardenEnv(any_hour=True, max_steps=5)
    day = env.config.day_length
    born = []
    for seed in range(12):
        observation, info = env.reset(seed=seed)
        assert info["tick"] == birth_tick(seed, day) < day  # the same hour for the same seed
        assert observation["body"][4] == 0  # and a newborn all the same
        born.append(info["tick"])
    assert len(set(born)) > 8 and min(born) < day // 3 and max(born) > 2 * day // 3  # spread out
    # The world it is born into is the seed's own world, a part of a day older.
    twin = World.from_map(maps.load("drosoville"), seed=5)
    for _ in range(birth_tick(5, day)):
        twin.step({})
    twin.spawn()
    env.reset(seed=5)
    assert env.world.state_hash() == twin.state_hash()
    # A life keeps its own count: age and the stats start at the birth, not at the world's dawn.
    _, _, _, truncated, info = [env.step(0) for _ in range(5)][-1]
    assert truncated and info["stats"].lifespan == 5 and info["tick"] == birth_tick(5, day) + 5
    assert NeuroGardenEnv().reset(seed=5)[1]["tick"] == 0  # dawn, unless asked otherwise
    unseeded = NeuroGardenEnv(any_hour=True)
    hours = {unseeded.reset()[1]["tick"] for _ in range(6)}  # no seed given: still an hour
    assert all(0 <= hour < day for hour in hours) and len(hours) > 1  # of the day, and not one


def test_death_terminates_and_time_limit_truncates():
    dying = NeuroGardenEnv(map=TINY, config=Config(initial_satiety=2, initial_health=5))
    dying.reset(seed=0)
    _, _, terminated, truncated, _ = dying.step(Action.IDLE)
    assert (terminated, truncated) == (False, False)
    _, reward, terminated, truncated, info = dying.step(Action.IDLE)
    assert (terminated, truncated) == (True, False)
    assert reward == 0.0
    assert info["stats"].death_causes == ("starvation",)
    assert [e.type for e in info["events"]] == ["need_depleted", "damaged", "died"]
    _, reward, terminated, _, _ = dying.step(Action.IDLE)  # finished episode: a no-op
    assert terminated and reward == 0.0

    short = NeuroGardenEnv(map=TINY, max_steps=3)
    short.reset(seed=0)
    flags = [short.step(Action.IDLE)[2:4] for _ in range(3)]
    assert flags == [(False, False), (False, False), (False, True)]


def test_truncation_freezes_the_episode_like_death_does():
    env = NeuroGardenEnv(map=TINY, max_steps=2)
    env.reset(seed=0)
    env.step(Action.IDLE)
    env.step(Action.IDLE)
    assert env.world.tick == 2
    for _ in range(2):  # further calls are a frozen no-op: same flags, the world never advances
        observation, reward, terminated, truncated, _ = env.step(Action.IDLE)
        assert (terminated, truncated) == (False, True)
        assert reward == 0.0
        assert env.world.tick == 2


def test_step_passes_the_action_through_engine_coercion_unchanged():
    for action in (True, 3.0, "abc"):  # neither a bool, nor a float, nor a string is an action id
        env = NeuroGardenEnv(map=CORRIDOR)
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        assert [e.type for e in info["events"]] == ["invalid_action"]

    env = NeuroGardenEnv(map=CORRIDOR)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.int64(2))  # MOVE_E: numpy ints still move the fly
    assert [e.type for e in info["events"]] == ["moved"]


def test_map_argument_loads_bundled_names_or_parses_raw_text():
    NeuroGardenEnv(map=".N.")  # a single-row map text, not a bundled name
    with pytest.raises(MapError):
        NeuroGardenEnv(map="atlantis")


def test_reward_is_pluggable_and_events_carry_no_coordinates():
    seen = []

    def spy(prev, body, events, died):
        seen.append((prev, body, events, died))
        return 2.5

    env = NeuroGardenEnv(map=TINY, reward=spy)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(Action.MOVE_W)
    assert reward == 2.5
    prev, body, events, died = seen[0]
    assert (prev.satiety, body.satiety, died) == (700, 699, False)
    assert events[0].type == "moved" and events[0].data == {"direction": 3}
    assert info["stats"].tiles_explored == 2


def test_info_stats_is_a_snapshot_not_the_live_tracker():
    env = NeuroGardenEnv(map=TINY)
    env.reset(seed=0)
    _, _, _, _, info1 = env.step(Action.IDLE)
    stats1 = info1["stats"]
    _, _, _, _, info2 = env.step(Action.IDLE)
    stats2 = info2["stats"]
    assert stats1 is not stats2
    assert stats1.lifespan == 1


def test_default_reward_is_wellbeing():
    env = NeuroGardenEnv(map=TINY)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(Action.IDLE)
    assert 0.0 < reward < 1.0


def test_ansi_render_returns_text():
    env = NeuroGardenEnv(map=TINY, render_mode="ansi")
    env.reset(seed=0)
    frame = env.render()
    assert "Day 1" in frame and "satiety" in frame
    with pytest.raises(ValueError):
        NeuroGardenEnv(render_mode="human")


def test_tiny_observation_is_a_unit_float_vector():
    env = TinyObservation(NeuroGardenEnv(map=TINY))
    observation, _ = env.reset(seed=0)
    assert observation.shape == (25,) and observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    # smell[humidity] own tile, touch.on_nest, body.satiety, env.light
    assert observation[5] == pytest.approx(0.9)
    assert observation[18] == 1.0
    assert observation[19] == pytest.approx(0.7)
    assert observation[24] == 0.0

    with_vision = TinyObservation(NeuroGardenEnv(map=TINY), include_vision=True)
    observation, _ = with_vision.reset(seed=0)
    assert observation.shape == (25 + 7 * 7 * 11,)
    assert with_vision.observation_space.contains(observation)
