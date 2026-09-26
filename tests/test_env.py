import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import neurogarden.dojo as dojo
from neurogarden.dojo import NeuroGardenEnv, TinyObservation
from neurogarden.engine import Action, Config, World, maps

TINY = """
#####
#.N~#
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
