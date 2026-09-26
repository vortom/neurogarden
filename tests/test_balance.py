"""The balance guard: Drosoville must be neither trivially survivable nor impossible."""

import statistics
import time

import pytest

from neurogarden.brains import RandomBrain, ScriptedBrain, run_episode
from neurogarden.dojo import NeuroGardenEnv

SEEDS = range(20)
MAX_STEPS = 6000


def lifespans(brain_cls):
    env = NeuroGardenEnv(max_steps=MAX_STEPS)
    return [run_episode(env, brain_cls(), seed=seed).lifespan for seed in SEEDS]


@pytest.mark.slow
def test_scripted_brain_outlives_random_brain():
    scripted = statistics.median(lifespans(ScriptedBrain))
    random = statistics.median(lifespans(RandomBrain))
    print(f"\nmedian lifespan: scripted {scripted}, random {random}")
    assert scripted >= 3600, "a sensible brain should live at least three days"
    assert scripted >= 3 * random, "a random brain should die young"


@pytest.mark.slow
def test_benchmark_steps_per_second_is_reported():
    env = NeuroGardenEnv(max_steps=MAX_STEPS)
    brain = ScriptedBrain()
    observation, _ = env.reset(seed=0)
    brain.reset(0)
    steps = 3000
    start = time.perf_counter()
    for _ in range(steps):
        observation, _, terminated, truncated, _ = env.step(brain.act(observation))
        if terminated or truncated:
            observation, _ = env.reset(seed=1)
    rate = steps / (time.perf_counter() - start)
    print(f"\n{rate:,.0f} steps per second (design target: 2,000; reported, not asserted)")
