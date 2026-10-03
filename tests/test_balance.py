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


@pytest.mark.slow
def test_the_evolved_brain_outlives_the_random_one():
    """The shipped weights must be worth shipping: a learned fly, not a lucky one."""
    from neurogarden.brains import EvolvedBrain

    evolved = statistics.median(lifespans(EvolvedBrain))
    random = statistics.median(lifespans(RandomBrain))
    print(f"\nmedian lifespan: evolved {evolved}, random {random}")
    assert evolved >= 4000, "the shipped weights should live most of a long life"
    assert evolved >= 2 * random, "evolution should at least double a random brain's life"


@pytest.mark.slow
def test_the_connectome_brain_is_reported_beside_its_control():
    """Real wiring against shuffled wiring, taught alike. The number is the point: it is
    printed, and only living longer than a random brain is asserted — whether the wiring helps
    is a result, not a requirement."""
    from neurogarden.brains import ConnectomeBrain, RandomGraphBrain

    try:
        ConnectomeBrain()
        RandomGraphBrain()
    except (ValueError, OSError) as missing:
        pytest.skip(f"needs the cached MaleCNS graph and the shipped weights: {missing}")
    env = NeuroGardenEnv(max_steps=3000, any_hour=True)  # born at any hour, as in a live garden
    seeds = range(10)

    def median(brain_cls):
        return statistics.median(run_episode(env, brain_cls(), seed=s).lifespan for s in seeds)

    connectome, control, random = (
        median(ConnectomeBrain),
        median(RandomGraphBrain),
        median(RandomBrain),
    )
    print(
        f"\nmedian lifespan: connectome {connectome}, shuffled control {control}, random {random}"
    )
    assert connectome > random, "the bred connectome brain should outlive a random one"
