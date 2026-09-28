import numpy as np
import pytest

from neurogarden.dojo.rewards import (
    BodyState,
    drive,
    make_homeostatic,
    resolve,
    survival,
    wellbeing,
)
from neurogarden.dojo.stats import StatsTracker, fitness_lifespan
from neurogarden.engine.events import Event

FULL = BodyState(1000, 1000, 1000, 1000, 10)
HALF = BodyState(500, 500, 500, 1000, 10)
EMPTY = BodyState(0, 0, 0, 1, 10)


def test_body_state_reads_the_body_channel():
    observation = {"body": np.array([700, 600, 500, 900, 7], dtype=np.int32)}
    assert BodyState.from_observation(observation) == BodyState(700, 600, 500, 900, 7)


def test_drive_is_zero_when_full_and_three_when_empty_and_ignores_health():
    assert drive(FULL) == 0.0
    assert drive(EMPTY) == 3.0
    assert drive(HALF) == pytest.approx(0.75)
    assert drive(BodyState(500, 500, 500, 1, 10)) == drive(HALF)


def test_wellbeing_is_one_minus_normalised_drive_and_zero_on_death():
    assert wellbeing(FULL, FULL, [], False) == 1.0
    assert wellbeing(FULL, HALF, [], False) == pytest.approx(0.75)
    assert wellbeing(FULL, EMPTY, [], False) == 0.0
    assert wellbeing(FULL, HALF, [], True) == 0.0


def test_survival_is_one_per_living_step():
    assert survival(FULL, HALF, [], False) == 1.0
    assert survival(FULL, HALF, [], True) == 0.0


def test_homeostatic_is_drive_reduction_with_a_death_penalty():
    homeostatic = make_homeostatic(death_penalty=10.0)
    assert homeostatic(HALF, FULL, [], False) == pytest.approx(0.75)
    assert homeostatic(FULL, HALF, [], False) == pytest.approx(-0.75)
    assert homeostatic(FULL, HALF, [], True) == pytest.approx(-10.75)


def test_resolve_takes_a_name_or_a_callable():
    assert resolve("wellbeing") is wellbeing
    assert resolve("survival") is survival
    assert resolve("homeostatic")(HALF, FULL, [], False) == pytest.approx(0.75)

    def custom(prev, body, events, died):
        return 42.0

    assert resolve(custom) is custom
    with pytest.raises(ValueError):
        resolve("glory")


def test_tracker_counts_own_events_and_explored_tiles():
    tracker = StatsTracker(agent_id=1, start=(2, 2), day_length=1200)
    events = [
        Event(0, "moved", 1, {"from": (2, 2), "to": (3, 2), "direction": 1}),
        Event(0, "moved", 2, {"from": (5, 5), "to": (5, 6), "direction": 2}),  # someone else
        Event(0, "ate", 1, {"bites_left": 3}),
        Event(0, "drank", 1),
        Event(0, "rested", 1, {"on_nest": False}),
        Event(0, "bumped", 1, {"direction": 0}),
        Event(0, "fruit_spawned", None, {"x": 1, "y": 1}),
    ]
    tracker.update(events, BodyState(1000, 1000, 1000, 1000, 1))
    tracker.update([Event(1, "moved", 1, {"from": (3, 2), "to": (2, 2), "direction": 3})], FULL)
    stats = tracker.stats
    assert (stats.bites, stats.drinks, stats.rest_ticks, stats.bumps) == (1, 1, 1, 1)
    assert stats.tiles_explored == 2  # revisiting the start adds nothing
    assert stats.lifespan == stats.score == 10
    assert fitness_lifespan(stats) == 10.0


def test_tracker_records_days_death_and_mean_wellbeing():
    tracker = StatsTracker(agent_id=1, start=(0, 0), day_length=4)
    tracker.update([], BodyState(1000, 1000, 1000, 1000, 1))
    tracker.update(
        [Event(1, "died", 1, {"causes": ["starvation"]})], BodyState(500, 500, 500, 0, 2)
    )
    stats = tracker.stats
    assert stats.death_causes == ("starvation",)
    assert stats.days == 0
    assert stats.mean_wellbeing == pytest.approx((1.0 + 0.75) / 2)
    tracker.update([], BodyState(1000, 1000, 1000, 1000, 9))
    assert tracker.stats.days == 2


def test_a_tracker_carries_on_after_a_round_trip_through_json():
    import json

    tracker = StatsTracker(agent_id=1, start=(2, 2), day_length=4)
    tracker.update([Event(0, "moved", 1, {"from": (2, 2), "to": (3, 2), "direction": 1})], FULL)
    tracker.update([Event(1, "ate", 1, {"bites_left": 3})], BodyState(500, 500, 500, 1000, 2))
    twin = StatsTracker.from_dict(json.loads(json.dumps(tracker.to_dict())))
    later = [Event(2, "moved", 1, {"from": (3, 2), "to": (2, 2), "direction": 3})]
    tracker.update(later, BodyState(400, 400, 400, 1000, 9))
    twin.update(later, BodyState(400, 400, 400, 1000, 9))
    assert twin.stats == tracker.stats
    assert twin.stats.tiles_explored == 2 and twin.stats.days == 2
