import dataclasses
import json

import numpy as np
import pytest

from neurogarden.engine import Action, Config, World
from neurogarden.engine.rng import SplitMix64

MAP = """
########
#..T...#
#.N...~#
#......#
########
"""

CORRIDOR = """
#####
#...#
#####
"""


def test_from_map_seeds_initial_fruit_and_keeps_the_map_text():
    world = World.from_map(MAP, seed=1)
    assert int((world.state.resource_kind == 1).sum()) == world.config.tree_initial_fruit
    assert world.map_text.splitlines()[0] == "########"
    assert world.tick == 0


def test_spawn_defaults_to_the_nest_facing_south_with_initial_needs():
    world = World.from_map(MAP)
    fly = world.spawn()
    agent = world.state.agents[fly]
    assert fly == 1
    assert (agent.x, agent.y, agent.facing) == (2, 2, 2)
    assert (agent.satiety, agent.hydration, agent.energy, agent.health) == (700, 700, 800, 1000)
    assert world.state.occupant[2, 2] == 1


def test_later_spawns_hatch_next_to_the_nest_not_in_a_corner():
    world = World.from_map(MAP)
    world.spawn()
    positions = [(a.x, a.y) for a in (world.state.agents[world.spawn()] for _ in range(4))]
    # nest at (2, 2): north, east, south, west neighbours in BFS order
    assert positions == [(2, 1), (3, 2), (2, 3), (1, 2)]
    nestless = World.from_map("...\n...")
    nestless.spawn()
    agent = nestless.state.agents[nestless.spawn()]
    assert (agent.x, agent.y) == (1, 0)  # no nest: row-major fallback


def test_bad_spawns_are_value_errors():
    world = World.from_map(MAP)
    with pytest.raises(ValueError):
        world.spawn(body="dragon")
    with pytest.raises(ValueError):
        world.spawn(at=(0, 0))  # rock
    world.spawn(at=(1, 1))
    with pytest.raises(ValueError):
        world.spawn(at=(1, 1))  # occupied


def test_observe_does_not_advance_time():
    world = World.from_map(MAP)
    fly = world.spawn()
    before = world.state_hash()
    observation = world.observe(fly)
    assert observation["body"].tolist() == [700, 700, 800, 1000, 0]
    assert world.state_hash() == before


def test_step_advances_one_tick_and_returns_observations_and_events():
    world = World.from_map(MAP, Config(fruit_spawn_permille=0))
    fly = world.spawn()
    result = world.step({fly: Action.MOVE_E})
    assert result.tick == world.tick == 1
    # tick 0 is the dark end of dawn, so the move costs the night price of 4
    assert result.observations[fly]["body"].tolist() == [699, 699, 796, 1000, 1]
    assert [e.type for e in result.events] == ["moved"]
    assert result.events[0].tick == 0
    assert result.events[0].data["to"] == (3, 2)
    assert result.agent_events[fly][0].data == {"direction": 1}


def test_missing_action_means_idle():
    world = World.from_map(MAP)
    fly = world.spawn()
    world.step({})
    assert world.state.agents[fly].energy == 799


def test_unknown_agent_id_is_a_value_error():
    world = World.from_map(MAP)
    with pytest.raises(ValueError):
        world.step({7: Action.IDLE})
    with pytest.raises(ValueError):
        world.observe(7)


def test_death_gives_one_final_observation_then_silence():
    world = World.from_map(MAP, Config(initial_satiety=1, initial_health=5))
    fly = world.spawn()
    result = world.step({fly: Action.IDLE})
    assert result.observations[fly]["body"][3] == 0
    assert [e.type for e in result.agent_events[fly]] == ["need_depleted", "damaged", "died"]
    assert world.state.occupant[2, 2] == 0
    after = world.step({fly: Action.MOVE_E})  # actions of the dead are ignored
    assert after.observations == {} and after.agent_events == {}


def test_observe_rejects_a_dead_agent():
    world = World.from_map(MAP, Config(initial_satiety=1, initial_health=5))
    fly = world.spawn()
    world.step({fly: Action.IDLE})
    assert not world.state.agents[fly].alive
    with pytest.raises(ValueError):
        world.observe(fly)


def test_two_agents_never_share_a_tile_and_the_loser_bumps():
    outcomes = set()
    for seed in range(20):
        world = World.from_map(CORRIDOR, seed=seed)
        left, right = world.spawn(at=(1, 1)), world.spawn(at=(3, 1))
        result = world.step({left: Action.MOVE_E, right: Action.MOVE_W})
        a, b = world.state.agents[left], world.state.agents[right]
        assert (a.x, a.y) != (b.x, b.y)
        assert sorted(e.type for e in result.events) == ["bumped", "moved"]
        assert int((world.state.occupant != 0).sum()) == 2
        outcomes.add("left" if a.x == 2 else "right")
    assert outcomes == {"left", "right"}  # the seeded shuffle decides who goes first


def run(seed, steps, action_seed=99):
    world = World.from_map(MAP, seed=seed)
    fly = world.spawn()
    rng = SplitMix64(action_seed)
    for _ in range(steps):
        world.step({fly: rng.randbelow(7)})
    return world


def test_same_seed_and_actions_give_the_same_hash():
    assert run(5, 300).state_hash() == run(5, 300).state_hash()
    assert run(5, 300).state_hash() != run(6, 300).state_hash()


def test_snapshot_restore_continue_equals_an_uninterrupted_run():
    rng = SplitMix64(3)
    actions = [rng.randbelow(7) for _ in range(400)]
    straight = World.from_map(MAP, seed=8)
    fly = straight.spawn()
    paused = World.from_map(MAP, seed=8)
    paused.spawn()
    for action in actions[:150]:
        straight.step({fly: action})
        paused.step({fly: action})
    resumed = World.restore(paused.snapshot())
    for action in actions[150:]:
        a = straight.step({fly: action})
        b = resumed.step({fly: action})
        if fly in a.observations:
            for name in a.observations[fly]:
                assert np.array_equal(a.observations[fly][name], b.observations[fly][name])
    assert resumed.state_hash() == straight.state_hash()


def test_fuzz_invariants_hold_under_random_actions():
    world = World.from_map(MAP, seed=2)
    flies = [world.spawn(), world.spawn()]
    rng = SplitMix64(17)
    for _ in range(1500):
        result = world.step({fly: rng.randbelow(9) for fly in flies})  # includes invalid ids 7, 8
        state = world.state
        for agent in state.agents.values():
            for need in (agent.satiety, agent.hydration, agent.energy, agent.health):
                assert 0 <= need <= 1000
            if agent.alive:
                assert state.occupant[agent.y, agent.x] == agent.id
                assert state.walkable(agent.x, agent.y)
        assert int((state.occupant != 0).sum()) == len(state.living())
        assert (state.resource_amount >= 0).all()
        assert ((state.resource_kind == 0) == (state.resource_amount == 0)).all()
        for agent_id, events in result.agent_events.items():
            for event in events:
                assert event.agent_id == agent_id
                assert not ({"x", "y", "from", "to"} & set(event.data))
        json.dumps([dataclasses.asdict(e) for e in result.events])  # what spectators/storage do


def test_spawn_coordinates_are_coerced_to_python_ints():
    world = World.from_map(MAP)
    world.spawn(at=(np.int64(1), np.int64(1)))
    json.dumps(world.snapshot())


def test_spawn_rejects_non_integral_coordinates():
    world = World.from_map(MAP)
    with pytest.raises(ValueError):
        world.spawn(at=(1.9, 1.2))


def test_despawn_removes_only_the_dead_and_changes_the_hash():
    world = World.from_map(MAP, Config(initial_satiety=1, initial_health=5))
    fly, other = world.spawn(), world.spawn(at=(5, 3))
    with pytest.raises(ValueError, match="alive"):
        world.despawn(fly)
    world.step({fly: Action.IDLE})  # both starve at once; only one corpse is cleared
    before = world.state_hash()
    world.despawn(fly)
    assert fly not in world.state.agents and other in world.state.agents
    assert world.state_hash() != before
    assert world.state.next_agent_id == other + 1  # ids are never reused
    with pytest.raises(ValueError, match="unknown agent"):
        world.despawn(fly)
    with pytest.raises(ValueError, match="unknown agent"):
        world.step({fly: Action.IDLE})


def test_a_despawned_world_restores_to_the_same_hash_and_keeps_its_map():
    world = World.from_map(MAP, Config(initial_satiety=1, initial_health=5), seed=4)
    fly = world.spawn()
    world.step({fly: Action.IDLE})
    world.despawn(fly)
    restored = World.restore(json.loads(json.dumps(world.snapshot())), world.map_text)
    assert restored.state_hash() == world.state_hash()
    assert restored.map_text == world.map_text
    assert World.restore(world.snapshot()).map_text is None
