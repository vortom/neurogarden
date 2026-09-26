from neurogarden.engine.body import Action
from neurogarden.engine.config import Config
from neurogarden.engine.metabolism import energy_delta, metabolise
from neurogarden.engine.state import Agent, new_state
from neurogarden.engine.tiles import parse_map

MAP = """
#####
#.N.#
#####
"""


def make(x=1, tick=200, satiety=500, hydration=500, energy=500, health=1000, **overrides):
    state = new_state(parse_map(MAP), Config(**overrides), seed=1)
    state.tick = tick  # 200 = full daylight, 1000 = night
    agent = Agent(1, "fly", x, 1, 2, satiety, hydration, energy, health)
    state.agents[1] = agent
    state.occupant[1, x] = 1
    return state, agent


def test_energy_delta_per_action_by_day():
    state, agent = make()
    assert energy_delta(state, agent, Action.IDLE) == -1
    assert energy_delta(state, agent, Action.MOVE_E) == -2
    assert energy_delta(state, agent, Action.CONSUME) == -1
    assert energy_delta(state, agent, Action.REST) == 8


def test_moving_at_night_costs_double_and_nest_rest_gains_more():
    state, agent = make(x=2, tick=1000)
    assert energy_delta(state, agent, Action.MOVE_W) == -4
    assert energy_delta(state, agent, Action.REST) == 12


def test_one_tick_drains_needs_and_ages():
    state, agent = make()
    metabolise(state, agent, Action.MOVE_E, [])
    assert (agent.satiety, agent.hydration, agent.energy, agent.age) == (499, 499, 498, 1)


def test_energy_is_clamped_at_1000():
    state, agent = make(energy=998)
    metabolise(state, agent, Action.REST, [])
    assert agent.energy == 1000


def test_health_regenerates_only_above_the_threshold():
    state, agent = make(health=900, satiety=301, hydration=301, energy=302)
    metabolise(state, agent, Action.IDLE, [])  # needs become 300, 300, 301
    assert agent.health == 901
    metabolise(state, agent, Action.IDLE, [])  # satiety drops to 299
    assert agent.health == 901


def test_depleted_need_emits_once_and_damages_every_tick():
    state, agent = make(satiety=1)
    events = []
    metabolise(state, agent, Action.IDLE, events)
    metabolise(state, agent, Action.IDLE, events)
    assert agent.health == 990
    assert [e.type for e in events] == ["need_depleted", "damaged", "damaged"]
    assert events[0].data == {"need": "satiety"}
    assert events[1].data == {"amount": 5, "causes": ["starvation"]}


def test_damage_stacks_per_depleted_need():
    state, agent = make(satiety=1, hydration=1, energy=1)
    metabolise(state, agent, Action.IDLE, [])
    assert agent.health == 985


def test_death_lists_every_cause_and_frees_the_tile():
    state, agent = make(satiety=1, hydration=1, health=10)
    events = []
    metabolise(state, agent, Action.IDLE, events)
    assert not agent.alive and agent.health == 0
    assert state.occupant[1, 1] == 0
    assert events[-1].type == "died"
    assert events[-1].data == {"causes": ["starvation", "dehydration"]}


def test_old_age_kills_a_healthy_agent():
    state, agent = make(max_age=2)
    events = []
    metabolise(state, agent, Action.IDLE, events)
    assert agent.alive
    metabolise(state, agent, Action.IDLE, events)
    assert not agent.alive
    assert events[-1].data == {"causes": ["old_age"]}


def test_no_max_age_by_default():
    state, agent = make()
    agent.age = 10**9
    metabolise(state, agent, Action.IDLE, [])
    assert agent.alive
