from neurogarden.engine import Config
from neurogarden.engine.events import Event
from neurogarden.server.chronicle import Chronicler, Subject, sector, stamp
from neurogarden.server.names import fly_name, mood
from neurogarden.server.roster import Roster


class FakePort:
    role = "agent"

    def __init__(self, owner):
        self.owner = owner
        self.closed = None

    def deliver(self, message):
        pass

    def close(self, code, reason):
        self.closed = (code, reason)


def test_fly_names_are_deterministic_two_words_and_vary_by_lineage():
    assert fly_name("alice", 1) == fly_name("alice", 1)
    assert fly_name("alice", 1) != fly_name("alice", 2)
    assert fly_name("alice", 1) != fly_name("bob", 1)
    assert len(fly_name("alice", 3).split()) == 2


def test_mood_reads_the_lowest_need():
    assert mood(900, 900, 900, 1000, True) == "content"
    assert mood(500, 900, 900, 1000, True) == "hungry"
    assert mood(900, 400, 900, 1000, True) == "thirsty"
    assert mood(900, 900, 590, 1000, True) == "sleepy"
    assert mood(300, 900, 900, 1000, True) == "desperate"
    assert mood(100, 900, 900, 1000, True) == "dying"
    assert mood(900, 900, 900, 200, True) == "dying"
    assert mood(900, 900, 900, 0, False) == "dead"


def test_roster_one_live_fly_per_owner_with_lineage_and_names():
    roster = Roster()
    alice = FakePort("alice")
    assert roster.attach(alice) is None
    assert roster.live_agent("alice") is None
    lineage, name = roster.born("alice", 7)
    assert (lineage, name) == (1, fly_name("alice", 1))
    assert roster.live_agent("alice") == 7 and roster.owner_of(7) == "alice"
    assert roster.port_for(7) is alice and roster.connected(7)
    state = roster.died(7, lifespan=1234)
    assert state.agent_id is None and state.best_lifespan == 1234 and state.lifespans == [1234]
    assert roster.born("alice", 9) == (2, fly_name("alice", 2))


def test_roster_supersede_and_detach():
    roster = Roster()
    old, new = FakePort("alice"), FakePort("alice")
    roster.attach(old)
    roster.born("alice", 1)
    assert roster.attach(new) is old
    assert roster.port_for(1) is new
    roster.detach(old)  # a stale detach of the superseded port changes nothing
    assert roster.port_for(1) is new
    roster.detach(new)
    assert roster.port_for(1) is None and not roster.connected(1)
    assert roster.attach(new) is None  # re-attaching the same port supersedes nobody


def test_scores_rank_best_lifespan_then_name():
    roster = Roster()
    for owner, agent_id, lifespan in (("bob", 1, 50), ("alice", 2, 500), ("carol", 3, 500)):
        roster.attach(FakePort(owner))
        roster.born(owner, agent_id)
        roster.died(agent_id, lifespan)
    assert [s.owner for s in roster.scores()] == ["alice", "carol", "bob"]


def test_sector_and_stamp():
    assert sector(0, 0, 32, 24) == "the north-west"
    assert sector(16, 12, 32, 24) == "the centre"
    assert sector(31, 23, 32, 24) == "the south-east"
    cfg = Config()
    assert stamp(0, cfg) == "Day 1, dawn"
    assert stamp(300, cfg) == "Day 1, day"
    assert stamp(750, cfg) == "Day 1, dusk"
    assert stamp(1200 + 900, cfg) == "Day 2, night"


def test_chronicler_writes_births_meals_depletion_and_deaths_but_not_every_bite():
    chronicler = Chronicler(Config(), 32, 24)
    dusty = Subject("alice", "Dusty Wing", 30, 2)
    subjects = {1: dusty}
    assert chronicler.born(0, dusty, 1) == "Day 1, dawn: alice's Dusty Wing hatches at the nest."
    assert "(life #2)" in chronicler.born(0, dusty, 2)
    ate = Event(300, "ate", 1, {"bites_left": 2})
    first = chronicler.lines(300, [ate, ate], subjects)
    assert first == ["Day 1, day: Dusty Wing (alice) finds fruit in the north-east."]
    assert chronicler.lines(310, [ate], subjects) == []  # same meal
    assert chronicler.lines(400, [ate], subjects) != []  # a new meal later
    died = Event(500, "died", 1, {"causes": ["starvation", "old_age"]})
    empty = Event(500, "need_depleted", 1, {"need": "hydration"})
    fruit = Event(500, "fruit_spawned", None, {"x": 1, "y": 1})
    lines = chronicler.lines(500, [empty, died, fruit], subjects)
    assert lines == [
        "Day 1, day: Dusty Wing (alice) has run out of hydration.",
        "Day 1, day: Dusty Wing (alice) dies of starvation, old_age in the north-east.",
    ]
    assert chronicler.lines(500, [Event(500, "ate", 2, {})], subjects) == []  # unknown agent
