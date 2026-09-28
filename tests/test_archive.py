"""The archive: what the runner writes, what history reads back, and the resume guard."""

import json

import pytest

from neurogarden.engine import Action, Config, World, maps
from neurogarden.engine.rng import SplitMix64
from neurogarden.server import Server, ServerConfig, WorldRunner, history
from neurogarden.server.archive import Archive, ArchiveError, TickInputs
from neurogarden.server.roster import GhostRoster, Roster
from test_runner import TINY, FakePort, joined_agent

SHORT = Config(initial_satiety=15, initial_hydration=15, initial_health=5)  # dead in ~16 ticks


def new_runner(archive=None, config=SHORT, seed=1, **options):
    world = World.from_map(TINY, config, seed=seed)
    if archive is not None and archive.world_info is None:
        archive.create_world("tiny", world.map_text, seed, config, world.snapshot())
    return WorldRunner(world, archive=archive, tps=10.0, map_name="tiny", **options)


def drive(runner, ports, steps, seed):
    """Random actions for every port with a fly, `steps` ticks; the same seed drives twins."""
    rng = SplitMix64(seed)
    for _ in range(steps):
        for port in ports:
            if port.of("observation") and runner.roster.live_agent(port.owner) is not None:
                runner.submit_action(port, port.last("observation").tick, rng.randbelow(7))
        runner.tick()


# --- the archive itself ------------------------------------------------------------------


def test_tick_inputs_round_trip_through_json():
    inputs = TickInputs(7, [(3, "fly", 2, 1)], [1, 2], {3: 4, 5: 0})
    assert TickInputs.from_json(7, inputs.to_json()) == inputs


def test_a_memory_archive_records_a_world_its_ticks_lives_and_log():
    runner = new_runner()
    archive = runner.archive
    assert archive.world_info is not None and archive.world_info.seed is None
    assert archive.next_tick() == 0 and archive.snapshot_ticks() == [0]
    alice = joined_agent(runner)
    runner.submit_action(alice, 0, int(Action.MOVE_E))
    runner.tick()
    recorded = list(archive.inputs(0))
    assert [r.tick for r in recorded] == [0, 1]
    assert recorded[0].spawns == [(1, "fly", 2, 2)] and recorded[0].actions == {1: 0}
    assert recorded[1].actions == {1: int(Action.MOVE_E)}
    (life,) = archive.lives()
    assert (life.owner, life.lineage, life.born_tick, life.alive) == ("alice", 1, 0, True)
    assert archive.life("alice", 1) == life and archive.life("alice", 2) is None
    assert archive.next_tick() == 2 and archive.tick_count() == 2
    assert [text for _, text in archive.recent_chronicle(5)] == [t for _, t in runner.chronicle]
    assert "hatches" in archive.recent_chronicle(1)[0][1]


def test_a_death_is_archived_with_its_stats_and_the_corpse_leaves_next_tick():
    runner = new_runner(config=Config(initial_satiety=2, initial_health=5))
    alice = joined_agent(runner)
    runner.tick()
    died = alice.last("died")
    life = archive_life = runner.archive.life("alice", 1)
    assert archive_life.died_tick == died.tick and life.causes == ("starvation",)
    assert life.stats.lifespan == died.stats.lifespan and life.lifespan == life.stats.lifespan
    assert 1 in runner.world.state.agents  # the corpse is in the frame of the tick it died
    runner.tick()
    assert 1 not in runner.world.state.agents
    assert list(runner.archive.inputs(0))[-1].despawns == [1]
    assert runner.roster.state("alice").best_lineage == 1


def test_snapshots_and_checkpoints_land_on_schedule(tmp_path):
    runner = new_runner(snapshot_every=4, checkpoint_every=3)
    joined_agent(runner)
    for _ in range(9):
        runner.tick()  # world at tick 10
    archive = runner.archive
    assert archive.snapshot_ticks() == [0, 4, 8]
    assert sorted(archive.checkpoints()) == [3, 6, 9]
    assert archive.checkpoints()[9] != archive.checkpoints()[6]
    tick, state, extras = archive.latest_snapshot(9)
    assert tick == 8 and state["tick"] == 8 and "1" in extras["trackers"]
    runner.save_snapshot()
    assert archive.snapshot_ticks()[-1] == 10


def test_a_transaction_rolls_back_a_tick_that_raises(monkeypatch):
    runner = new_runner()
    joined_agent(runner)

    def boom(*args):
        raise RuntimeError("the sky fell")

    monkeypatch.setattr(runner, "_collect_actions", boom)
    with pytest.raises(RuntimeError):
        runner.tick()
    assert runner.archive.tick_count() == 1 and runner.broken
    runner.save_snapshot()  # a broken world is never written down
    assert runner.archive.snapshot_ticks() == [0]


def test_a_file_archive_is_reopened_and_foreign_ones_are_refused(tmp_path):
    path = str(tmp_path / "garden.db")
    assert Archive.peek(path) is None
    runner = new_runner(Archive.open(path), seed=7)
    joined_agent(runner)
    runner.archive.close()
    held = Archive.peek(path)
    assert (held.map_name, held.seed, held.config) == ("tiny", 7, SHORT)
    again = Archive.open(path, readonly=True)
    assert again.tick_count() == 1 and again.lives()[0].owner == "alice"
    again.close()
    with pytest.raises(ArchiveError, match="no archive"):
        Archive.open(str(tmp_path / "missing.db"), readonly=True)
    (tmp_path / "notes.db").write_text("not sqlite at all")
    with pytest.raises(ArchiveError, match="not a NeuroGarden archive"):
        Archive.peek(str(tmp_path / "notes.db"))
    with Archive.open(path)._db as db:
        db.execute("UPDATE world SET archive_version = 99")
    with pytest.raises(ArchiveError, match="version 99"):
        Archive.open(path)


# --- history: rebuild, playback, verify ------------------------------------------------------


def test_playback_relives_recorded_ticks_on_a_world_of_its_own():
    runner = new_runner()
    alice = joined_agent(runner)
    drive(runner, [alice], 20, seed=3)
    moments = list(history.playback(runner.archive, 5, 15))
    assert [m.tick for m in moments] == list(range(5, 15))
    assert moments[-1].world.tick == 15 and moments[-1].world is not runner.world
    assert all(1 in m.result.observations for m in moments)
    whole = list(history.playback(runner.archive, 0))
    assert whole[-1].world.state_hash() == runner.world.state_hash()


def test_verify_reproduces_every_hash_and_notices_an_edited_action():
    runner = new_runner(snapshot_every=5, checkpoint_every=4)
    alice = joined_agent(runner)
    drive(runner, [alice], 22, seed=4)
    report = history.verify(runner.archive)
    assert (report.ticks, report.checkpoints, report.snapshots) == (23, 5, 4)
    db = runner.archive._db
    (text,) = db.execute("SELECT inputs FROM ticks WHERE tick = 7").fetchone()
    edited = json.loads(text)
    edited["actions"]["1"] = (edited["actions"]["1"] + 1) % 7
    db.execute("UPDATE ticks SET inputs = ? WHERE tick = 7", (json.dumps(edited),))
    with pytest.raises(ArchiveError, match="tick 8 does not reproduce"):
        history.verify(runner.archive)


def test_world_at_refuses_ticks_the_archive_does_not_reach():
    runner = new_runner()
    joined_agent(runner)
    with pytest.raises(ArchiveError, match="ends at tick 1"):
        history.world_at(runner.archive, 5)
    with pytest.raises(ArchiveError, match="hatched agent 9"):
        history.apply_inputs(
            World.from_map(TINY, SHORT, seed=1), TickInputs(0, [(9, "fly", 2, 2)], [], {})
        )


# --- the resume guard --------------------------------------------------------------------------


def twins(tmp_path, **options):
    """Two runners fed the same story: one on a file, one that never stops."""
    path = str(tmp_path / "garden.db")
    interrupted = new_runner(Archive.open(path), **options)
    straight = new_runner(**options)
    return path, interrupted, straight


def same_story(runners, steps, seed, owners=("alice", "bob")):
    ports = {}
    for runner in runners:
        ports[runner] = [joined_agent(runner, owner) for owner in owners]
    for runner in runners:
        drive(runner, ports[runner], steps, seed)
    return ports


def assert_same_world(a, b):
    assert a.world.tick == b.world.tick
    assert a.world.state_hash() == b.world.state_hash()
    assert list(a.chronicle) == list(b.chronicle)
    for owner in a.roster.owners:
        x, y = a.roster.state(owner), b.roster.state(owner)
        assert (x.lineage, x.agent_id, x.name) == (y.lineage, y.agent_id, y.name)
        assert (x.best_lifespan, x.best_lineage, x.lifespans) == (
            y.best_lifespan,
            y.best_lineage,
            y.lifespans,
        )
    assert a.roster.agents == b.roster.agents
    assert {k: t.stats for k, t in a._trackers.items()} == {
        k: t.stats for k, t in b._trackers.items()
    }
    assert a._chronicler.last_meal == b._chronicler.last_meal


@pytest.mark.parametrize("clean_stop", [True, False])
def test_a_resumed_world_equals_the_uninterrupted_twin(tmp_path, clean_stop):
    path, interrupted, straight = twins(tmp_path, snapshot_every=7, checkpoint_every=5)
    same_story([interrupted, straight], 30, seed=11)
    assert any(not life.alive for life in interrupted.archive.lives())  # somebody starved
    if clean_stop:
        interrupted.save_snapshot()
    interrupted.archive.close()

    archive = Archive.open(path)
    if clean_stop:
        assert archive.snapshot_ticks()[-1] == archive.next_tick()  # nothing to replay
    resumed = WorldRunner.from_archive(archive, tps=10.0)
    assert_same_world(resumed, straight)
    assert resumed.missed == {a: 0 for a in resumed._trackers}

    # the story goes on: every owner comes back and both worlds get the same actions
    ports = {r: [FakePort("alice"), FakePort("bob")] for r in (resumed, straight)}
    for runner, its in ports.items():
        for port in its:
            runner.attach(port)
            runner.request_join(port)
        runner.tick()
    hatched = [[p.last("joined") for p in its] for its in ports.values()]
    assert hatched[0] == hatched[1] and all(j.lineage == 2 for j in hatched[0])
    for runner, its in ports.items():
        drive(runner, its, 40, seed=12)
    assert_same_world(resumed, straight)
    assert history.verify(archive).ticks == 73  # 2 hatchings + 30 + 1 rejoin tick + 40


def test_a_returning_owner_finds_the_fly_that_waited_through_the_restart(tmp_path):
    path, interrupted, straight = twins(tmp_path, config=Config())
    ports = same_story([interrupted, straight], 5, seed=1, owners=("alice",))
    fly = interrupted.roster.live_agent("alice")
    interrupted.archive.close()
    resumed = WorldRunner.from_archive(Archive.open(path), tps=10.0)
    assert resumed.roster.live_agent("alice") == fly and not resumed.roster.connected(fly)
    watcher = FakePort("w", role="spectator")
    resumed.add_spectator(watcher)
    resumed.tick()  # the fly idles, away, and is still in the frame
    (shown,) = watcher.last("frame").agents
    assert (shown.agent_id, shown.owner, shown.connected) == (fly, "alice", False)
    alice = FakePort("alice")
    resumed.attach(alice)
    resumed.request_join(alice)
    resumed.tick()
    assert alice.last("joined").reattached and alice.last("joined").agent_id == fly
    assert any("back at the controls" in t for _, t in resumed.chronicle)
    assert ports  # the twin only served to tell the story


def test_roster_restore_rebuilds_lineages_scores_and_the_live_fly():
    runner = new_runner(config=Config(initial_satiety=6, initial_health=5))
    alice = joined_agent(runner)
    for _ in range(8):
        runner.tick()
    assert alice.of("died")
    runner.request_join(alice)
    runner.tick()
    restored = Roster()
    restored.restore(runner.archive.lives())
    state = restored.state("alice")
    original = runner.roster.state("alice")
    assert (state.lineage, state.agent_id, state.name) == (2, original.agent_id, original.name)
    assert (state.best_lifespan, state.best_lineage, state.lifespans) == (
        original.best_lifespan,
        1,
        original.lifespans,
    )
    ghost = GhostRoster(runner.archive.lives())
    assert ghost.connected(1) and ghost.scores() == [] and ghost.record(1).lineage == 1


# --- through the server ------------------------------------------------------------------------


def test_the_server_resumes_its_archive_and_the_npc_keeps_its_fly(tmp_path):
    import asyncio

    path = str(tmp_path / "garden.db")
    settings = dict(port=0, tps=50.0, npcs=[("scripted", 1)], archive=path)

    async def first_run():
        async with Server(ServerConfig(**settings)) as server:
            await asyncio.sleep(0.2)
        runner = server.runner  # read once the ticker has stopped
        return runner.world.tick, runner.roster.live_agent("npc-scripted-1"), server.resumed

    async def second_run():
        async with Server(ServerConfig(**settings)) as server:
            runner = server.runner
            at_start = runner.world.tick
            await asyncio.sleep(0.1)
            npc = runner.roster.state("npc-scripted-1")
            back = [t for _, t in runner.chronicle if "back at the controls" in t]
            return at_start, npc.agent_id, npc.lineage, server.resumed, back

    ticks, fly, resumed = asyncio.run(first_run())
    assert ticks > 3 and fly is not None and not resumed
    at_start, same_fly, lineage, resumed, back = asyncio.run(second_run())
    assert resumed and at_start == ticks and same_fly == fly and lineage == 1 and back
    with pytest.raises(ArchiveError, match="holds tiny|holds drosoville"):
        Server(ServerConfig(**{**settings, "seed": 5}))
    with pytest.raises(ArchiveError, match="other engine settings"):
        Server(ServerConfig(**{**settings, "config": Config(day_length=2400)}))


def test_a_full_map_replay_is_deterministic_through_history():
    runner = WorldRunner(World.from_map(maps.load("drosoville"), Config(), seed=1), tps=10.0)
    alice, bob = joined_agent(runner, "alice"), joined_agent(runner, "bob")
    drive(runner, [alice, bob], 30, seed=9)
    assert history.rebuild(runner.archive).world.state_hash() == runner.world.state_hash()


# --- what the review asked for ---------------------------------------------------------------


def test_a_bare_runner_archive_has_no_seed_and_the_server_still_serves_it(tmp_path):
    path = str(tmp_path / "bare.db")
    runner = WorldRunner(World.from_map(maps.load("drosoville")), archive=Archive.open(path))
    joined_agent(runner, "carol")
    runner.archive.close()
    assert Archive.peek(path).seed is None

    async def resume():
        async with Server(ServerConfig(port=0, tps=50.0, npcs=[], archive=path)) as server:
            return server.resumed, server.runner.roster.live_agent("carol")

    import asyncio

    assert asyncio.run(resume()) == (True, 1)


def test_one_writer_per_archive_file(tmp_path):
    pytest.importorskip("fcntl")
    path = str(tmp_path / "garden.db")
    first = Archive.open(path)
    new_runner(first)  # a world, so a reader has something to read
    with pytest.raises(ArchiveError, match="already being served"):
        Archive.open(path)
    reader = Archive.open(path, readonly=True)  # readers are welcome meanwhile
    assert reader.world_info is not None
    reader.close()
    first.close()
    second = Archive.open(path)  # the lock went with the first
    second.close()


def test_a_path_with_uri_characters_opens_read_only(tmp_path):
    path = str(tmp_path / "world #1 100%.db")
    runner = new_runner(Archive.open(path))
    joined_agent(runner)
    runner.archive.close()
    reader = Archive.open(path, readonly=True)
    assert reader.lives()[0].owner == "alice"
    reader.close()


def test_a_missing_directory_is_refused_not_a_traceback(tmp_path):
    with pytest.raises(ArchiveError, match="cannot open"):
        Archive.open(str(tmp_path / "no" / "such" / "dir" / "garden.db"))


def test_the_bound_of_a_life_still_going_is_fixed_when_asked():
    runner = new_runner(config=Config())
    alice = joined_agent(runner)
    for _ in range(4):
        runner.tick()
    life = runner.archive.life("alice", 1)
    assert life.alive and runner.archive.life_end(life) == 5
    runner.tick()
    assert runner.archive.life_end(life) == 6  # what is archived now, not chased later
    assert [m.tick for m in history.playback(runner.archive, 0, 5)] == [0, 1, 2, 3, 4]
    assert alice.of("observation")


@pytest.mark.parametrize("snapshot_every", [1000, 2])
def test_a_rebuilt_world_remembers_meals_like_the_chronicler(snapshot_every):
    """The fruit tile is at (3, 1), one step east and one north of the nest."""
    runner = new_runner(config=Config(), snapshot_every=snapshot_every)
    alice = joined_agent(runner)
    for action in (Action.MOVE_E, Action.MOVE_N, Action.CONSUME, Action.CONSUME, Action.IDLE):
        runner.submit_action(alice, alice.last("observation").tick, int(action))
        runner.tick()
    bites = sum(
        1
        for m in alice.inbox
        if m.type == "observation"
        for e in m.payload.events
        if e.type == "ate"
    )
    assert bites == 2  # two bites of the same meal: one line, one memory
    assert runner._chronicler.last_meal == {1: 3}
    assert sum("finds fruit" in text for _, text in runner.chronicle) == 1
    rebuilt = history.rebuild(runner.archive)
    assert rebuilt.last_meal == runner._chronicler.last_meal
    assert rebuilt.world.state_hash() == runner.world.state_hash()


def test_an_archive_that_died_before_its_world_row_is_a_fresh_one(tmp_path):
    """serve creates the file and its tables before the world row: a crash in between must
    not leave a file that can neither be resumed nor started."""
    path = str(tmp_path / "garden.db")
    Archive.open(path).close()  # the schema is there, the world is not
    assert Archive.peek(path) is None
    runner = new_runner(Archive.open(path), seed=2)
    assert runner.archive.world_info.seed == 2
    runner.archive.close()
    assert Archive.peek(path).seed == 2
