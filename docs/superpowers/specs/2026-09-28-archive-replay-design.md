# NeuroGarden sub-project 4: the archive (persistence + replays)

Date: 2026-09-28. Builds on the architecture spec (2026-09-20), the engine/dojo
spec (2026-09-20), the protocol/server/SDK spec (2026-09-26) and the web client
spec (2026-09-26). Section 9 records what the implementation added or changed.

## 1. Goal

A world that survives its process, and lives that can be watched again.

- **Resume.** `neurogarden serve --archive garden.db` twice is one world, not
  two: the same tick, the same fruit, the same flies (a fly whose owner is away
  waits, idle, for the owner to come back), the same lineage counters, names,
  scores and naturalist's log.
- **Provenance.** Everything the engine was ever told is written down: every
  tick's inputs (spawns, despawns, actions), so any stretch of the world's
  history can be re-run through the engine and checked against stored hashes.
- **Replay.** A life can be watched again — in the terminal or as a *ghost* in
  the browser — and exported as a dataset of (observation, action) pairs. This
  is what sub-project 5 learns from.
- **Corpses leave.** The engine gains `World.despawn(agent_id)`, a logged input,
  so dead flies no longer accumulate in the state (a known gap since
  sub-project 1).

Out of scope: Parquet export (numpy `.npz` is the dataset format for now;
the architecture spec defers the trajectory export format), retention and
compaction of old ticks, multi-world archives, replays over the SDK.

## 2. Package layout

```text
src/neurogarden/
  engine/world.py        + World.despawn, World.restore(data, map_text)
  dojo/stats.py          + StatsTracker.to_dict / from_dict
  server/archive.py      SQLite archive: schema, writes, reads, identity   (new)
  server/history.py      rebuild a world from the archive; playback        (new)
  server/runner.py       writes through the archive; corpses; snapshots
  server/roster.py       Roster.restore(lives); best_lineage
  server/gateway.py      ghost replays for spectators
  server/app.py          ServerConfig.archive; resume or create
  protocol/messages.py   replay (client → server), replay (server → client)
  cli.py                 --archive; lives, replay, export, verify
web/src/                 ghost mode (?ghost=owner/lineage)
```

Storage is SQLite from the standard library (`sqlite3`), one file per world,
WAL journal. No new dependency. A path of `:memory:` is a world that lives
only as long as its process — the default for `ServerConfig`, so tests and
throwaway worlds leave nothing behind; the CLI defaults to `neurogarden.db`.

## 3. The engine: despawn

`World.despawn(agent_id)` removes a **dead** agent from `state.agents`. It is
an input like `spawn`: pure, deterministic, and recorded by whoever drives the
world. Despawning a living or unknown agent raises `ValueError`. `occupant` is
already cleared at death, so despawn changes nothing on the map; it changes the
state hash (fewer agents) and shrinks snapshots. `RULES_VERSION` stays 2: no
existing replay changes meaning.

`World.restore(data, map_text=None)` keeps the map text with the restored
world, so a resumed world can still describe itself.

## 4. The archive

### 4.1 Tables

```sql
world      (id=1, archive_version, map_name, map_text, seed, config, rules_version, created_at)
ticks      (tick PRIMARY KEY, inputs)          -- inputs applied to go from tick to tick+1
lives      (agent_id PRIMARY KEY, owner, lineage, name, body, born_tick,
            died_tick, causes, stats, UNIQUE(owner, lineage))
chronicle  (tick, text)
checkpoints(tick PRIMARY KEY, state_hash)      -- every checkpoint_every ticks
snapshots  (tick PRIMARY KEY, state, extras)   -- tick 0, then every snapshot_every ticks, and at shutdown
```

`ticks.inputs` is JSON: `{"despawns": [id, ...], "spawns": [[id, body, x, y], ...],
"actions": {"id": action, ...}}` — applied in that order, then `step`. Spawns
carry their tile, so a replay never depends on the spawn rule of the day.

`snapshots.state` is the engine snapshot (`World.snapshot()`, `rng_state` as a
decimal string). `snapshots.extras` is the runner's sidecar: the stats trackers
of living flies and the chronicler's meal memory, so a resumed fly's obituary
counts the bites it took before the restart.

`world.seed` is informational (identity checks and banners); the world is
*defined* by its tick-0 snapshot, so a world that arrived from a snapshot
archives just as well as one built from a map.

### 4.2 Writes

One transaction per tick, committed at the end of `WorldRunner.tick()`. A tick
that raises is rolled back and the world stops (as before). At 5 ticks per
second a commit in WAL mode costs about a millisecond.

- `record_tick(tick, spawns, despawns, actions)`
- `born(agent_id, owner, lineage, name, body, born_tick)` /
  `died(agent_id, died_tick, causes, stats)`
- `note(tick, text)` for every chronicle line
- `checkpoint(tick, state_hash)` every `checkpoint_every` (100) ticks
- `save_snapshot(tick, state, extras)` at tick 0, every `snapshot_every`
  (600) ticks, and once more when the server shuts down cleanly, so a resume
  replays nothing.

### 4.3 Reads

`world_info`, `next_tick()` (the number of recorded ticks), `lives(owner=None)`,
`life(owner, lineage)`, `recent_chronicle(n)`, `latest_snapshot(at_or_before)`,
`inputs(from_tick, to_tick)` in batches (a reader never holds a cursor open
across an `await`, so the runner may write between batches), `checkpoints()`.

### 4.4 Identity

An archive holds one world. Opening it for serving checks, in this order:
`archive_version` is the one this code writes; `rules_version` matches the
engine (else: "this archive was written under rules_version 1; the engine is 2
— start a new archive"); the requested map and seed match, when the caller
gave any. The CLI reads the archive first and takes map and seed from it
unless `--map`/`--seed` were passed explicitly, so `neurogarden serve` alone
resumes whatever `neurogarden.db` holds.

## 5. Rebuild and playback (`server/history.py`)

`rebuild(archive, upto=None)` returns the world at tick `upto` (default: the
next tick to run) plus the trackers: restore the latest snapshot at or before
the target, then apply the recorded inputs of every tick after it, updating
trackers exactly as the runner does. Cost: at most `snapshot_every` steps
after a crash, zero after a clean shutdown.

`playback(archive, from_tick, to_tick)` yields one `Moment(tick, world,
events, observations)` per recorded tick in `[from_tick, to_tick)`, on its own
`World` (the live world is never touched). Used by the terminal replay, the
ghost stream, the dataset export and `verify`.

`WorldRunner.from_archive(archive, ...)` = rebuild + `Roster.restore(lives)` +
the last 30 chronicle lines. Not restored, by design: missed counters (reset
to 0), speech bubbles, connections (every fly starts the resumed world idle
and away; its owner's next `hello` + `join` reattaches it, "back at the
controls").

## 6. Ghost replays over the protocol

Client → server `replay` (spectators only; an agent gets a non-fatal error):

```json
{"owner": "alice", "lineage": 2, "speed": 4.0}
```

`speed` in `[0.25, 64]`, ticks per second relative to the world's `tps`. The
server answers with `world` (the map), then

```json
{"type": "replay", "payload": {"owner": "alice", "lineage": 2, "name": "Dusty Wing",
  "born_tick": 1200, "died_tick": 4211, "lifespan": 3012, "causes": ["starvation"],
  "speed": 4.0, "done": false}}
```

then one `frame` per recorded tick of that life, then `replay` again with
`done: true`. A ghost is a detour: while it plays the spectator receives no
live frames, and when it is over (played out, failed, or refused with
`error(no_such_life)`) the spectator is back on the live stream, starting with
`world` and the recent chronicle as for a new spectator. A new `replay`
replaces the running one. Ghost frames are ordinary frames: names and lineages
come from the archive, every fly counts as connected, `scores` is empty, `say`
is empty (bubbles are not archived). A life still going when asked for plays
up to what the archive holds and is then `done` too.

## 7. CLI

```text
neurogarden serve  [--archive PATH|:memory:] [--map M] [--seed S] ...
neurogarden lives   --archive PATH [--owner O]            the hall of flies
neurogarden replay  --archive PATH --owner O --life N [--speed X] [--ascii] [--frames N]
neurogarden export  --archive PATH --owner O --life N --out FILE.npz
neurogarden verify  --archive PATH                        re-run history, check every hash
```

The serve banner says whether the world is new or resumed ("resuming
drosoville (seed 0) at tick 12345 — day 11, 7 lives so far").

`export` writes an `.npz` with one array per sense channel (stacked along a
leading time axis), `actions`, `ticks`, and `meta` (a JSON string: owner,
lineage, name, born/died ticks, causes, rules_version, the body's catalog
entry). The observations are regenerated through the engine — nothing but
actions is stored — so the file is exactly what the brain saw.

## 8. Web: ghosts

`?ghost=alice/2[&speed=4]` opens the page as a ghost viewer: spectator socket
only, `replay` sent after `welcome`, play panel hidden, a banner "👻 alice's
Dusty Wing #2 · 4× · tick 1300 of 3012", the ghost drawn as the highlighted
fly. When the ghost has faded the banner offers the way back to the living
garden. Each row of the hall of flies links to its owner's best life (the
frame's `scores[]` gains `best_lineage`, additive).

## 9. Testing

- Engine: despawn semantics; a despawned world hashes differently and
  restores identically.
- Archive: schema and identity checks; `:memory:` works; a tick's inputs
  round-trip; lives and chronicle round-trip.
- The resume guard: run a runner on a file for N ticks with two flies and a
  death, close it, `from_archive`, run both the interrupted and an uninterrupted
  twin for M more ticks with the same actions — same state hash, same stats,
  same roster and scores. Also after a crash (no shutdown snapshot).
- Corpses leave the state the tick after they die; the nest is free again.
- `verify` passes on a real archive and fails when a tick's actions are edited.
- Ghost stream over real sockets: `world`, `replay`, the right number of
  frames, `replay(done)`; a bad life answers `no_such_life`; an agent may not.
- CLI: `lives`, `replay --frames`, `export` array shapes, `serve` resumes and
  refuses a wrong seed with one line.
- Web (vitest): ghost URL parsing, `replay` in the view model, banner text.

## 10. What the implementation added (and where it deviates)

Additions:

- A ghost also gets the naturalist's lines of those days: the archived
  `chronicle` rows stamped with each replayed tick are sent after that tick's
  frame, so the log on the ghost page tells the story as it happened.
- `verify` checks every snapshot too, not only the checkpoints: each stored
  snapshot must hash to what the re-run has at that tick.
- `Archive.peek(path)` reads a file's identity without creating it; the CLI
  uses it so `serve` resumes whatever the archive holds.
- `WorldRunner.broken`: a tick that raised leaves the world in memory
  half-changed, so `save_snapshot()` refuses to write it down.
- `OwnerScore.best_lineage` (additive) so the hall of flies can link to the
  right life.
- From the reviews: one writer per archive file (`<path>.lock`, an advisory
  lock held while the archive is open; a second `serve` on the same file is
  refused); one ghost a second per spectator (`error(replay_busy)`), since
  starting one rebuilds a world; the rebuild of a ghost's starting world
  yields to the live tick loop every 50 engine steps, and a ghost frame is
  only built when the spectator has read the previous one (latest wins
  anyway); the bound of a ghost is fixed when asked (`Archive.life_end`), so
  a life still going plays up to what was archived and then is `done`;
  `Replay.lineage` has an upper bound; the chronicler's meal memory is kept
  during the rebuild after a crash, so the resumed log matches the
  uninterrupted one; `verify` is a consistency check from the first
  snapshot, not tamper evidence (the genesis is trusted as it stands).

Deviations:

- The runner's `spawn_log` / `action_log` lists are gone rather than kept
  beside the archive; tests read `archive.inputs(0)` and `archive.lives()`.
- `ServerConfig.archive` defaults to `:memory:` and the *CLI* defaults to
  `neurogarden.db` (§2 said the file is the default; it is, for people).
- `neurogarden replay` paces itself with `--fps` (default 20) rather than a
  multiple of a `tps` the archive does not record.
- A ghost's `world_at` replays from the nearest snapshot on the event loop
  (synchronously, at most `snapshot_every` steps, tens of milliseconds); a
  world with many simultaneous ghost watchers would want a thread and its
  own read-only connection.
- Corpses are cleared by scanning the state for dead agents at the start of
  each tick, not by a list the runner keeps: a resumed world (or one that
  arrived with corpses) needs no special case.
