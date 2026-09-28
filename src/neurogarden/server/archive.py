"""The archive: everything the engine was ever told, in one SQLite file per world.

Ticks hold the engine's inputs (despawns, spawns with their tiles, actions); lives hold
who the flies were; snapshots make resuming and replaying cheap; checkpoints make the
whole history checkable. Nothing here decides anything — the runner writes, the
history module reads, the engine stays pure.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from neurogarden.dojo.stats import EpisodeStats
from neurogarden.engine.config import RULES_VERSION, Config

try:
    import fcntl  # POSIX: one writer per archive file (see Archive.open)
except ImportError:  # pragma: no cover - Windows has no fcntl; two servers on one file collide
    fcntl = None

ARCHIVE_VERSION = 1
MEMORY = ":memory:"  # a world that lives only as long as its process
_BATCH = 256  # ticks read per query, so no cursor is ever held open across an await

_SCHEMA = """
CREATE TABLE IF NOT EXISTS world (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    archive_version INTEGER NOT NULL,
    map_name TEXT NOT NULL,
    map_text TEXT NOT NULL,
    seed INTEGER,
    config TEXT NOT NULL,
    rules_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ticks (tick INTEGER PRIMARY KEY, inputs TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lives (
    agent_id INTEGER PRIMARY KEY,
    owner TEXT NOT NULL,
    lineage INTEGER NOT NULL,
    name TEXT NOT NULL,
    body TEXT NOT NULL,
    born_tick INTEGER NOT NULL,
    died_tick INTEGER,
    causes TEXT,
    stats TEXT,
    UNIQUE (owner, lineage)
);
CREATE TABLE IF NOT EXISTS chronicle (tick INTEGER NOT NULL, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chronicle_tick ON chronicle (tick);
CREATE TABLE IF NOT EXISTS checkpoints (tick INTEGER PRIMARY KEY, state_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots (
    tick INTEGER PRIMARY KEY, state TEXT NOT NULL, extras TEXT NOT NULL
);
"""


class ArchiveError(ValueError):
    """The archive cannot be used as asked: missing, foreign, or inconsistent."""


@dataclass(frozen=True)
class WorldInfo:
    map_name: str
    map_text: str
    seed: int | None
    config: Config
    rules_version: int
    archive_version: int
    created_at: str


@dataclass(frozen=True)
class Life:
    agent_id: int
    owner: str
    lineage: int
    name: str
    body: str
    born_tick: int
    died_tick: int | None = None
    causes: tuple[str, ...] = ()
    stats: EpisodeStats | None = None

    @property
    def alive(self) -> bool:
        return self.died_tick is None

    @property
    def lifespan(self) -> int | None:
        """Ticks lived, once the life is over."""
        return None if self.stats is None else self.stats.lifespan


@dataclass(frozen=True)
class TickInputs:
    """What the engine was told at one tick, in the order it was told: despawn, spawn, step."""

    tick: int
    spawns: list[tuple[int, str, int, int]]  # agent_id, body, x, y
    despawns: list[int]
    actions: dict[int, int]

    def to_json(self) -> str:
        return json.dumps(
            {
                "despawns": list(self.despawns),
                "spawns": [list(spawn) for spawn in self.spawns],
                "actions": {str(agent_id): action for agent_id, action in self.actions.items()},
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, tick: int, text: str) -> TickInputs:
        raw = json.loads(text)
        return cls(
            tick=tick,
            spawns=[(int(a), str(b), int(x), int(y)) for a, b, x, y in raw["spawns"]],
            despawns=[int(a) for a in raw["despawns"]],
            actions={int(a): int(action) for a, action in raw["actions"].items()},
        )


def _stats_from_json(text: str | None) -> EpisodeStats | None:
    return None if text is None else EpisodeStats.from_dict(json.loads(text))


def _life(row) -> Life:
    agent_id, owner, lineage, name, body, born, died, causes, stats = row
    return Life(
        agent_id=agent_id,
        owner=owner,
        lineage=lineage,
        name=name,
        body=body,
        born_tick=born,
        died_tick=died,
        causes=tuple(json.loads(causes)) if causes else (),
        stats=_stats_from_json(stats),
    )


_LIFE_COLUMNS = "agent_id, owner, lineage, name, body, born_tick, died_tick, causes, stats"


def _lock(path: str):
    """Hold `<path>.lock` for as long as the archive is open: a second server on the same file
    would resume the first one's world and collide with it tick by tick."""
    if fcntl is None:
        return None
    try:
        lock = open(f"{path}.lock", "w")  # noqa: SIM115 - held until Archive.close
    except OSError as err:  # no such directory, no permission
        raise ArchiveError(f"cannot open {path}: {err.strerror}") from None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise ArchiveError(f"{path} is already being served by another neurogarden") from None
    return lock


class Archive:
    def __init__(self, connection: sqlite3.Connection, path: str, lock=None) -> None:
        self._db = connection
        self._lock = lock
        self.path = path

    # --- opening ----------------------------------------------------------------------

    @classmethod
    def open(cls, path: str = MEMORY, *, readonly: bool = False) -> Archive:
        """Open (and, unless read-only, create) the archive at `path`; `:memory:` for none."""
        lock = None
        try:
            if readonly:
                if path == MEMORY or not os.path.exists(path):
                    raise ArchiveError(f"no archive at {path}")
                uri = f"{Path(path).resolve().as_uri()}?mode=ro"
                db = sqlite3.connect(uri, uri=True, isolation_level=None)
            else:
                lock = _lock(path) if path != MEMORY else None
                db = sqlite3.connect(path, isolation_level=None)  # explicit BEGIN/COMMIT below
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=NORMAL")
                db.executescript(_SCHEMA)
        except sqlite3.Error as err:  # an unwritable directory, a file that is not a database
            if lock is not None:
                lock.close()
            raise ArchiveError(f"cannot open {path}: {err}") from None
        archive = cls(db, path, lock)
        try:  # a foreign archive is refused before anything is written
            info = archive.world_info
        except sqlite3.DatabaseError as err:
            archive.close()
            raise ArchiveError(f"{path} is not a NeuroGarden archive: {err}") from None
        if info is not None and info.archive_version != ARCHIVE_VERSION:
            archive.close()
            raise ArchiveError(
                f"{path} is an archive of version {info.archive_version}; "
                f"this neurogarden writes version {ARCHIVE_VERSION}"
            )
        return archive

    @classmethod
    def peek(cls, path: str) -> WorldInfo | None:
        """What world a file holds, without creating the file. None for nothing there."""
        if path == MEMORY or not os.path.exists(path):
            return None
        archive = cls.open(path, readonly=True)  # raises for a file that is not an archive
        try:
            return archive.world_info
        finally:
            archive.close()

    def close(self) -> None:
        self._db.close()
        if self._lock is not None:
            self._lock.close()  # closing the file releases the lock
            self._lock = None

    # --- the world ------------------------------------------------------------------------

    @property
    def world_info(self) -> WorldInfo | None:
        try:
            row = self._db.execute(
                "SELECT map_name, map_text, seed, config, rules_version, archive_version, "
                "created_at FROM world WHERE id = 1"
            ).fetchone()
        except sqlite3.OperationalError:  # a read-only open of a file without the tables
            return None
        if row is None:
            return None
        map_name, map_text, seed, config, rules_version, archive_version, created_at = row
        return WorldInfo(
            map_name=map_name,
            map_text=map_text,
            seed=seed,
            config=Config.from_dict(json.loads(config)),
            rules_version=rules_version,
            archive_version=archive_version,
            created_at=created_at,
        )

    def create_world(
        self, map_name: str, map_text: str, seed: int | None, config: Config, snapshot: dict
    ) -> None:
        """Record a new world; `snapshot` (the engine's, at its current tick) defines it."""
        if self.world_info is not None:
            raise ArchiveError(f"{self.path} already holds a world")
        with self.transaction():
            self._db.execute(
                "INSERT INTO world VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ARCHIVE_VERSION,
                    map_name,
                    map_text,
                    seed,
                    json.dumps(config.to_dict()),
                    RULES_VERSION,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            self.save_snapshot(snapshot["tick"], snapshot, {})

    # --- writing a tick -------------------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """One tick, one transaction: all of it lands or none of it does."""
        self._db.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def record_tick(
        self,
        tick: int,
        spawns: list[tuple[int, str, int, int]],
        despawns: list[int],
        actions: dict[int, int],
    ) -> None:
        inputs = TickInputs(tick, spawns, despawns, actions)
        self._db.execute("INSERT INTO ticks VALUES (?, ?)", (tick, inputs.to_json()))

    def born(
        self, agent_id: int, owner: str, lineage: int, name: str, body: str, born_tick: int
    ) -> None:
        self._db.execute(
            "INSERT INTO lives (agent_id, owner, lineage, name, body, born_tick) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (agent_id, owner, lineage, name, body, born_tick),
        )

    def died(self, agent_id: int, died_tick: int, causes: list[str], stats: EpisodeStats) -> None:
        self._db.execute(
            "UPDATE lives SET died_tick = ?, causes = ?, stats = ? WHERE agent_id = ?",
            (died_tick, json.dumps(list(causes)), json.dumps(stats.to_dict()), agent_id),
        )

    def note(self, tick: int, text: str) -> None:
        self._db.execute("INSERT INTO chronicle VALUES (?, ?)", (tick, text))

    def checkpoint(self, tick: int, state_hash: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?, ?)", (tick, state_hash))

    def save_snapshot(self, tick: int, state: dict, extras: dict) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?)",
            (tick, json.dumps(state, separators=(",", ":")), json.dumps(extras)),
        )

    # --- reading ----------------------------------------------------------------------------

    def next_tick(self) -> int:
        """The tick the world would run next: one past the last recorded, else where it began."""
        (last,) = self._db.execute("SELECT MAX(tick) FROM ticks").fetchone()
        if last is not None:
            return last + 1
        (first,) = self._db.execute("SELECT MIN(tick) FROM snapshots").fetchone()
        return 0 if first is None else first

    def lives(self, owner: str | None = None) -> list[Life]:
        """Every life, oldest first; or one owner's, by lineage."""
        if owner is None:
            rows = self._db.execute(f"SELECT {_LIFE_COLUMNS} FROM lives ORDER BY agent_id")
        else:
            rows = self._db.execute(
                f"SELECT {_LIFE_COLUMNS} FROM lives WHERE owner = ? ORDER BY lineage", (owner,)
            )
        return [_life(row) for row in rows.fetchall()]

    def life(self, owner: str, lineage: int) -> Life | None:
        row = self._db.execute(
            f"SELECT {_LIFE_COLUMNS} FROM lives WHERE owner = ? AND lineage = ?", (owner, lineage)
        ).fetchone()
        return None if row is None else _life(row)

    def life_end(self, life: Life) -> int:
        """One past the last recorded tick of a life: its last frame, or what is archived so far.

        The bound is fixed when asked, so replaying a life still going ends where the archive
        stood, instead of chasing a world that keeps writing."""
        return self.next_tick() if life.died_tick is None else life.died_tick + 1

    def recent_chronicle(self, lines: int) -> list[tuple[int, str]]:
        rows = self._db.execute(
            "SELECT tick, text FROM chronicle ORDER BY rowid DESC LIMIT ?", (lines,)
        ).fetchall()
        return [(tick, text) for tick, text in reversed(rows)]

    def chronicle_between(self, from_tick: int, to_tick: int) -> list[tuple[int, str]]:
        """Lines stamped in [from_tick, to_tick), in the order they were written."""
        rows = self._db.execute(
            "SELECT tick, text FROM chronicle WHERE tick >= ? AND tick < ? ORDER BY rowid",
            (from_tick, to_tick),
        ).fetchall()
        return [(tick, text) for tick, text in rows]

    def latest_snapshot(self, at_or_before: int) -> tuple[int, dict, dict] | None:
        row = self._db.execute(
            "SELECT tick, state, extras FROM snapshots WHERE tick <= ? ORDER BY tick DESC LIMIT 1",
            (at_or_before,),
        ).fetchone()
        if row is None:
            return None
        tick, state, extras = row
        return tick, json.loads(state), json.loads(extras)

    def snapshot_ticks(self) -> list[int]:
        return [tick for (tick,) in self._db.execute("SELECT tick FROM snapshots ORDER BY tick")]

    def inputs(self, from_tick: int, to_tick: int | None = None) -> Iterator[TickInputs]:
        """The recorded inputs of ticks in [from_tick, to_tick), fetched in batches."""
        tick = from_tick
        while to_tick is None or tick < to_tick:
            limit = _BATCH if to_tick is None else min(_BATCH, to_tick - tick)
            rows = self._db.execute(
                "SELECT tick, inputs FROM ticks WHERE tick >= ? ORDER BY tick LIMIT ?",
                (tick, limit),
            ).fetchall()
            if not rows:
                return
            for recorded, text in rows:
                if recorded != tick:
                    raise ArchiveError(f"tick {tick} is missing from the archive")
                yield TickInputs.from_json(recorded, text)
                tick += 1

    def checkpoints(self) -> dict[int, str]:
        return dict(self._db.execute("SELECT tick, state_hash FROM checkpoints ORDER BY tick"))

    def tick_count(self) -> int:
        (count,) = self._db.execute("SELECT COUNT(*) FROM ticks").fetchone()
        return count
