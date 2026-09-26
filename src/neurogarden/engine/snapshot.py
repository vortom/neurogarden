"""Snapshot, restore and the canonical state hash (the determinism contract)."""

from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import asdict, fields

import numpy as np

from .body import get_body
from .config import RULES_VERSION, Config
from .rng import SplitMix64
from .state import Agent, WorldState

# layer name -> little-endian dtype, in canonical hash order
LAYERS = (
    ("terrain", "<u1"),
    ("resource_kind", "<u1"),
    ("resource_amount", "<i2"),
    ("resource_age", "<i4"),
    ("occupant", "<i4"),
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "rules_version",
        "config",
        "tick",
        "rng_state",
        "width",
        "height",
        "layers",
        "agents",
        "next_agent_id",
    }
)
_AGENT_KEYS = frozenset(f.name for f in fields(Agent))


def _layer_bytes(state: WorldState, name: str, dtype: str) -> bytes:
    return np.ascontiguousarray(getattr(state, name)).astype(dtype).tobytes(order="C")


def snapshot(state: WorldState) -> dict:
    """JSON-serialisable copy of the complete dynamic state."""
    return {
        "rules_version": RULES_VERSION,
        "config": state.config.to_dict(),
        "tick": state.tick,
        "rng_state": str(state.rng.state),  # a u64 exceeds JavaScript number precision
        "width": state.width,
        "height": state.height,
        "layers": {
            name: base64.b64encode(_layer_bytes(state, name, dtype)).decode("ascii")
            for name, dtype in LAYERS
        },
        "agents": [asdict(state.agents[i]) for i in sorted(state.agents)],
        "next_agent_id": state.next_agent_id,
    }


def _validate(data: dict) -> None:
    missing = _TOP_LEVEL_KEYS - set(data)
    if missing:
        raise ValueError(f"snapshot missing key(s): {sorted(missing)}")
    missing_layers = {name for name, _ in LAYERS} - set(data["layers"])
    if missing_layers:
        raise ValueError(f"snapshot missing layer(s): {sorted(missing_layers)}")
    for entry in data["agents"]:
        keys = set(entry)
        if keys != _AGENT_KEYS:
            offending = sorted(keys ^ _AGENT_KEYS)
            raise ValueError(f"agent snapshot has unexpected or missing key(s): {offending}")


def restore(data: dict) -> WorldState:
    _validate(data)
    if data["rules_version"] != RULES_VERSION:
        raise ValueError(
            f"snapshot has rules_version {data['rules_version']}, engine is {RULES_VERSION}"
        )
    shape = (data["height"], data["width"])
    layers = {
        name: np.frombuffer(base64.b64decode(data["layers"][name]), dtype=dtype)
        .reshape(shape)
        .astype(np.dtype(dtype).newbyteorder("="))
        for name, dtype in LAYERS
    }
    rng = SplitMix64(0)
    rng.state = int(data["rng_state"])
    agents = {entry["id"]: Agent(**entry) for entry in data["agents"]}
    return WorldState(
        config=Config.from_dict(data["config"]),
        rng=rng,
        agents=agents,
        next_agent_id=data["next_agent_id"],
        tick=data["tick"],
        **layers,
    )


def state_hash(state: WorldState) -> str:
    digest = hashlib.sha256()
    digest.update(
        struct.pack("<IQQHH", RULES_VERSION, state.tick, state.rng.state, state.width, state.height)
    )
    for name, dtype in LAYERS:
        digest.update(_layer_bytes(state, name, dtype))
    digest.update(struct.pack("<II", state.next_agent_id, len(state.agents)))
    for agent_id in sorted(state.agents):
        a = state.agents[agent_id]
        digest.update(
            struct.pack(
                "<12q",
                a.id,
                get_body(a.body).index,
                a.x,
                a.y,
                a.facing,
                a.satiety,
                a.hydration,
                a.energy,
                a.health,
                a.age,
                int(a.alive),
                int(a.bumped),
            )
        )
    return digest.hexdigest()
