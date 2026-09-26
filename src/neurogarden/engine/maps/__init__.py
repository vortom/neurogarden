"""Bundled text maps."""

from __future__ import annotations

from importlib import resources


def available() -> tuple[str, ...]:
    files = resources.files(__name__).iterdir()
    return tuple(sorted(f.name[:-4] for f in files if f.name.endswith(".txt")))


def load(name: str) -> str:
    """Text of a bundled map, e.g. load("drosoville")."""
    if name not in available():
        raise ValueError(f"unknown map {name!r}; available: {', '.join(available())}")
    return resources.files(__name__).joinpath(f"{name}.txt").read_text(encoding="utf-8")
