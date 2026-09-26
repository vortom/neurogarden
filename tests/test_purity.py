"""Engine purity: no I/O, no external randomness, no dependency on dojo/brains."""

from __future__ import annotations

import ast
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1] / "src" / "neurogarden" / "engine"
ALLOWED_OPEN_FILE = ENGINE_ROOT / "maps" / "__init__.py"
FORBIDDEN_PREFIXES = (
    "random",
    "time",
    "os",
    "numpy.random",
    "neurogarden.dojo",
    "neurogarden.brains",
)


def _imported_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    return []


def test_engine_has_no_forbidden_imports_and_no_stray_file_io():
    for path in sorted(ENGINE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for name in _imported_names(node):
                assert not name.startswith(FORBIDDEN_PREFIXES), f"{path}: forbidden import {name}"
            if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == "open")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "open")
            ):
                assert path == ALLOWED_OPEN_FILE, f"{path}: open() outside maps/__init__.py"
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "random"
                and isinstance(node.value, ast.Name)
                and node.value.id in ("np", "numpy")
            ):
                raise AssertionError(f"{path}: np.random referenced")
