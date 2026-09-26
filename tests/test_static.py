"""The built browser client is served by the world server on plain HTTP GETs."""

import asyncio
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from neurogarden.server import Server, ServerConfig
from neurogarden.server.static_files import STATIC_DIR, bundle_present, resolve

FAST = dict(port=0, tps=50.0, npcs=[], hello_timeout=0.5)


def test_the_committed_bundle_is_present_and_self_consistent():
    assert bundle_present()
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for asset in re.findall(r'(?:src|href)="\./(assets/[^"]+)"', index):
        assert (STATIC_DIR / asset).is_file(), asset
    assert "NEUROGARDEN" in index


def test_resolve_maps_paths_inside_the_static_directory_only(tmp_path: Path):
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("// js")
    (tmp_path / "secret.bin").write_bytes(b"\x00")
    assert resolve("/", tmp_path) == tmp_path / "index.html"
    assert resolve("/index.html?x=1#top", tmp_path) == tmp_path / "index.html"
    assert resolve("/assets/app.js", tmp_path) == tmp_path / "assets" / "app.js"
    assert resolve("/missing.html", tmp_path) is None
    assert resolve("/../pyproject.toml", tmp_path) is None
    assert resolve("/assets/../../etc/passwd", tmp_path) is None
    assert resolve("assets/app.js", tmp_path) is None
    assert resolve("/assets", tmp_path) is None  # a directory, not a file
    assert resolve("/secret.bin", tmp_path) is None  # unknown content type


def fetch(url: str) -> tuple[int, str, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read().decode("utf-8"), dict(response.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8"), dict(err.headers)


def test_the_server_serves_the_page_and_keeps_the_health_check():
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            base = f"http://127.0.0.1:{server.port}"
            return await asyncio.to_thread(
                lambda: (fetch(f"{base}/"), fetch(f"{base}/healthz"), fetch(f"{base}/nope"))
            )

    (status, body, headers), (health, ok, _), (missing, _, _) = asyncio.run(scenario())
    assert status == 200 and "NEUROGARDEN" in body
    assert headers["Content-Type"].startswith("text/html")
    assert headers["Cache-Control"] == "no-cache"
    assert (health, ok) == (200, "OK\n")
    assert missing == 404


def test_no_web_turns_the_page_off_but_not_the_health_check():
    async def scenario():
        async with Server(ServerConfig(**FAST, web=False)) as server:
            base = f"http://127.0.0.1:{server.port}"
            return await asyncio.to_thread(
                lambda: (fetch(f"{base}/")[0], fetch(f"{base}/healthz")[0])
            )

    assert asyncio.run(scenario()) == (404, 200)


@pytest.mark.parametrize("path", ["/../pyproject.toml", "/assets/../../uv.lock"])
def test_traversal_is_a_404(path):
    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            return await asyncio.to_thread(
                lambda: fetch(f"http://127.0.0.1:{server.port}{path}")[0]
            )

    assert asyncio.run(scenario()) == 404
