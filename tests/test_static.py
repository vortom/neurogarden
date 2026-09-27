"""The built browser client is served by the world server on plain HTTP GETs."""

import asyncio
import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from neurogarden.server import Server, ServerConfig
from neurogarden.server.app import allowed_origins
from neurogarden.server.static_files import (
    CSP,
    IMMUTABLE,
    STATIC_DIR,
    bundle_present,
    request_path,
    resolve,
)

FAST = dict(port=0, tps=50.0, npcs=[], hello_timeout=0.5)
WEB_DIR = Path(__file__).parent.parent / "web"


def assets_of(index: str) -> list[str]:
    return re.findall(r'(?:src|href)="\./(assets/[^"]+)"', index)


def test_the_committed_bundle_is_present_and_self_consistent():
    assert bundle_present()
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    referenced = assets_of(index)
    assert referenced, "index.html references no bundle at all"
    for asset in referenced:
        assert (STATIC_DIR / asset).is_file(), asset
        assert resolve("/" + asset) == STATIC_DIR / asset, asset
    assert "NEUROGARDEN" in index


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.mark.slow
def test_the_committed_bundle_is_what_the_sources_build_today(tmp_path: Path):
    """The same guard as `npm --prefix web run check`, for anyone who forgot to rebuild."""
    npm = shutil.which("npm")
    if npm is None or not (WEB_DIR / "node_modules").is_dir():
        pytest.skip("needs npm and `npm --prefix web install`")
    fresh = tmp_path / "static"
    # --outDir sends the build somewhere else, so emptyOutDir never touches the committed one.
    build = subprocess.run(
        [
            npm,
            "--prefix",
            str(WEB_DIR),
            "run",
            "build",
            "--",
            "--outDir",
            str(fresh),
            "--emptyOutDir",
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    assert sorted(tree(fresh)) == sorted(tree(STATIC_DIR))
    assert tree(fresh) == tree(STATIC_DIR), "run `npm --prefix web run build` and commit the bundle"


def test_the_page_stays_within_the_content_security_policy_we_send():
    """The CSP allows an inline stylesheet, a data: icon and same-origin scripts — no more."""
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "<style>" in index and "unsafe-inline" in CSP  # the stylesheet is inline
    assert 'href="data:image/svg+xml' in index and "img-src 'self' data:" in CSP
    assert not re.search(r"<script(?![^>]*\bsrc=\"\./)", index)  # no inline or foreign script
    assert not re.search(r'(?:src|href)="(?:https?:)?//', index)  # nothing off this origin


def test_request_path_drops_the_query_and_the_fragment():
    assert request_path("/index.html?x=1#top") == "/index.html"
    assert request_path("/healthz?probe=1") == "/healthz"
    assert request_path("/assets/app.js") == "/assets/app.js"


def test_resolve_maps_paths_inside_the_static_directory_only(tmp_path: Path):
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("// js")
    (tmp_path / "logo.PNG").write_bytes(b"\x89PNG")
    (tmp_path / "secret.bin").write_bytes(b"\x00")
    assert resolve("/", tmp_path) == tmp_path / "index.html"
    assert resolve(request_path("/index.html?x=1#top"), tmp_path) == tmp_path / "index.html"
    assert resolve("/assets/app.js", tmp_path) == tmp_path / "assets" / "app.js"
    assert resolve("/logo.PNG", tmp_path) == tmp_path / "logo.PNG"  # suffix, case-insensitively
    assert resolve("/missing.html", tmp_path) is None
    assert resolve("/../pyproject.toml", tmp_path) is None
    assert resolve("/assets/../../etc/passwd", tmp_path) is None
    assert resolve("assets/app.js", tmp_path) is None
    assert resolve("/assets", tmp_path) is None  # a directory, not a file
    assert resolve("/secret.bin", tmp_path) is None  # unknown content type


def fetch(url: str, method: str = "GET") -> tuple[int, bytes, dict]:
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read(), dict(err.headers)


def served(*calls):
    """Run a live server on an ephemeral port and make each (path, method) request."""

    async def scenario():
        async with Server(ServerConfig(**FAST)) as server:
            base = f"http://127.0.0.1:{server.port}"
            return await asyncio.to_thread(
                lambda: [fetch(f"{base}{path}", method) for path, method in calls]
            )

    return asyncio.run(scenario())


def test_the_server_serves_the_page_with_its_security_headers():
    ((status, body, headers),) = served(("/", "GET"))
    assert status == 200 and b"NEUROGARDEN" in body
    assert headers["Content-Type"].startswith("text/html")
    assert headers["Cache-Control"] == "no-cache"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Content-Security-Policy"] == CSP
    assert int(headers["Content-Length"]) == len(body)


def test_the_hashed_asset_is_javascript_the_browser_may_keep_forever():
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    asset = assets_of(index)[0]
    ((status, body, headers),) = served((f"/{asset}", "GET"))
    assert status == 200 and body
    assert headers["Content-Type"].startswith("text/javascript")
    assert headers["Cache-Control"] == IMMUTABLE


def test_head_answers_like_get_but_without_the_body():
    (get, head) = served(("/", "GET"), ("/", "HEAD"))
    assert (get[0], head[0]) == (200, 200)
    assert head[1] == b"" and get[1] != b""
    assert head[2]["Content-Length"] == get[2]["Content-Length"]
    assert head[2]["Content-Type"] == get[2]["Content-Type"]


def test_anything_but_get_or_head_is_refused_with_the_methods_we_do_know():
    ((status, _, headers),) = served(("/", "POST"))
    assert status == 405
    assert headers["Allow"] == "GET, HEAD"


def test_the_health_check_ignores_a_query_string():
    ((status, body, _),) = served(("/healthz?probe=1", "GET"))
    assert (status, body) == (200, b"OK\n")


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
    ((status, _, _),) = served((path, "GET"))
    assert status == 404


def free_port() -> int:
    """A port that was free a moment ago: `--port 0` cannot have an origin list."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_the_origin_list_covers_the_addresses_the_page_is_served_from():
    assert allowed_origins("127.0.0.1", 8765) == [
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://[::1]:8765",
        None,
    ]
    assert allowed_origins("garden.example", 8765) == [
        "http://garden.example:8765",
        "https://garden.example:8765",
        None,
    ]
    # A browser leaves out the port a scheme owns, so both spellings are allowed there
    assert allowed_origins("garden.example", 80) == [
        "http://garden.example:80",
        "http://garden.example",
        "https://garden.example:80",
        None,
    ]
    assert allowed_origins("127.0.0.1", 0) is None  # not knowable before the bind


def test_a_websocket_from_another_site_is_refused_but_a_brain_without_one_is_not():
    port = free_port()

    async def scenario():
        async with Server(ServerConfig(**{**FAST, "port": port})) as server:
            url = f"ws://127.0.0.1:{server.port}"
            with pytest.raises(InvalidStatus) as refused:
                async with connect(url, additional_headers={"Origin": "http://evil.example"}):
                    pass
            async with connect(url):  # the SDK sends no Origin at all
                pass
            async with connect(url, additional_headers={"Origin": f"http://localhost:{port}"}):
                pass
            return refused.value.response.status_code

    assert asyncio.run(scenario()) == 403
