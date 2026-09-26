"""Serve the built web client on plain HTTP GETs of the WebSocket port."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
}


def resolve(path: str, root: Path = STATIC_DIR) -> Path | None:
    """The file a request path maps to inside `root`, or None (unknown, traversal, binary)."""
    path = path.split("?", 1)[0].split("#", 1)[0]
    if path in ("", "/"):
        path = "/index.html"
    if not path.startswith("/") or "\\" in path or "\x00" in path:
        return None
    relative = path.lstrip("/")
    if any(part in ("", ".", "..") for part in relative.split("/")):
        return None
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        return None
    if candidate.suffix not in CONTENT_TYPES:
        return None
    return candidate


def bundle_present(root: Path = STATIC_DIR) -> bool:
    return (root / "index.html").is_file()


def make_process_request(web: bool = True, root: Path = STATIC_DIR):
    """websockets' process_request hook: health check, then the page, else 404."""

    def process_request(connection, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None  # a real WebSocket handshake: let it through
        if request.path == "/healthz":
            return connection.respond(HTTPStatus.OK, "OK\n")
        target = resolve(request.path, root) if web else None
        if target is None:
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        response = connection.respond(HTTPStatus.OK, target.read_text(encoding="utf-8"))
        del response.headers["Content-Type"]  # websockets' Headers appends; replace instead
        response.headers["Content-Type"] = CONTENT_TYPES[target.suffix]
        response.headers["Cache-Control"] = "no-cache"
        return response

    return process_request
