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
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
}
METHODS = ("GET", "HEAD")
NO_CACHE = "no-cache"
# Vite names everything under /assets/ after a hash of its contents, so a name never
# means two different files and the browser may keep it forever.
IMMUTABLE = "public, max-age=31536000, immutable"
# What the page actually needs: its own bundle, the inline stylesheet in index.html, a
# data: URI favicon, and a WebSocket back to the world it was served from.
CSP = (
    "default-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self' ws: wss:"
)


def request_path(target: str) -> str:
    """The path of a request target, without its query or fragment."""
    return target.split("?", 1)[0].split("#", 1)[0]


def resolve(path: str, root: Path = STATIC_DIR) -> Path | None:
    """The file a path maps to inside `root`, or None (unknown, traversal, binary).

    `path` is a request path with the query and fragment already stripped; see
    `request_path`.
    """
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
    if candidate.suffix.lower() not in CONTENT_TYPES:
        return None
    return candidate


def bundle_present(root: Path = STATIC_DIR) -> bool:
    return (root / "index.html").is_file()


def _reply(connection, status: HTTPStatus, body: bytes, content_type: str, cache: str, head: bool):
    """A response with a bytes body: `connection.respond` only knows how to send text."""
    response = connection.respond(status, "")
    headers = response.headers
    for name, value in (
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),  # a HEAD still says how long the body would be
        ("Cache-Control", cache),
        ("X-Content-Type-Options", "nosniff"),
        ("Content-Security-Policy", CSP),
    ):
        if name in headers:
            del headers[name]  # websockets' Headers appends; replace instead
        headers[name] = value
    response.body = b"" if head else body
    return response


def make_process_request(web: bool = True, root: Path = STATIC_DIR):
    """websockets' process_request hook: health check, then the page, else 404."""
    bytes_of: dict[Path, bytes] = {}

    def read(target: Path) -> bytes:
        """Read a file once and keep it: the tick loop shares this event loop."""
        if target not in bytes_of:
            bytes_of[target] = target.read_bytes()
        return bytes_of[target]

    def process_request(connection, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None  # a real WebSocket handshake: let it through
        head = request.method == "HEAD"
        if request.method not in METHODS:
            response = _reply(
                connection,
                HTTPStatus.METHOD_NOT_ALLOWED,
                b"method not allowed\n",
                CONTENT_TYPES[".txt"],
                NO_CACHE,
                head=False,
            )
            response.headers["Allow"] = ", ".join(METHODS)
            return response
        path = request_path(request.path)
        if path == "/healthz":
            return _reply(connection, HTTPStatus.OK, b"OK\n", CONTENT_TYPES[".txt"], NO_CACHE, head)
        target = resolve(path, root) if web else None
        if target is None:
            return _reply(
                connection,
                HTTPStatus.NOT_FOUND,
                b"not found\n",
                CONTENT_TYPES[".txt"],
                NO_CACHE,
                head,
            )
        cache = IMMUTABLE if path.startswith("/assets/") else NO_CACHE
        content_type = CONTENT_TYPES[target.suffix.lower()]
        return _reply(connection, HTTPStatus.OK, read(target), content_type, cache, head)

    return process_request
