"""Compose a live world: engine + runner + gateway + hosted brains."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import signal
from dataclasses import dataclass, field

from websockets.asyncio.server import serve as websockets_serve

from neurogarden.brains import BRAINS
from neurogarden.engine import RULES_VERSION, Config, World, maps

from .archive import MEMORY, Archive, ArchiveError, WorldInfo
from .gateway import Gateway
from .ports import LocalPort
from .runner import WorldRunner
from .static_files import STATIC_DIR, bundle_present, make_process_request

log = logging.getLogger("neurogarden.server")
DEFAULT_TOKEN = "dev"
DEFAULT_MOTD = "Small worlds. Strange minds. Be kind to the flies."
DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass
class ServerConfig:
    map: str = "drosoville"
    seed: int = 0
    tps: float = 5.0
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = DEFAULT_TOKEN
    npcs: list[tuple[str, int]] = field(default_factory=lambda: [("scripted", 1)])
    hello_timeout: float = 5.0
    web: bool = True  # serve the built browser client on plain HTTP GETs
    motd: str = DEFAULT_MOTD
    config: Config | None = None
    archive: str = MEMORY  # the world's file; `:memory:` for a world that ends with its process


def check_identity(info: WorldInfo, config: ServerConfig, path: str) -> None:
    """An archive holds one world; serving it under another name or seed is refused."""
    if info.rules_version != RULES_VERSION:
        raise ArchiveError(
            f"{path} was written under rules_version {info.rules_version}; this engine is "
            f"{RULES_VERSION} — resume it with the older neurogarden, or start a new archive"
        )
    if (info.map_name, info.seed) != (config.map, config.seed):
        raise ArchiveError(
            f"{path} holds {info.map_name} (seed {info.seed}), not {config.map} "
            f"(seed {config.seed}); pass --archive another.db for a new world"
        )
    if config.config is not None and config.config != info.config:
        raise ArchiveError(f"{path} was created with other engine settings than these")


def _is_loopback(host: str) -> bool:
    """'' means every interface for websockets, so it is not loopback — nor is 0.0.0.0 or ::."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def allowed_origins(host: str, port: int) -> list[str | None] | None:
    """The `Origin` values a browser may connect from, or None to check nothing.

    Only the page this server itself serves has any business opening a socket to it: a
    page on another site must not be able to fly someone's fly (cross-site WebSocket
    hijacking). Non-browsers — the SDK, `neurogarden watch` — send no `Origin` at all,
    which is what the `None` in the list allows. websockets wants the list before the
    bind, so an ephemeral port (`--port 0`, which only the tests use) cannot have one:
    nothing is known yet about the address the page will be served from.
    """
    if port == 0:
        return None
    loopback = _is_loopback(host)
    hosts = ("127.0.0.1", "localhost", "[::1]") if loopback else (host,)
    schemes = ("http",) if loopback else ("http", "https")
    origins: list[str | None] = []
    for scheme in schemes:
        for name in hosts:
            origins.append(f"{scheme}://{name}:{port}")
            if port == DEFAULT_PORTS[scheme]:
                origins.append(f"{scheme}://{name}")  # a browser omits the scheme's own port
    return [*origins, None]


def parse_npc(spec: str) -> tuple[str, int]:
    """'scripted:2' -> ('scripted', 2); 'random' -> ('random', 1); 'none' -> no NPCs."""
    kind, _, count = spec.partition(":")
    if kind == "none":
        return ("none", 0)
    if kind not in BRAINS:
        raise ValueError(f"unknown brain {kind!r}; choose from {sorted(BRAINS)}")
    number = int(count) if count else 1
    if number < 0:
        raise ValueError("NPC count must not be negative")
    return (kind, number)


class Server:
    """`async with Server(config) as server:` runs a world until the block ends."""

    def __init__(self, config: ServerConfig) -> None:
        if not _is_loopback(config.host) and config.token == DEFAULT_TOKEN:
            raise ValueError(f"refusing to bind {config.host} with the default token; pass --token")
        self.config = config
        self.archive = Archive.open(config.archive)
        try:
            info = self.archive.world_info
            self.resumed = info is not None
            if info is None:
                map_text = maps.load(config.map) if config.map in maps.available() else config.map
                world = World.from_map(map_text, config.config or Config(), config.seed)
                self.archive.create_world(
                    config.map, world.map_text, config.seed, world.config, world.snapshot()
                )
                self.runner = WorldRunner(
                    world,
                    archive=self.archive,
                    tps=config.tps,
                    map_name=config.map,
                    motd=config.motd,
                )
            else:
                check_identity(info, config, config.archive)
                self.runner = WorldRunner.from_archive(
                    self.archive, tps=config.tps, motd=config.motd
                )
        except BaseException:
            self.archive.close()
            raise
        self.gateway = Gateway(self.runner, config.token, config.hello_timeout)
        self.stop = asyncio.Event()
        self.port: int = config.port
        self._ws = None
        self._ticker: asyncio.Task | None = None

    def _hatch_npcs(self) -> None:
        for kind, count in self.config.npcs:
            for index in range(1, count + 1):
                brain = BRAINS[kind](seed=index)
                owner = f"npc-{kind}-{index}"
                port = LocalPort(brain, owner, self.runner, self.runner.catalog.bodies["fly"])
                self.runner.attach(port)
                self.runner.request_join(port)

    async def __aenter__(self) -> Server:
        self._hatch_npcs()
        if self.config.web and not bundle_present():
            log.warning(
                "no browser client in %s: run `npm --prefix web run build` to serve the page",
                STATIC_DIR,
            )
        self._ws = await websockets_serve(
            self.gateway.handle,
            self.config.host,
            self.config.port,
            origins=allowed_origins(self.config.host, self.config.port),
            process_request=make_process_request(self.config.web),
        )
        self.port = self._ws.sockets[0].getsockname()[1]
        self._ticker = asyncio.get_running_loop().create_task(self.runner.run(self.stop))
        log.info(
            "world %s (seed %d) %s on ws://%s:%d at %.1f tps, tick %d",
            self.config.map,
            self.config.seed,
            "resumed" if self.resumed else "live",
            self.config.host,
            self.port,
            self.config.tps,
            self.runner.world.tick,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        self.stop.set()
        try:
            if self._ticker is not None:
                (stopped,) = await asyncio.gather(self._ticker, return_exceptions=True)
                if isinstance(stopped, BaseException):  # retrieved, so never "never retrieved"
                    log.error("the world stopped ticking", exc_info=stopped)
        finally:
            try:
                if self._ws is not None:  # the listener closes even if the ticker died
                    self._ws.close(close_connections=True, code=1001, reason="server shutting down")
                    await self._ws.wait_closed()
            finally:
                self.archive.close()

    @property
    def url(self) -> str:
        return f"ws://{self.config.host}:{self.port}"

    async def wait_stopped(self) -> None:
        await self.stop.wait()


async def serve(config: ServerConfig, on_ready=None) -> None:
    """Run until SIGINT/SIGTERM; `on_ready(server)` sees the port the world actually bound."""
    async with Server(config) as server:
        if on_ready is not None:
            on_ready(server)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, server.stop.set)
            except NotImplementedError:  # pragma: no cover - Windows
                pass
        await server.wait_stopped()
