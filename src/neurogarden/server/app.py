"""Compose a live world: engine + runner + gateway + hosted brains."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import signal
from dataclasses import dataclass, field

from websockets.asyncio.server import serve as websockets_serve

from neurogarden.brains import BRAINS
from neurogarden.engine import Config, World, maps

from .gateway import Gateway, health_check
from .ports import LocalPort
from .runner import WorldRunner

log = logging.getLogger("neurogarden.server")
DEFAULT_TOKEN = "dev"
DEFAULT_MOTD = "Small worlds. Strange minds. Be kind to the flies."


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
    motd: str = DEFAULT_MOTD
    config: Config | None = None


def _is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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
        map_text = maps.load(config.map) if config.map in maps.available() else config.map
        world = World.from_map(map_text, config.config or Config(), config.seed)
        self.runner = WorldRunner(world, tps=config.tps, map_name=config.map, motd=config.motd)
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
        self._ws = await websockets_serve(
            self.gateway.handle,
            self.config.host,
            self.config.port,
            process_request=health_check,
        )
        self.port = self._ws.sockets[0].getsockname()[1]
        self._ticker = asyncio.get_running_loop().create_task(self.runner.run(self.stop))
        log.info(
            "world %s (seed %d) live on ws://%s:%d at %.1f tps",
            self.config.map,
            self.config.seed,
            self.config.host,
            self.port,
            self.config.tps,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        self.stop.set()
        if self._ticker is not None:
            await self._ticker
        if self._ws is not None:
            self._ws.close(close_connections=True, code=1001, reason="server shutting down")
            await self._ws.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://{self.config.host}:{self.port}"

    async def wait_stopped(self) -> None:
        await self.stop.wait()


async def serve(config: ServerConfig) -> None:
    """Run until SIGINT/SIGTERM."""
    async with Server(config) as server:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, server.stop.set)
            except NotImplementedError:  # pragma: no cover - Windows
                pass
        await server.wait_stopped()
