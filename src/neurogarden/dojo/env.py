"""Gymnasium environment: the engine in lockstep, one fly, learner-side reward."""

from __future__ import annotations

import dataclasses

import gymnasium
import numpy as np
from gymnasium import spaces

from neurogarden.engine import maps
from neurogarden.engine.body import action_names, observation_spec
from neurogarden.engine.clock import day_number
from neurogarden.engine.config import Config
from neurogarden.engine.rng import SplitMix64
from neurogarden.engine.tiles import parse_map
from neurogarden.engine.world import World

from .render_ansi import render
from .rewards import BodyState, RewardFn, resolve
from .stats import StatsTracker

BODY = "fly"
_BIRTH_XOR = 0xB1F7_0DA7_5EED_C10C  # decorrelates the hour of birth from the world's own draws


def birth_tick(seed: int, day_length: int) -> int:
    """The tick a fly hatches at when a life may begin at any hour: a fixed function of the
    world seed, so a life is still reproducible from its seed alone."""
    return SplitMix64(seed ^ _BIRTH_XOR).randbelow(day_length)


class NeuroGardenEnv(gymnasium.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(
        self,
        map: str = "drosoville",
        config: Config | None = None,
        reward: str | RewardFn = "wellbeing",
        max_steps: int = 6000,
        render_mode: str | None = None,
        any_hour: bool = False,
    ) -> None:
        """`any_hour=False`: every life begins at world tick 0, at dawn. `any_hour=True`: the
        world first runs empty until `birth_tick(seed)`, so the fly is born at some hour of
        the first day — as in a live garden, where a fly hatches whenever its owner joins. A
        brain that only ever met dawn births has age and daylight locked together in
        everything it learned, and may be lost when they come apart."""
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._map_text = maps.load(map) if map in maps.available() else parse_map(map).text
        self.config = config or Config()
        self._reward = resolve(reward)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.any_hour = any_hour
        self.observation_space = spaces.Dict(
            {
                name: spaces.Box(
                    low=channel.low,
                    high=channel.high,
                    shape=channel.shape,
                    dtype=np.dtype(channel.dtype).type,
                )
                for name, channel in observation_spec(BODY).items()
            }
        )
        self.action_space = spaces.Discrete(len(action_names(BODY)))
        self.world: World | None = None
        self._fly = 0
        self._steps = 0
        self._done = True
        self._terminated = False
        self._truncated = False
        self._observation: dict[str, np.ndarray] = {}
        self._prev_body: BodyState | None = None
        self._tracker: StatsTracker | None = None

    @property
    def agent_id(self) -> int:
        """Engine id of the fly this environment controls."""
        return self._fly

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        world_seed = seed if seed is not None else int(self.np_random.integers(0, 2**63 - 1))
        self.world = World.from_map(self._map_text, self.config, world_seed)
        if self.any_hour:
            for _ in range(birth_tick(world_seed, self.config.day_length)):
                self.world.step({})
        self._fly = self.world.spawn(body=BODY)
        agent = self.world.state.agents[self._fly]
        self._tracker = StatsTracker(self._fly, (agent.x, agent.y), self.config.day_length)
        self._steps = 0
        self._done = False
        self._terminated = False
        self._truncated = False
        self._observation = self.world.observe(self._fly)
        self._prev_body = BodyState.from_observation(self._observation)
        return self._observation, self._info([])

    def step(self, action):
        if self.world is None:
            raise RuntimeError("call reset() before step()")
        if self._done:  # episode already ended: frozen no-op, the world does not advance
            return self._observation, 0.0, self._terminated, self._truncated, self._info([])
        result = self.world.step({self._fly: action})
        self._observation = result.observations[self._fly]
        body = BodyState.from_observation(self._observation)
        died = not self.world.state.agents[self._fly].alive
        events = result.agent_events[self._fly]
        reward = float(self._reward(self._prev_body, body, events, died))
        self._tracker.update(result.events, body)
        self._prev_body = body
        self._steps += 1
        truncated = not died and self._steps >= self.max_steps
        self._done = died or truncated
        self._terminated = died
        self._truncated = truncated
        return self._observation, reward, died, truncated, self._info(events)

    def render(self):
        if self.render_mode == "ansi" and self.world is not None:
            return render(self.world, self._fly)
        return None

    def _info(self, events: list) -> dict:
        return {
            "events": events,
            "stats": dataclasses.replace(self._tracker.stats),  # snapshot: never the live tracker
            "tick": self.world.tick,
            "day": day_number(self.world.tick, self.config),
        }
