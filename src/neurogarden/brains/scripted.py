"""A hand-written survivor: the bar every learned brain has to clear."""

from __future__ import annotations

import numpy as np

from neurogarden.engine.body import DIRECTIONS, VISION_SIZE, Action
from neurogarden.engine.config import NIGHT_LIGHT_THRESHOLD
from neurogarden.engine.rng import SplitMix64
from neurogarden.engine.tiles import WALKABLE, Resource

_FRUIT, _HUMIDITY, _NEST = 0, 1, 2  # rows of the smell channel
_MOVES = (Action.MOVE_N, Action.MOVE_E, Action.MOVE_S, Action.MOVE_W)
_CENTRE = VISION_SIZE // 2


class ScriptedBrain:
    """First matching rule wins:

    1. consume when on fruit and hungry, or next to water and thirsty
    2. energy critical -> rest in place
    3. a need is urgent -> forage for it, even at night
    4. night or tired -> follow the nest scent; rest on the nest
    5. a need is below its threshold -> forage for the lower of satiety / hydration
    6. otherwise explore
    """

    def __init__(
        self,
        hungry_below: int = 600,
        thirsty_below: int = 600,
        urgent_below: int = 250,
        tired_below: int = 250,
        rested_above: int = 700,
        critical_energy: int = 120,
        keep_heading_permille: int = 800,
        explore_commit: int = 25,
        seed: int = 0,
    ) -> None:
        self.hungry_below = hungry_below
        self.thirsty_below = thirsty_below
        self.urgent_below = urgent_below
        self.tired_below = tired_below
        self.rested_above = rested_above
        self.critical_energy = critical_energy
        self.keep_heading_permille = keep_heading_permille
        self.explore_commit = explore_commit
        self._seed = seed
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self._rng = SplitMix64(self._seed if seed is None else seed)
        self._heading = self._rng.randbelow(4)
        self._resting = False
        self._explore_left = 0

    def act(self, observation: dict[str, np.ndarray]) -> int:
        satiety, hydration, energy = (int(v) for v in observation["body"][:3])
        bumped, on_resource, water_adjacent, on_nest = (int(v) for v in observation["touch"])
        night = int(observation["env"][0]) < NIGHT_LIGHT_THRESHOLD
        hungry, thirsty = satiety < self.hungry_below, hydration < self.thirsty_below

        if (on_resource == Resource.FRUIT and hungry) or (water_adjacent and thirsty):
            return Action.CONSUME
        if energy < self.critical_energy:
            self._resting = True
            return Action.REST
        if min(satiety, hydration) < self.urgent_below:
            return self._forage(observation, _FRUIT if satiety <= hydration else _HUMIDITY, bumped)

        if energy < self.tired_below:
            self._resting = True
        elif energy >= self.rested_above:
            self._resting = False
        if night or self._resting:
            if on_nest:
                return Action.REST
            step = self._follow(observation, _NEST)
            return step if step is not None else Action.REST

        if hungry or thirsty:
            return self._forage(observation, _FRUIT if satiety <= hydration else _HUMIDITY, bumped)
        return self._explore(observation, bumped)

    def _walkable(self, observation: dict[str, np.ndarray], direction: int) -> bool:
        dx, dy = DIRECTIONS[direction]
        terrain, _, occupant = observation["vision"][_CENTRE + dy, _CENTRE + dx]
        return terrain in WALKABLE and occupant == 0

    def _follow(self, observation: dict[str, np.ndarray], scent: int) -> Action | None:
        """Step to the walkable neighbour whose scent beats the own tile, if any."""
        smell = observation["smell"][scent]
        best, best_value = None, int(smell[0])
        for direction in range(4):
            value = int(smell[1 + direction])
            if value > best_value and self._walkable(observation, direction):
                best, best_value = direction, value
        if best is None:
            return None
        self._heading = best
        return _MOVES[best]

    def _forage(self, observation: dict[str, np.ndarray], scent: int, bumped: int) -> Action:
        if self._explore_left == 0:
            step = self._follow(observation, scent)
            if step is not None:
                return step
            if int(observation["smell"][scent][0]) > 0:
                # The scent leads through water or a tree: walk it off before trying again.
                self._explore_left = self.explore_commit
        return self._explore(observation, bumped)

    def _explore(self, observation: dict[str, np.ndarray], bumped: int) -> Action:
        """Persistent random walk that avoids turning straight back."""
        if self._explore_left > 0:
            self._explore_left -= 1
        options = [d for d in range(4) if self._walkable(observation, d)]
        if not options:
            return Action.IDLE
        keep = (
            self._heading in options
            and not bumped
            and self._rng.randbelow(1000) < self.keep_heading_permille
        )
        if not keep:
            onward = [d for d in options if d != (self._heading + 2) % 4] or options
            self._heading = onward[self._rng.randbelow(len(onward))]
        return _MOVES[self._heading]
