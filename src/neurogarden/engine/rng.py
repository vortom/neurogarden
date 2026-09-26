"""SplitMix64: the only source of randomness inside the engine.

Fully specified so that replays survive dependency upgrades and a future port
to another language can reproduce every draw bit for bit.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1


class SplitMix64:
    __slots__ = ("state",)

    def __init__(self, seed: int = 0) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def randbelow(self, n: int) -> int:
        """Uniform integer in [0, n) by rejection sampling (no modulo bias)."""
        if n <= 0:
            raise ValueError("n must be positive")
        limit = ((1 << 64) // n) * n
        while True:
            r = self.next_u64()
            if r < limit:
                return r % n

    def shuffle(self, items: list) -> None:
        """Fisher-Yates, in place. A list of fewer than two items draws nothing."""
        for i in range(len(items) - 1, 0, -1):
            j = self.randbelow(i + 1)
            items[i], items[j] = items[j], items[i]
