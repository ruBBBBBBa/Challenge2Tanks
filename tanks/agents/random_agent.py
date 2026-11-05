# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
from __future__ import annotations

from typing import Optional

import numpy as np

from tanks.agents.base import Agent
from tanks.game import N_ACTIONS


class RandomAgent(Agent):
    """Uniform random actions. Level 0 / sanity check."""

    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, state: dict) -> int:
        return int(self.rng.integers(N_ACTIONS))
