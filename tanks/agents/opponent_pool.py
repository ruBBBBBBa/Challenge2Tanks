# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
OpponentPool: a different opponent every episode, sampled with weights.

    pool = OpponentPool({"basic": 0.4, "aggressive": 0.3, "model:models/v1.zip": 0.3})
    env = TanksEnv(opponent=pool)          # or directly TanksEnv(opponent={...})

Weights can be changed during training (see tanks/curriculum.py):
    env.set_opponent({"aggressive": 0.5, "baseline:medium": 0.5})
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from tanks.agents.base import Agent


class OpponentPool(Agent):
    def __init__(self, weights: Dict[str, float], seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[str, Agent] = {}
        self.current: Optional[Agent] = None
        self.current_spec: Optional[str] = None
        self.set_weights(weights)

    def set_weights(self, weights: Dict[str, float]) -> None:
        specs = [s for s, w in weights.items() if w > 0]
        if not specs:
            raise ValueError("OpponentPool needs at least one opponent with weight > 0")
        p = np.array([weights[s] for s in specs], dtype=np.float64)
        self.specs, self.probs = specs, p / p.sum()

    @property
    def name(self) -> str:
        return f"pool[{self.current_spec}]" if self.current_spec else "pool"

    def _get(self, spec: str) -> Agent:
        from tanks.agents.base import load_agent
        if spec not in self._cache:
            self._cache[spec] = load_agent(spec)
        return self._cache[spec]

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_spec = str(self.rng.choice(self.specs, p=self.probs))
        self.current = self._get(self.current_spec)
        self.current.reset(int(self.rng.integers(2**31 - 1)))

    def act(self, state: dict) -> int:
        if self.current is None:
            self.reset()
        return self.current.act(state)
