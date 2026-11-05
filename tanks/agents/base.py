# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Common agent interface + `load_agent(spec)`.

Every player (random bot, heuristics, SB3 models, opponent pools) is an Agent:

    class Agent:
        def reset(self, seed=None): ...      # called at the start of every game
        def act(self, state) -> int: ...     # egocentric state -> action (0..5)

Agent specs (strings accepted everywhere: TanksEnv(opponent=...), play_match.py, evaluate.py):

    "random"                         uniform random actions
    "basic" | "aggressive" | "defensive"
    "param:aggression=0.8,caution=0.2,trigger=1.0,noise=0.05"
    "baseline:easy" | "baseline:medium" | "baseline:hard"   -> tanks/agents/baselines/<level>.zip
    "model:models/my_v1.zip"  or just "models/my_v1.zip"     -> any SB3 model trained on the official obs
    "submission:<name>"              -> submissions/<name>/model.zip
    "my:<name>"                      -> your own bot, MY_BOTS[name] in student/my_opponents.py
    {"basic": 0.5, "baseline:easy": 0.5}                      -> OpponentPool (one opponent per episode)
"""
from __future__ import annotations

import os
from typing import Optional, Union

CHALLENGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASELINES_DIR = os.path.join(CHALLENGE_ROOT, "tanks", "agents", "baselines")


class Agent:
    name = "agent"

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, state: dict) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


def _resolve_path(path: str) -> str:
    """Accept paths relative to the current directory or to the challenge root."""
    for candidate in (path, os.path.join(CHALLENGE_ROOT, path)):
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, "model.zip")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Model not found: {path}")


def _parse_params(text: str) -> dict:
    params = {}
    for part in filter(None, text.split(",")):
        key, value = part.split("=")
        params[key.strip()] = float(value)
    return params


def load_agent(spec: Union[str, dict, Agent], **kwargs) -> Agent:
    """Build an Agent from a spec (see module docstring)."""
    from tanks.agents.heuristics import PRESETS, ParamHeuristic
    from tanks.agents.model_agent import ModelAgent
    from tanks.agents.opponent_pool import OpponentPool
    from tanks.agents.random_agent import RandomAgent

    if isinstance(spec, Agent):
        return spec
    if isinstance(spec, dict):
        return OpponentPool(spec, **kwargs)
    if not isinstance(spec, str):
        raise TypeError(f"Invalid agent spec: {spec!r}")

    s = spec.strip()
    if s == "random":
        return RandomAgent()
    if s in PRESETS:
        return ParamHeuristic(name=s, **PRESETS[s])
    if s.startswith("param:"):
        params = _parse_params(s[len("param:"):])
        return ParamHeuristic(name=s, **params)
    if s.startswith("baseline:"):
        level = s.split(":", 1)[1]
        path = os.path.join(BASELINES_DIR, level + ".zip")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Baseline '{level}' not found at {path}. Ask your teacher for the baseline models.")
        return ModelAgent(path, name=s, **kwargs)
    if s.startswith("submission:"):
        name = s.split(":", 1)[1]
        return ModelAgent(_resolve_path(os.path.join("submissions", name)), name=s, **kwargs)
    if s.startswith("my:"):
        return _load_student_bot(s.split(":", 1)[1], s)
    if s.startswith("model:"):
        s = s[len("model:"):]
    if s.endswith(".zip") or os.path.isdir(s) or os.path.isdir(os.path.join(CHALLENGE_ROOT, s)):
        return ModelAgent(_resolve_path(s), **kwargs)
    raise ValueError(f"Unknown agent spec: {spec!r}")


def _load_student_bot(name: str, spec: str) -> Agent:
    """Spec "my:<name>" -> MY_BOTS[name]() from student/my_opponents.py (training only)."""
    import importlib
    import sys
    if CHALLENGE_ROOT not in sys.path:
        sys.path.insert(0, CHALLENGE_ROOT)
    module = importlib.import_module("student.my_opponents")
    bots = getattr(module, "MY_BOTS", {})
    if name not in bots:
        raise ValueError(f"Unknown bot {spec!r}: add it to MY_BOTS in student/my_opponents.py "
                         f"(available: {sorted(bots)})")
    agent = bots[name]()
    agent.name = spec
    return agent


def available_baselines():
    if not os.path.isdir(BASELINES_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(BASELINES_DIR) if f.endswith(".zip"))
