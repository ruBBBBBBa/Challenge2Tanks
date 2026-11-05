# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
from tanks.agents.base import Agent, available_baselines, load_agent
from tanks.agents.heuristics import (PRESETS, AggressiveHeuristic, BasicHeuristic, DefensiveHeuristic,
                               ParamHeuristic)
from tanks.agents.model_agent import ModelAgent, load_sb3_model
from tanks.agents.opponent_pool import OpponentPool
from tanks.agents.random_agent import RandomAgent

OFFICIAL_BOTS = ["random", "basic", "aggressive", "defensive"]

__all__ = [
    "Agent", "load_agent", "available_baselines", "PRESETS", "ParamHeuristic", "BasicHeuristic",
    "AggressiveHeuristic", "DefensiveHeuristic", "ModelAgent", "load_sb3_model", "OpponentPool",
    "RandomAgent", "OFFICIAL_BOTS",
]
