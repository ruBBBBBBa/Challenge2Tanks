# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""Tanks challenge engine: game rules, Gymnasium env, official observation, bots and maps."""
from tanks.agents import OFFICIAL_BOTS, Agent, available_baselines, load_agent
from tanks.curriculum import WinRateCurriculum
from tanks.evaluation import evaluate_agent
from tanks.tanks_env import TanksEnv

__all__ = ["TanksEnv", "Agent", "load_agent", "available_baselines", "OFFICIAL_BOTS",
           "WinRateCurriculum", "evaluate_agent"]
