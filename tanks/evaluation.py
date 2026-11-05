# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""Win rate of an agent against a list of opponents (used by tools/evaluate.py and the Teacher scripts)."""
from typing import List, Optional

import numpy as np

from tanks.agents import OFFICIAL_BOTS, available_baselines, load_agent
from tanks.maps import list_maps
from tanks.match import play_match


def evaluate_agent(agent_spec, opponents: Optional[List[str]] = None, games: int = 20,
                   maps: Optional[List[str]] = None, seed: int = 0, verbose: bool = True) -> dict:
    """Returns {opponent: {"win": w, "draw": d, "lose": l, "win_rate": ...}}."""
    opponents = opponents or OFFICIAL_BOTS + [f"baseline:{b}" for b in available_baselines()]
    maps = maps or list_maps() + ["random"]
    agent = load_agent(agent_spec)
    rng = np.random.default_rng(seed)
    results = {}
    for opp_spec in opponents:
        opp = load_agent(opp_spec)
        counts = {"win": 0, "draw": 0, "lose": 0}
        damage = []
        for g in range(games):
            side = g % 2
            game_map = maps[(g // 2) % len(maps)]
            players = (agent, opp) if side == 0 else (opp, agent)
            res = play_match(players[0], players[1], game_map, seed=int(rng.integers(2**31 - 1)))
            if res["winner"] is None:
                counts["draw"] += 1
            elif res["winner"] == side:
                counts["win"] += 1
            else:
                counts["lose"] += 1
            damage.append(res["damage"][side])
        counts["win_rate"] = counts["win"] / games
        counts["avg_damage"] = float(np.mean(damage))
        results[opp_spec] = counts
        if verbose:
            print(f"vs {opp_spec:22s} win {counts['win']:3d}  draw {counts['draw']:3d}  lose {counts['lose']:3d}"
                  f"  | win rate {100 * counts['win_rate']:5.1f}%  avg damage {counts['avg_damage']:.2f}")
    return results
