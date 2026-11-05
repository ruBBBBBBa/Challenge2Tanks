# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Reward functions.  <-- This is one of the main things you should experiment with.

A reward function has the signature

    reward_fn(prev_state, state, events) -> float

  prev_state / state : egocentric states before and after the step (see game.TanksGame.get_state)
  events             : what happened to YOU during the step (see game.new_events):
                       shots_fired, hits_dealt, hits_taken, bricks_destroyed, bumped,
                       bullets_cancelled, result ("win"/"lose"/"draw"/None), end_reason

Pass it to the environment:  TanksEnv(opponent="basic", reward_fn=my_reward)
(With SubprocVecEnv define it at module level so it can be pickled.)
"""
from __future__ import annotations

from typing import Dict, Optional

DEFAULT_WEIGHTS: Dict[str, float] = {
    "win": 10.0,
    "lose": -10.0,
    "draw": -2.0,            # discourage hiding until the timeout
    "hits_dealt": 2.0,
    "hits_taken": -2.0,
    "bricks_destroyed": 0.1,
    "shots_fired": -0.02,
    "bumped": -0.05,
    "step": -0.005,
}


class WeightedReward:
    """Linear combination of the events. `WeightedReward(hits_dealt=3.0)` overrides one weight."""

    def __init__(self, weights: Optional[Dict[str, float]] = None, **overrides: float):
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weights.update(weights or {})
        self.weights.update(overrides)

    def __call__(self, prev_state: dict, state: dict, events: dict) -> float:
        wt = self.weights
        r = wt["step"]
        for key in ("hits_dealt", "hits_taken", "bricks_destroyed", "shots_fired", "bumped"):
            r += wt[key] * events[key]
        if events["result"] is not None:
            r += wt[events["result"]]
        return float(r)


default_reward = WeightedReward()


# --------------------------------------------------------------------------
# Example of shaping (TODO: design your own!)
# --------------------------------------------------------------------------
def _dist(state: dict) -> Optional[int]:
    me, en = state["me"], state["enemy"]
    if not en["visible"]:
        return None
    return abs(me["x"] - en["x"]) + abs(me["y"] - en["y"])


def approach_and_aim_reward(prev_state: dict, state: dict, events: dict) -> float:
    """Default reward + small bonus for getting closer and for having the enemy in your sights."""
    from tanks.game import clear_line, direction_to

    r = default_reward(prev_state, state, events)
    d0, d1 = _dist(prev_state), _dist(state)
    if d0 is not None and d1 is not None:
        r += 0.02 * (d0 - d1)
    me, en = state["me"], state["enemy"]
    if en["visible"] and direction_to(me["x"], me["y"], en["x"], en["y"]) == me["dir"] \
            and clear_line(state["tiles"], me["x"], me["y"], en["x"], en["y"]):
        r += 0.01
    return r
