"""
YOUR reward function. This is your main design lever.

`train.py` and `student/train_advanced.py` both use `my_reward` from this file.

Signature:   my_reward(prev_state, state, events) -> float

  prev_state / state : egocentric game state before and after the step (dict):
      state["me"]      -> {"x", "y", "dir", "hp", "cooldown"}
      state["enemy"]   -> {"visible", "x", "y", "dir", "hp", "cooldown"}  (x/y/dir None if hidden in a bush)
      state["bullets"] -> list of {"x", "y", "dir", "mine"}
      state["tiles"], state["brick_hp"], state["step"], state["max_steps"], ...
  events : what happened to YOU in this step:
      shots_fired, hits_dealt, hits_taken, bricks_destroyed, bumped, bullets_cancelled,
      result ("win" / "lose" / "draw" / None), end_reason

Rules: keep the functions at module level (SubprocVecEnv needs to pickle them), and
remember that the reward only matters for training. The tournament only looks at who wins.
"""
from tanks.game import clear_line, direction_to
from tanks.rewards import DEFAULT_WEIGHTS, WeightedReward, approach_and_aim_reward, default_reward  # noqa: F401

# ----------------------------------------------------------------------------
# 1) Event weights. Start by tuning these (copy of tanks.rewards.DEFAULT_WEIGHTS)
# ----------------------------------------------------------------------------
MY_WEIGHTS = {
    "win": 10.0,
    "lose": -10.0,
    "draw": -2.0,             # negative: hiding until the timeout should not pay off
    "hits_dealt": 2.0,
    "hits_taken": -2.0,
    "bricks_destroyed": 0.1,
    "shots_fired": -0.02,     # small cost per shot, otherwise the agent spams fire
    "bumped": -0.05,          # bumped into a wall / water / the other tank
    "step": -0.005,           # small time pressure
}
weighted = WeightedReward(MY_WEIGHTS)


# ----------------------------------------------------------------------------
# 2) Shaping (optional). Extra terms computed from the state.
# ----------------------------------------------------------------------------
def enemy_distance(state):
    """Manhattan distance to the enemy, or None if it is hidden."""
    me, en = state["me"], state["enemy"]
    if not en["visible"]:
        return None
    return abs(me["x"] - en["x"]) + abs(me["y"] - en["y"])


def enemy_in_my_sights(state):
    """True if I am facing the enemy with a clear line of fire."""
    me, en = state["me"], state["enemy"]
    return bool(en["visible"]
                and direction_to(me["x"], me["y"], en["x"], en["y"]) == me["dir"]
                and clear_line(state["tiles"], me["x"], me["y"], en["x"], en["y"]))


USE_SHAPING = False            # TODO: try True and compare the curves in TensorBoard


def my_reward(prev_state, state, events):
    r = weighted(prev_state, state, events)

    if USE_SHAPING:
        # TODO: design your own terms. Beware: shaping that is too strong gets exploited
        # (e.g. an agent that stays in sight forever without killing).
        d0, d1 = enemy_distance(prev_state), enemy_distance(state)
        if d0 is not None and d1 is not None:
            r += 0.02 * (d0 - d1)          # getting closer
        if enemy_in_my_sights(state) and state["me"]["cooldown"] == 0:
            r += 0.01                      # aiming with a loaded gun

    return float(r)
