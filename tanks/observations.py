# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Official vector observation.

IMPORTANT: the tournament feeds your model.zip with exactly this observation
(`vector_obs(state)`), so your model must be trained on it. You may change the
reward, the algorithm, the network and the opponents, but not the observation.

Most features are expressed in the tank's own frame ("forward" / "right"),
which matches the relative action space (forward, backward, turn, fire).

Layout (OBS_SIZE = 77 floats, roughly in [-1, 1]):

  [0:8]   me        x, y (normalised 0..1), facing one-hot (up, right, down, left),
                    hp / max_hp, cooldown / fire_cooldown
  [8:20]  enemy     visible, forward offset, right offset, enemy facing relative to
                    mine one-hot (same, right, opposite, left), hp / max_hp,
                    cooldown / fire_cooldown, manhattan distance, enemy in my sights,
                    I am in its sights
                    (position/facing/cooldown are 0 when the enemy hides in a bush)
  [20:52] rays      8 rays (forward, forward-right, right, back-right, back,
                    back-left, left, forward-left), 4 values each:
                    distance to first wall/brick, that obstacle is a brick,
                    distance to first cell a tank cannot drive on, enemy on the ray (1 = adjacent)
  [52:76] bullets   3 nearest enemy bullets, 8 values each:
                    present, forward offset, right offset, bullet direction relative to
                    me one-hot, threat (flying towards me with nothing in between)
  [76]    time      step / max_steps
"""
from __future__ import annotations

import numpy as np

from tanks.game import BRICK, BULLET_BLOCKING, DRIVABLE, DX, DY, clear_line, direction_to

OBS_SIZE = 77
N_RAYS = 8
N_BULLETS = 3

OBS_NAMES = (
    ["me_x", "me_y", "me_dir_up", "me_dir_right", "me_dir_down", "me_dir_left", "me_hp", "me_cooldown"]
    + ["en_visible", "en_fwd", "en_right", "en_rel_same", "en_rel_right", "en_rel_opp", "en_rel_left",
       "en_hp", "en_cooldown", "en_dist", "en_in_my_sights", "me_in_its_sights"]
    + [f"ray{k}_{f}" for k in range(N_RAYS) for f in ("solid_dist", "is_brick", "block_dist", "enemy")]
    + [f"bullet{k}_{f}" for k in range(N_BULLETS)
       for f in ("present", "fwd", "right", "rel_same", "rel_right", "rel_opp", "rel_left", "threat")]
    + ["time"]
)
assert len(OBS_NAMES) == OBS_SIZE


def _frame(d: int):
    """Forward and right unit vectors for facing d (screen coords, y down)."""
    f = (DX[d], DY[d])
    r = (DX[(d + 1) % 4], DY[(d + 1) % 4])
    return f, r


def _rel(dx: int, dy: int, f, r):
    return dx * f[0] + dy * f[1], dx * r[0] + dy * r[1]


def vector_obs(state: dict) -> np.ndarray:
    tiles = state["tiles"]
    h, w = tiles.shape
    grid = tiles.tolist()   # python lists are much faster than numpy for cell-by-cell access
    cfg = state["config"]
    me, en = state["me"], state["enemy"]
    mx, my, md = me["x"], me["y"], me["dir"]
    f, r = _frame(md)
    scale = float(max(w, h))
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    # ---- me
    obs[0] = mx / (w - 1)
    obs[1] = my / (h - 1)
    obs[2 + md] = 1.0
    obs[6] = me["hp"] / cfg["max_hp"]
    obs[7] = me["cooldown"] / cfg["fire_cooldown"]

    # ---- enemy
    i = 8
    obs[i + 7] = en["hp"] / cfg["max_hp"]
    if en["visible"]:
        ex, ey, ed = en["x"], en["y"], en["dir"]
        fwd, right = _rel(ex - mx, ey - my, f, r)
        obs[i] = 1.0
        obs[i + 1] = fwd / scale
        obs[i + 2] = right / scale
        obs[i + 3 + (ed - md) % 4] = 1.0
        obs[i + 8] = en["cooldown"] / cfg["fire_cooldown"]
        obs[i + 9] = (abs(ex - mx) + abs(ey - my)) / (w + h)
        obs[i + 10] = float(direction_to(mx, my, ex, ey) == md and clear_line(tiles, mx, my, ex, ey))
        obs[i + 11] = float(direction_to(ex, ey, mx, my) == ed and clear_line(tiles, mx, my, ex, ey))

    # ---- rays
    i = 20
    for k in range(N_RAYS):
        q, rem = divmod(k, 2)
        # k even: cardinal (forward, right, back, left); k odd: diagonal between k-1 and k+1
        d1 = (md + q) % 4
        vx, vy = DX[d1], DY[d1]
        if rem:
            d2 = (md + q + 1) % 4
            vx, vy = vx + DX[d2], vy + DY[d2]
        solid_dist = block_dist = None
        is_brick = 0.0
        enemy = 0.0
        x, y, dist = mx, my, 0
        while True:
            x += vx
            y += vy
            dist += 1
            if not (0 <= x < w and 0 <= y < h):
                if solid_dist is None:
                    solid_dist = dist
                if block_dist is None:
                    block_dist = dist
                break
            t = grid[y][x]
            if block_dist is None and t not in DRIVABLE:
                block_dist = dist
            if t in BULLET_BLOCKING:
                solid_dist = dist
                is_brick = float(t == BRICK)
                break
            if en["visible"] and enemy == 0.0 and x == en["x"] and y == en["y"]:
                enemy = 1.0 - (dist - 1) / scale
        obs[i + 4 * k] = solid_dist / scale
        obs[i + 4 * k + 1] = is_brick
        obs[i + 4 * k + 2] = block_dist / scale
        obs[i + 4 * k + 3] = enemy

    # ---- enemy bullets
    i = 52
    enemy_bullets = [b for b in state["bullets"] if not b["mine"]]
    enemy_bullets.sort(key=lambda b: abs(b["x"] - mx) + abs(b["y"] - my))
    for k, b in enumerate(enemy_bullets[:N_BULLETS]):
        j = i + 8 * k
        fwd, right = _rel(b["x"] - mx, b["y"] - my, f, r)
        obs[j] = 1.0
        obs[j + 1] = fwd / scale
        obs[j + 2] = right / scale
        obs[j + 3 + (b["dir"] - md) % 4] = 1.0
        towards_me = direction_to(b["x"], b["y"], mx, my) == b["dir"]
        obs[j + 7] = float(towards_me and clear_line(tiles, b["x"], b["y"], mx, my))

    # ---- time
    obs[76] = state["step"] / state["max_steps"]
    return np.clip(obs, -1.0, 1.0)
