"""
StudentTanksEnv: TanksEnv + map sets + hooks for callbacks (PFSP, reward blend).

    env = StudentTanksEnv(opponent={...}, maps="mixed", reward_fn=...)
    vec_env.env_method("set_pool_weights", {...})     # keeps the OpponentPool model cache
    vec_env.env_method("set_reward_param", "blend", 0.5)

Map sets: "official", "official_aug" (official x {orig, mirror, rot90, rot90+mirror}),
"random", "mixed" (50 % official_aug + 50 % random), "gen" (student/maps/gen, see map_gen.py),
"mixed2" (35 % official_aug + 30 % random + 35 % gen). A list is passed through unchanged.

Prepared starts: with probability `scenario_prob`, after the reset both tanks get random valid
positions (>= 4 apart, connected), random directions, HP 1-3, cooldowns, and in half of those
cases step_count in [250, 480]. A share `close_prob` of them are close combat instead
(half "diag": diagonally adjacent; half "faceoff": aligned 1-5 cells apart facing each other).
info["scenario"] = False or the kind of start ("random", "diag", "faceoff").

    python student/envs.py        # (re)generates student/maps/*.txt and checks them
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # challenge root

from tanks import TanksEnv  # noqa: E402
from tanks.agents.opponent_pool import OpponentPool  # noqa: E402
from tanks.game import BUSH, DRIVABLE, DX, DY, clear_line, direction_to  # noqa: E402
from tanks.observations import vector_obs  # noqa: E402
from tanks.maps import MAPS_DIR, list_maps, load_map  # noqa: E402
from tanks.maps.generator import connected, is_symmetric  # noqa: E402

AUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")
GEN_DIR = os.path.join(AUG_DIR, "gen")

# The maps are point-symmetric (rot180), so these 4 cover the whole dihedral group.
VARIANTS = {
    "orig": lambda a: a,
    "mirror": lambda a: a[:, ::-1],
    "rot90": lambda a: np.rot90(a),
    "rot90m": lambda a: np.rot90(a)[:, ::-1],
}


def _official_aug():
    paths = sorted(glob.glob(os.path.join(AUG_DIR, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No maps in {AUG_DIR}: run `python student/envs.py`")
    return paths


def resolve_maps(maps):
    if maps is None or not isinstance(maps, str):
        return maps
    if maps == "official":
        return list_maps()
    if maps == "official_aug":
        return _official_aug()
    if maps == "random":
        return ["random"]
    if maps == "mixed":
        aug = _official_aug()
        return aug + ["random"] * len(aug)
    if maps == "gen":
        return _gen()
    if maps == "mixed2":
        # TanksEnv samples the list uniformly: repeat entries to get 35 / 30 / 35
        aug, gen = _official_aug(), _gen()
        return aug * round(len(gen) / len(aug)) + ["random"] * round(len(gen) * 30 / 35) + gen
    raise ValueError(f"Unknown map set {maps!r}")


class StudentTanksEnv(TanksEnv):
    def __init__(self, *args, maps="mixed", scenario_prob=0.0, close_prob=0.5, **kwargs):
        super().__init__(*args, maps=resolve_maps(maps), **kwargs)
        self.scenario_prob = scenario_prob
        self.close_prob = close_prob

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["scenario"] = False
        if self.scenario_prob > 0 and self.np_random.random() < self.scenario_prob:
            info["scenario"] = self._prepare_start(self.np_random)
            self._state = self.game.get_state(self.side)
            obs = vector_obs(self._state)
        return obs, info

    def _prepare_start(self, rng):
        """Prepared start; returns its kind ("random", "diag", "faceoff") or False if none was placed."""
        g, cfg = self.game, self.game.cfg
        kind = "random"
        if rng.random() < self.close_prob:
            kind = "diag" if rng.random() < 0.5 else "faceoff"
        placed = {"random": _place_random, "diag": _place_diag, "faceoff": _place_faceoff}[kind](g.tiles, rng)
        if placed is None:
            return False   # ponytail: keep the normal start if no valid placement (tiny/odd maps only)
        poses, cooldowns = placed
        if cooldowns is None:
            cooldowns = rng.integers(cfg.fire_cooldown + 1, size=2)
        for t, (x, y, d), cd in zip(g.tanks, poses, cooldowns):
            t.x, t.y, t.dir, t.cooldown = int(x), int(y), int(d), int(cd)
            t.hp = int(rng.integers(1, cfg.max_hp + 1))
        for t, en in zip(g.tanks, g.tanks[::-1]):
            t.damage_dealt = cfg.max_hp - en.hp            # keeps the timeout tie-break consistent
        if rng.random() < 0.5:
            g.step_count = int(rng.integers(250, 481)) * cfg.max_steps // 500
        return kind

    def set_pool_weights(self, weights):
        if isinstance(self.opponent, OpponentPool):
            self.opponent.set_weights(weights)
        else:
            self.set_opponent(weights)

    def set_reward_param(self, name, value):
        setattr(self.reward_fn, name, value)


def _gen():
    paths = sorted(glob.glob(os.path.join(GEN_DIR, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No maps in {GEN_DIR}: run `python student/map_gen.py`")
    return paths


def _drivable(tiles, x, y):
    return 0 <= x < tiles.shape[1] and 0 <= y < tiles.shape[0] and tiles[y, x] in DRIVABLE


def _place_random(tiles, rng, min_dist=4, tries=100):
    """Two connected cells >= min_dist apart, random directions."""
    cells = np.argwhere(np.isin(tiles, DRIVABLE))       # (y, x)
    for _ in range(tries):
        (y0, x0), (y1, x1) = cells[rng.choice(len(cells), 2, replace=False)]
        if abs(x0 - x1) + abs(y0 - y1) >= min_dist and connected(tiles, (x0, y0), (x1, y1), allow_bricks=True):
            return [(x0, y0, rng.integers(4)), (x1, y1, rng.integers(4))], None
    return None


def _place_diag(tiles, rng, tries=200):
    """Diagonally adjacent. 30 % of the time we look for a blocked common cell (water, wall...).
    Half of the cases with a free common cell: both tanks face that cell (the deadlock of league 1)."""
    want_blocked = rng.random() < 0.3
    cells = np.argwhere(np.isin(tiles, DRIVABLE))
    found = fallback = None
    for _ in range(tries):
        y0, x0 = cells[rng.integers(len(cells))]
        dx, dy = rng.choice([-1, 1], size=2)
        x1, y1 = x0 + dx, y0 + dy
        if not _drivable(tiles, x1, y1):
            continue
        common = [(x1, y0), (x0, y1)]
        free = [c for c in common if _drivable(tiles, *c)]
        if not connected(tiles, (x0, y0), (x1, y1), allow_bricks=True):
            continue
        fallback = fallback or (x0, y0, x1, y1, free)
        if want_blocked == (len(free) < 2):
            found = (x0, y0, x1, y1, free)
            break
    if (found or fallback) is None:
        return None
    x0, y0, x1, y1, free = found or fallback
    if free and rng.random() < 0.5:
        cx, cy = free[rng.integers(len(free))]
        d0, d1 = direction_to(x0, y0, cx, cy), direction_to(x1, y1, cx, cy)
    else:
        d0, d1 = rng.integers(4, size=2)
    return [(x0, y0, d0), (x1, y1, d1)], None


def _place_faceoff(tiles, rng, tries=200):
    """Aligned 1-5 cells apart with a clear line of fire, facing and seeing each other.
    Cooldowns equal in half of the cases (as after cancelling each other's bullets)."""
    cells = np.argwhere(np.isin(tiles, DRIVABLE))
    for _ in range(tries):
        y0, x0 = cells[rng.integers(len(cells))]
        d = int(rng.integers(4))
        k = int(rng.integers(1, 6))
        x1, y1 = x0 + DX[d] * k, y0 + DY[d] * k
        if _drivable(tiles, x1, y1) and clear_line(tiles, x0, y0, x1, y1) \
                and (k <= 2 or BUSH not in (tiles[y0, x0], tiles[y1, x1])):   # they must see each other
            cd = rng.integers(6, size=2)          # 0..fire_cooldown (default rules)
            if rng.random() < 0.5:
                cd[1] = cd[0]
            else:
                while cd[1] == cd[0]:
                    cd[1] = rng.integers(6)
            return [(x0, y0, d), (x1, y1, (d + 2) % 4)], cd
    return None


def check_map(path):
    m = load_map(path)
    (x0, y0, _), (x1, y1, _) = m.spawns
    assert is_symmetric(m.tiles), f"{path}: not point-symmetric"
    assert (x1, y1) == (m.width - 1 - x0, m.height - 1 - y0), f"{path}: spawns not symmetric"
    assert m.tiles[y0, x0] in DRIVABLE and m.tiles[y1, x1] in DRIVABLE, f"{path}: spawn not drivable"
    assert connected(m.tiles, (x0, y0), (x1, y1), allow_bricks=True), f"{path}: spawns not connected"


def build_official_aug():
    os.makedirs(AUG_DIR, exist_ok=True)
    for name in list_maps():
        with open(os.path.join(MAPS_DIR, name + ".txt"), encoding="utf-8") as f:
            grid = np.array([list(ln.rstrip("\n\r")) for ln in f if ln.strip() and not ln.startswith(";")])
        for var, fn in VARIANTS.items():
            path = os.path.join(AUG_DIR, f"{name}__{var}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"; {name} ({var}) - generated by student/envs.py\n")
                f.write("\n".join("".join(r) for r in fn(grid)) + "\n")
            check_map(path)
            print("ok", os.path.relpath(path))


if __name__ == "__main__":
    build_official_aug()
    env = StudentTanksEnv(opponent={"basic": 1.0, "random": 1.0}, maps="mixed")
    assert len(env.maps) == 2 * 4 * len(list_maps())
    pool = env.opponent
    env.reset(seed=0)
    env.set_pool_weights({"random": 1.0})
    assert env.opponent is pool and pool.specs == ["random"] and pool._cache
    print("StudentTanksEnv OK")
