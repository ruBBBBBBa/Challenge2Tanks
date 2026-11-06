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
cases step_count in [250, 480]. info["scenario"] tells which episodes started that way.

    python student/envs.py        # (re)generates student/maps/*.txt and checks them
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # challenge root

from tanks import TanksEnv  # noqa: E402
from tanks.agents.opponent_pool import OpponentPool  # noqa: E402
from tanks.game import DRIVABLE  # noqa: E402
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
    def __init__(self, *args, maps="mixed", scenario_prob=0.0, **kwargs):
        super().__init__(*args, maps=resolve_maps(maps), **kwargs)
        self.scenario_prob = scenario_prob

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["scenario"] = False
        if self.scenario_prob > 0 and self.np_random.random() < self.scenario_prob:
            self._prepare_start(self.np_random)
            self._state = self.game.get_state(self.side)
            obs, info["scenario"] = vector_obs(self._state), True
        return obs, info

    def _prepare_start(self, rng, min_dist=4, tries=100):
        g, cfg = self.game, self.game.cfg
        cells = np.argwhere(np.isin(g.tiles, DRIVABLE))       # (y, x)
        for _ in range(tries):
            (y0, x0), (y1, x1) = cells[rng.choice(len(cells), 2, replace=False)]
            if abs(x0 - x1) + abs(y0 - y1) >= min_dist and connected(g.tiles, (x0, y0), (x1, y1), allow_bricks=True):
                break
        else:
            return   # ponytail: keep the normal start if no valid pair is found (tiny maps only)
        for t, (x, y) in zip(g.tanks, ((x0, y0), (x1, y1))):
            t.x, t.y = int(x), int(y)
            t.dir = int(rng.integers(4))
            t.hp = int(rng.integers(1, cfg.max_hp + 1))
            t.cooldown = int(rng.integers(cfg.fire_cooldown + 1))
        for t, en in zip(g.tanks, g.tanks[::-1]):
            t.damage_dealt = cfg.max_hp - en.hp            # keeps the timeout tie-break consistent
        if rng.random() < 0.5:
            g.step_count = int(rng.integers(250, 481)) * cfg.max_steps // 500

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
