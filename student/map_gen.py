"""
Our own point-symmetric 20x20 map generator (more variety than tanks/maps/generator.py).

    gen_map(rng, style="river", density=0.15, mix={"wall": 1, "brick": 2, "water": 1, "bush": 2})
    python student/map_gen.py --n 400          # -> student/maps/gen/gen_<style>_<k>.txt

Styles: open (few small features), maze (bars on a lattice), river (meandering water with fords),
bushes (large bush fields), classic (rectangles like the teacher's generator).
Spawns as in the teacher's generator (player 1 top right, player 0 its rotation, 3x3 cleared);
both spawns are always connected without breaking bricks.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # challenge root

from tanks.game import BRICK, BUSH, DOWN, EMPTY, UP, WALL, WATER, GameMap  # noqa: E402
from tanks.maps.generator import connected, is_symmetric, place_rect, rotate_cell  # noqa: E402

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "gen")
TILES = {"wall": WALL, "brick": BRICK, "water": WATER, "bush": BUSH}
STYLES = ("open", "maze", "river", "bushes", "classic")
W = H = 20


def _material(rng, mix):
    names = list(mix)
    p = np.array([mix[n] for n in names], dtype=float)
    return TILES[names[int(rng.choice(len(names), p=p / p.sum()))]]


def _feature(rng, tiles, mix, max_len=6, max_block=3):
    tile = _material(rng, mix)
    if rng.random() < 0.5:
        rw, rh = (int(rng.integers(2, max_len + 1)), 1) if rng.random() < 0.5 else (1, int(rng.integers(2, max_len + 1)))
    else:
        rw, rh = int(rng.integers(1, max_block + 1)), int(rng.integers(1, max_block + 1))
    place_rect(tiles, int(rng.integers(0, W - rw + 1)), int(rng.integers(0, H - rh + 1)), rw, rh, tile)


def _fill(rng, tiles, density, mix, **kw):
    """Add features until roughly `density` of the cells are obstacles/bushes."""
    for _ in range(200):
        if (tiles != EMPTY).mean() >= density:
            break
        _feature(rng, tiles, mix, **kw)


def _river(rng, tiles, mix):
    """Meandering water band through the centre (mirrored by place_rect) with 1-3 pairs of fords."""
    vertical = rng.random() < 0.5
    width = int(rng.integers(1, 3))

    def put(x, y, rw, rh, tile):  # (x, y) along / across the river
        place_rect(tiles, y, x, rh, rw, tile) if vertical else place_rect(tiles, x, y, rw, rh, tile)

    y = 9
    for x in range(10, -1, -1):           # x = 10 overlaps the mirrored half: no gap at the centre
        ny = int(np.clip(y + rng.choice([-1, 0, 0, 1]), 3, 15)) if x < 9 else y
        lo = min(y, ny)
        put(x, lo, 1, abs(ny - y) + width, WATER)   # covers the step: no diagonal leaks
        y = ny
    for _ in range(int(rng.integers(1, 4))):
        put(int(rng.integers(0, 10)), 2, 1, 16, EMPTY)
    _fill(rng, tiles, (tiles != EMPTY).mean() + 0.06, {k: v for k, v in mix.items() if k != "water"})


def _maze(rng, tiles, density, mix):
    step = int(rng.integers(3, 5))
    for y in range(1, H, step):
        for x in range(1, W, step):
            if rng.random() < 0.3 + density:
                n = int(rng.integers(2, step + 2))
                tile = _material(rng, {k: v for k, v in mix.items() if k != "bush"} or {"wall": 1})
                place_rect(tiles, x, y, n, 1, tile) if rng.random() < 0.5 else place_rect(tiles, x, y, 1, n, tile)


def _bushes(rng, tiles, density, mix):
    for _ in range(int(rng.integers(3, 7))):
        place_rect(tiles, int(rng.integers(0, W - 4)), int(rng.integers(0, H - 4)),
                   int(rng.integers(2, 6)), int(rng.integers(2, 6)), BUSH)
    _fill(rng, tiles, density, mix)


def gen_map(rng, style=None, density=None, mix=None, max_tries=200, name=None):
    style = style or STYLES[int(rng.integers(len(STYLES)))]
    for _ in range(max_tries):
        d = density if density is not None else float(rng.uniform(0.06, 0.3))
        m = mix or {k: float(rng.uniform(0.2, 1.0)) for k in TILES}
        tiles = np.zeros((H, W), dtype=np.int8)
        if style == "open":
            _fill(rng, tiles, min(d, 0.1), m, max_len=3, max_block=2)
        elif style == "maze":
            _maze(rng, tiles, d, m)
        elif style == "river":
            _river(rng, tiles, m)
            _fill(rng, tiles, (tiles != EMPTY).mean() + d / 2, m)
        elif style == "bushes":
            _bushes(rng, tiles, d, m)
        else:
            _fill(rng, tiles, d, m)

        sx1, sy1 = int(rng.integers(W - 6, W - 1)), int(rng.integers(1, 5))
        sx0, sy0 = rotate_cell(sx1, sy1, W, H)
        place_rect(tiles, sx1 - 1, sy1 - 1, 3, 3, EMPTY)
        if is_symmetric(tiles) and connected(tiles, (sx0, sy0), (sx1, sy1)):
            return GameMap(name=name or f"gen_{style}", tiles=tiles, spawns=[(sx0, sy0, UP), (sx1, sy1, DOWN)])
    raise RuntimeError(f"Could not generate a valid '{style}' map")


def write_maps(n, seed=0, out=GEN_DIR):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for k in range(n):
        style = STYLES[k % len(STYLES)]
        m = gen_map(rng, style)
        path = os.path.join(out, f"gen_{style}_{k:03d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"; gen_{style}_{k:03d} - student/map_gen.py seed={seed}\n" + "\n".join(m.to_ascii()) + "\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from student.envs import check_map
    paths = write_maps(args.n, args.seed)
    for p in paths:
        check_map(p)       # symmetric, spawns symmetric and drivable, reloads from the .txt
    print(f"{len(paths)} maps OK in {os.path.relpath(GEN_DIR)}")
