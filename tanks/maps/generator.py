# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Random point-symmetric map generator.

Every feature placed at (x, y) is also placed at its 180-degree rotation
(W-1-x, H-1-y), so both players get exactly the same map from their side.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from tanks.game import BRICK, BUSH, DOWN, EMPTY, UP, WALL, WATER, DRIVABLE, GameMap


def rotate_cell(x: int, y: int, w: int, h: int):
    return w - 1 - x, h - 1 - y


def place_rect(tiles: np.ndarray, x: int, y: int, rw: int, rh: int, tile: int) -> None:
    """Place a rectangle and its point-symmetric copy."""
    h, w = tiles.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + rw), min(h, y + rh)
    tiles[y0:y1, x0:x1] = tile
    # rotated copy
    rx0, ry0 = w - x1, h - y1
    rx1, ry1 = w - x0, h - y0
    tiles[ry0:ry1, rx0:rx1] = tile


def is_symmetric(tiles: np.ndarray) -> bool:
    return bool((tiles == tiles[::-1, ::-1]).all())


def connected(tiles: np.ndarray, a, b, allow_bricks: bool = False) -> bool:
    """BFS between two cells over drivable tiles (optionally treating bricks as passable)."""
    h, w = tiles.shape
    ok = set(DRIVABLE) | ({BRICK} if allow_bricks else set())
    seen = {a}
    q = deque([a])
    while q:
        x, y = q.popleft()
        if (x, y) == b:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen and tiles[ny, nx] in ok:
                seen.add((nx, ny))
                q.append((nx, ny))
    return False


def random_map(rng: Optional[np.random.Generator] = None, width: int = 20, height: int = 20,
               n_features: int = 14, max_tries: int = 100) -> GameMap:
    """Generate a random point-symmetric map where both spawns are connected without breaking bricks."""
    rng = rng if rng is not None else np.random.default_rng()
    tile_choices = [WALL, BRICK, BRICK, WATER, BUSH, BUSH]
    for _ in range(max_tries):
        tiles = np.zeros((height, width), dtype=np.int8)
        for _ in range(n_features):
            tile = int(rng.choice(tile_choices))
            if rng.random() < 0.5:  # bar
                if rng.random() < 0.5:
                    rw, rh = int(rng.integers(2, 7)), 1
                else:
                    rw, rh = 1, int(rng.integers(2, 7))
            else:                   # block
                rw, rh = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            x = int(rng.integers(0, width - rw + 1))
            y = int(rng.integers(0, height - rh + 1))
            place_rect(tiles, x, y, rw, rh, tile)

        # spawn of player 1 in the top-right area, player 0 is its rotation (bottom-left)
        sx1, sy1 = int(rng.integers(width - 6, width - 1)), int(rng.integers(1, 5))
        sx0, sy0 = rotate_cell(sx1, sy1, width, height)
        place_rect(tiles, sx1 - 1, sy1 - 1, 3, 3, EMPTY)   # clear 3x3 around spawns
        if not is_symmetric(tiles):
            continue
        if not connected(tiles, (sx0, sy0), (sx1, sy1)):
            continue
        return GameMap(name="random", tiles=tiles, spawns=[(sx0, sy0, UP), (sx1, sy1, DOWN)])
    raise RuntimeError("Could not generate a valid random map")
