# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Map loading.

    load_map("arena_01")            -> maps/arena_01.txt
    load_map("path/to/my_map.txt")  -> any file
    load_map("random", rng)         -> random symmetric map (maps/generator.py)
    list_maps()                     -> names of the official public maps

Map files are ASCII, one character per cell:
    .  empty      #  wall (indestructible)     B  brick (destructible)
    ~  water (blocks tanks, not bullets)       *  bush (hides tanks)
    1  spawn player 0                          2  spawn player 1
Lines starting with ';' are comments. Maps must be point-symmetric (180 degrees).
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from tanks.game import GameMap
from tanks.maps.generator import random_map, is_symmetric

MAPS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def list_maps(directory: str = MAPS_DIR) -> List[str]:
    return sorted(f[:-4] for f in os.listdir(directory) if f.endswith(".txt"))


def load_map(name: str, rng: Optional[np.random.Generator] = None) -> GameMap:
    if name == "random":
        return random_map(rng)
    path = name if name.endswith(".txt") else os.path.join(MAPS_DIR, name + ".txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Map not found: {name} (available: {list_maps()})")
    key = os.path.abspath(path)
    if key not in _CACHE:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        game_map = GameMap.from_ascii(lines, name=os.path.splitext(os.path.basename(path))[0])
        if not is_symmetric(game_map.tiles):
            print(f"[maps] Warning: map '{game_map.name}' is not point-symmetric (unfair for one side)")
        _CACHE[key] = game_map
    return _CACHE[key]   # TanksGame copies the tiles, the cached map is never modified


__all__ = ["load_map", "list_maps", "random_map", "MAPS_DIR"]
