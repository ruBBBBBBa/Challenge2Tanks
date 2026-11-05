# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Hand-written opponents. The code is open on purpose: study it, exploit it,
and write your own bots (subclass Agent) to train against.

All presets are the same `ParamHeuristic` with different parameters:

    aggression  0..1   preferred fighting distance (1 = close combat, 0 = long range)
    caution     0..1   probability of dodging bullets / retreating when too close
    trigger     0..1   probability of firing when the enemy is in its sights
    noise       0..1   probability of a random action
    flank       0..1   probability of approaching from the side instead of walking
                       into the enemy's line of aim
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional, Tuple

import numpy as np

from tanks.agents.base import Agent
from tanks.game import (BACKWARD, BRICK, BULLET_BLOCKING, BUSH, DRIVABLE, DX, DY, FIRE, FORWARD, N_ACTIONS, NOOP,
                  TURN_LEFT, TURN_RIGHT, clear_line, direction_to, valid_actions)

PRESETS = {
    "basic": dict(aggression=0.5, caution=0.4, trigger=0.8, noise=0.10, flank=0.5),
    "aggressive": dict(aggression=1.0, caution=0.2, trigger=1.0, noise=0.03, flank=0.4),
    "defensive": dict(aggression=0.1, caution=0.9, trigger=0.9, noise=0.03, flank=0.9),
}

Cell = Tuple[int, int]


# --------------------------------------------------------------------------
# Small helpers (feel free to reuse them in your own bots)
# --------------------------------------------------------------------------
def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def turn_towards(current: int, desired: int) -> int:
    diff = (desired - current) % 4
    if diff == 0:
        return NOOP
    return TURN_LEFT if diff == 3 else TURN_RIGHT


def bfs_first_step(tiles: np.ndarray, start: Cell, is_goal: Callable[[Cell], bool],
                   passable: Callable[[Cell], bool]):
    """Breadth-first search. Returns (first direction or None if already at goal, goal cell) or None."""
    if is_goal(start):
        return None, start
    h, w = tiles.shape
    parent = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        for d in range(4):
            n = (c[0] + DX[d], c[1] + DY[d])
            if n in parent or not (0 <= n[0] < w and 0 <= n[1] < h) or not passable(n):
                continue
            parent[n] = c
            if is_goal(n):
                # walk back to the cell next to start
                cell = n
                while parent[cell] != start:
                    cell = parent[cell]
                first = direction_to(start[0], start[1], cell[0], cell[1])
                return first, n
            q.append(n)
    return None


def distance_field(tiles: np.ndarray, source: Cell, passable: Callable[[Cell], bool]) -> dict:
    h, w = tiles.shape
    dist = {source: 0}
    q = deque([source])
    while q:
        c = q.popleft()
        for d in range(4):
            n = (c[0] + DX[d], c[1] + DY[d])
            if n not in dist and 0 <= n[0] < w and 0 <= n[1] < h and passable(n):
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


def firing_positions(tiles: np.ndarray, target: Cell, max_range: int) -> set:
    """Cells from which a bullet can reach `target` (same row/column, no wall/brick between)."""
    h, w = tiles.shape
    grid = tiles.tolist()
    cells = set()
    for d in range(4):
        x, y = target
        for _ in range(max_range):
            x, y = x + DX[d], y + DY[d]
            if not (0 <= x < w and 0 <= y < h) or grid[y][x] in BULLET_BLOCKING:
                break
            cells.add((x, y))
    return cells


def danger_cells(state: dict, include_enemy_ray: bool = True, horizon: int = 3) -> set:
    """Cells crossed by enemy bullets in the next `horizon` steps (+ the enemy's line of aim)."""
    tiles = state["tiles"]
    h, w = tiles.shape
    grid = tiles.tolist()
    cells = set()

    def trace(x, y, d, n):
        for _ in range(n):
            x, y = x + DX[d], y + DY[d]
            if not (0 <= x < w and 0 <= y < h) or grid[y][x] in BULLET_BLOCKING:
                return
            cells.add((x, y))

    reach = state["config"]["bullet_speed"] * horizon
    for b in state["bullets"]:
        if not b["mine"]:
            cells.add((b["x"], b["y"]))
            trace(b["x"], b["y"], b["dir"], reach)
    en = state["enemy"]
    if include_enemy_ray and en["visible"]:
        trace(en["x"], en["y"], en["dir"], max(w, h))
    return cells


def threatened(state: dict, cell: Cell, horizon: int = 3) -> bool:
    """True if an enemy bullet will cross `cell` within `horizon` steps."""
    reach = state["config"]["bullet_speed"] * horizon
    for b in state["bullets"]:
        if b["mine"]:
            continue
        if direction_to(b["x"], b["y"], cell[0], cell[1]) == b["dir"] \
                and manhattan((b["x"], b["y"]), cell) <= reach \
                and clear_line(state["tiles"], b["x"], b["y"], cell[0], cell[1]):
            return True
    return False


def enemy_in_sights(state: dict) -> bool:
    me, en = state["me"], state["enemy"]
    if not en["visible"]:
        return False
    return direction_to(me["x"], me["y"], en["x"], en["y"]) == me["dir"] and \
        clear_line(state["tiles"], me["x"], me["y"], en["x"], en["y"])


def me_in_enemy_sights(state: dict) -> bool:
    me, en = state["me"], state["enemy"]
    if not en["visible"]:
        return False
    return direction_to(en["x"], en["y"], me["x"], me["y"]) == en["dir"] and \
        clear_line(state["tiles"], me["x"], me["y"], en["x"], en["y"])


# --------------------------------------------------------------------------
# Parametric heuristic
# --------------------------------------------------------------------------
class ParamHeuristic(Agent):
    def __init__(self, aggression: float = 0.5, caution: float = 0.5, trigger: float = 0.9,
                 noise: float = 0.05, flank: float = 0.6, name: Optional[str] = None,
                 seed: Optional[int] = None):
        self.aggression = float(aggression)
        self.caution = float(caution)
        self.trigger = float(trigger)
        self.noise = float(noise)
        self.flank = float(flank)
        self.name = name or f"param(a={aggression},c={caution},t={trigger},n={noise},f={flank})"
        self.rng = np.random.default_rng(seed)
        self.last_seen: Optional[Cell] = None
        self.wander: Optional[Cell] = None

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.last_seen = None
        self.wander = None

    # ---------------------------------------------------------------- policy
    def act(self, state: dict) -> int:
        tiles = state["tiles"]
        me, en = state["me"], state["enemy"]
        pos = (me["x"], me["y"])
        enemy_cell = (en["x"], en["y"]) if en["visible"] else None

        if enemy_cell is not None:
            self.last_seen = enemy_cell
        elif self.last_seen is not None and manhattan(pos, self.last_seen) <= 1:
            self.last_seen = None   # it was here, now it is gone

        if self.rng.random() < self.noise:
            return int(self.rng.integers(N_ACTIONS))

        # decisions for this step
        self._avoid_bullets = self.rng.random() < 0.5 + 0.5 * self.caution
        self._flanking = enemy_cell is not None and self.rng.random() < self.flank
        self._danger = danger_cells(state, include_enemy_ray=self._flanking) if (
            self._avoid_bullets or self._flanking) else set()

        # 1) dodge incoming bullets
        if threatened(state, pos) and self.rng.random() < self.caution:
            a = self._dodge(state)
            if a is not None:
                return a

        # 2) shoot
        if me["cooldown"] == 0 and enemy_in_sights(state) and self.rng.random() < self.trigger:
            return FIRE

        # 3) aligned with a clear line: face the enemy (or step aside while reloading)
        if enemy_cell is not None:
            d = direction_to(pos[0], pos[1], enemy_cell[0], enemy_cell[1])
            if d is not None and clear_line(tiles, pos[0], pos[1], enemy_cell[0], enemy_cell[1]):
                targeted = me_in_enemy_sights(state)
                # it aims at me and I cannot answer right now -> leave the line
                slow_answer = me["cooldown"] > 1 or (d - me["dir"]) % 4 == 2
                if targeted and slow_answer and self.rng.random() < 0.3 + 0.7 * self.caution:
                    a = self._sidestep(state, d)
                    if a is not None:
                        return a
                if d != me["dir"]:
                    return turn_towards(me["dir"], d)
                if self.aggression > 0.7 and not targeted and valid_actions(state)[FORWARD]:
                    return FORWARD
                return NOOP

        # 4) movement
        target = self.last_seen or self._wander_target(state, pos)
        preferred = int(round(2 + (1.0 - self.aggression) * 8))
        if enemy_cell is not None and manhattan(pos, enemy_cell) < preferred - 2 \
                and self.rng.random() < self.caution:
            return self._retreat(state, enemy_cell)
        return self._go_to_firing_position(state, target, preferred)

    # --------------------------------------------------------------- helpers
    def _passable(self, state: dict) -> Callable[[Cell], bool]:
        """Fast passability test (cached in the state dict): drivable and not the enemy cell."""
        if "_free" not in state:
            free = [[t in DRIVABLE for t in row] for row in state["tiles"].tolist()]
            en = state["enemy"]
            if en["visible"]:
                free[en["y"]][en["x"]] = False
            state["_free"] = free
        free = state["_free"]
        return lambda c: free[c[1]][c[0]]

    def _move(self, state: dict, d: int) -> int:
        """Action that moves (or prepares to move) in absolute direction d."""
        me = state["me"]
        if d == me["dir"]:
            nx, ny = me["x"] + DX[d], me["y"] + DY[d]
            h, w = state["tiles"].shape
            if 0 <= nx < w and 0 <= ny < h and state["tiles"][ny, nx] == BRICK:
                return FIRE if me["cooldown"] == 0 else NOOP
            return FORWARD
        if d == (me["dir"] + 2) % 4 and valid_actions(state)[BACKWARD]:
            return BACKWARD
        return turn_towards(me["dir"], d)

    def _go_to_firing_position(self, state: dict, target: Cell, preferred: int) -> int:
        tiles = state["tiles"]
        pos = (state["me"]["x"], state["me"]["y"])
        max_range = max(preferred, 3)

        firing_cells = firing_positions(tiles, target, max_range)
        base = self._passable(state)
        danger = getattr(self, "_danger", set())
        res = None
        if danger:
            # first try: stay out of bullet paths (and out of the enemy's aim when flanking)
            safe_goals = firing_cells - danger
            res = bfs_first_step(tiles, pos, safe_goals.__contains__,
                                 lambda c: base(c) and c not in danger)
        if res is None:
            res = bfs_first_step(tiles, pos, firing_cells.__contains__, base)
        if res is None:
            # no reachable firing position: dig through bricks towards the target
            grid = tiles.tolist()
            passable_or_brick = lambda c: grid[c[1]][c[0]] in DRIVABLE or grid[c[1]][c[0]] == BRICK
            res = bfs_first_step(tiles, pos, lambda c: manhattan(c, target) <= 1, passable_or_brick)
        if res is None:
            mask = valid_actions(state)
            return int(self.rng.choice(np.flatnonzero(mask)))
        d, goal = res
        if d is None:   # already at the firing cell: face the target
            fd = direction_to(pos[0], pos[1], target[0], target[1])
            if goal == target and self.last_seen is None:
                self.wander = None
            return turn_towards(state["me"]["dir"], fd) if fd is not None else NOOP
        return self._move(state, d)

    def _retreat(self, state: dict, enemy_cell: Cell) -> int:
        tiles, me = state["tiles"], state["me"]
        pos = (me["x"], me["y"])
        passable = self._passable(state)
        grid = tiles.tolist()
        dist = distance_field(tiles, enemy_cell, lambda c: grid[c[1]][c[0]] in DRIVABLE)
        best, best_score = None, -1e9
        for d in [None, 0, 1, 2, 3]:
            c = pos if d is None else (pos[0] + DX[d], pos[1] + DY[d])
            h, w = tiles.shape
            if d is not None and not (0 <= c[0] < w and 0 <= c[1] < h and passable(c)):
                continue
            score = dist.get(c, 0) - 5 * threatened(state, c)
            if self.caution > 0.5 and tiles[c[1], c[0]] == BUSH:
                score += 1.5
            if score > best_score:
                best, best_score = d, score
        if best is None:
            fd = direction_to(pos[0], pos[1], enemy_cell[0], enemy_cell[1])
            return turn_towards(me["dir"], fd) if fd is not None else NOOP
        # retreat backwards when possible (keeps the gun pointing at the enemy)
        return self._move(state, best)

    def _dodge(self, state: dict) -> Optional[int]:
        me = state["me"]
        pos = (me["x"], me["y"])
        # a bullet coming straight at the front can be shot down
        for b in state["bullets"]:
            if not b["mine"] and b["dir"] == (me["dir"] + 2) % 4 \
                    and direction_to(pos[0], pos[1], b["x"], b["y"]) == me["dir"] \
                    and clear_line(state["tiles"], pos[0], pos[1], b["x"], b["y"]) \
                    and me["cooldown"] == 0:
                return FIRE
        mask = valid_actions(state)
        for a, d in ((FORWARD, me["dir"]), (BACKWARD, (me["dir"] + 2) % 4)):
            c = (pos[0] + DX[d], pos[1] + DY[d])
            if mask[a] and not threatened(state, c):
                return a
        # turn so that next step we can leave the line of fire
        for d in ((me["dir"] + 1) % 4, (me["dir"] + 3) % 4):
            c = (pos[0] + DX[d], pos[1] + DY[d])
            h, w = state["tiles"].shape
            if 0 <= c[0] < w and 0 <= c[1] < h and self._passable(state)(c) and not threatened(state, c):
                return turn_towards(me["dir"], d)
        return None

    def _sidestep(self, state: dict, enemy_dir: int) -> Optional[int]:
        me = state["me"]
        pos = (me["x"], me["y"])
        mask = valid_actions(state)
        for a, d in ((FORWARD, me["dir"]), (BACKWARD, (me["dir"] + 2) % 4)):
            if d in (enemy_dir, (enemy_dir + 2) % 4):
                continue   # moving along the line of fire does not help
            if mask[a]:
                return a
        return None

    def _wander_target(self, state: dict, pos: Cell) -> Cell:
        tiles = state["tiles"]
        h, w = tiles.shape
        if self.wander is None or self.wander == pos:
            cells = np.argwhere(np.isin(tiles, DRIVABLE))
            # prefer the middle of the map
            center = np.array([h / 2, w / 2])
            d = np.abs(cells - center).sum(axis=1)
            candidates = cells[d <= np.percentile(d, 40)]
            y, x = candidates[self.rng.integers(len(candidates))]
            self.wander = (int(x), int(y))
        return self.wander


def BasicHeuristic(**kw) -> ParamHeuristic:
    return ParamHeuristic(name="basic", **{**PRESETS["basic"], **kw})


def AggressiveHeuristic(**kw) -> ParamHeuristic:
    return ParamHeuristic(name="aggressive", **{**PRESETS["aggressive"], **kw})


def DefensiveHeuristic(**kw) -> ParamHeuristic:
    return ParamHeuristic(name="defensive", **{**PRESETS["defensive"], **kw})
