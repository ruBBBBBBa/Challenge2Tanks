# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Pure game logic for Tanks (no Gymnasium, no rendering).

The same rules are used by the training environment (tanks_env.py),
the match runner (match.py), the tournament and the replays.

Coordinates: (x, y) with x = column, y = row, y grows downwards.
Arrays are indexed as tiles[y, x].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
EMPTY, WALL, BRICK, WATER, BUSH = 0, 1, 2, 3, 4
TILE_NAMES = {EMPTY: "empty", WALL: "wall", BRICK: "brick", WATER: "water", BUSH: "bush"}
CHAR_TO_TILE = {".": EMPTY, "#": WALL, "B": BRICK, "~": WATER, "*": BUSH}
TILE_TO_CHAR = {v: k for k, v in CHAR_TO_TILE.items()}

# Tiles a tank can drive on / tiles that stop bullets
DRIVABLE = (EMPTY, BUSH)
BULLET_BLOCKING = (WALL, BRICK)

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIR_NAMES = ["up", "right", "down", "left"]
DX = (0, 1, 0, -1)
DY = (-1, 0, 1, 0)

NOOP, FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, FIRE = range(6)
N_ACTIONS = 6
ACTION_NAMES = ["noop", "forward", "backward", "turn_left", "turn_right", "fire"]


# --------------------------------------------------------------------------
# Configuration and map
# --------------------------------------------------------------------------
@dataclass
class GameConfig:
    """Game rules. The tournament always uses the default values."""

    max_hp: int = 3                 # hit points per tank (no respawn)
    fire_cooldown: int = 5          # steps between two shots
    bullet_speed: int = 2           # cells per step
    max_steps: int = 500            # then tie-break: more HP, then more damage dealt, else draw
    brick_hp: int = 2               # hits needed to destroy a brick
    bush_reveal_distance: int = 2   # a tank in a bush is visible only within this Chebyshev distance

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "GameConfig":
        if d is None:
            return cls()
        if isinstance(d, GameConfig):
            return d
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GameMap:
    name: str
    tiles: np.ndarray                       # (H, W) int8
    spawns: List[Tuple[int, int, int]]      # [(x, y, dir) for player 0, player 1]

    @property
    def width(self) -> int:
        return int(self.tiles.shape[1])

    @property
    def height(self) -> int:
        return int(self.tiles.shape[0])

    def to_ascii(self) -> List[str]:
        rows = [[TILE_TO_CHAR[int(t)] for t in row] for row in self.tiles]
        for i, (x, y, _) in enumerate(self.spawns):
            rows[y][x] = str(i + 1)
        return ["".join(r) for r in rows]

    @classmethod
    def from_ascii(cls, lines: Sequence[str], name: str = "custom") -> "GameMap":
        lines = [ln.rstrip("\n\r") for ln in lines if ln.strip() and not ln.lstrip().startswith(";")]
        h, w = len(lines), len(lines[0])
        tiles = np.zeros((h, w), dtype=np.int8)
        spawn_pos: Dict[int, Tuple[int, int]] = {}
        for y, row in enumerate(lines):
            if len(row) != w:
                raise ValueError(f"Map '{name}': row {y} has length {len(row)}, expected {w}")
            for x, ch in enumerate(row):
                if ch in ("1", "2"):
                    spawn_pos[int(ch) - 1] = (x, y)
                    tiles[y, x] = EMPTY
                elif ch in CHAR_TO_TILE:
                    tiles[y, x] = CHAR_TO_TILE[ch]
                else:
                    raise ValueError(f"Map '{name}': unknown character {ch!r} at ({x}, {y})")
        if set(spawn_pos) != {0, 1}:
            raise ValueError(f"Map '{name}' needs exactly one '1' and one '2' spawn")
        spawns = []
        for p in (0, 1):
            x, y = spawn_pos[p]
            spawns.append((x, y, UP if y >= h / 2 else DOWN))  # face the middle of the map
        return cls(name=name, tiles=tiles, spawns=spawns)


# --------------------------------------------------------------------------
# Game objects
# --------------------------------------------------------------------------
class Tank:
    __slots__ = ("x", "y", "dir", "hp", "cooldown", "damage_dealt", "shots_fired")

    def __init__(self, x: int, y: int, d: int, hp: int):
        self.x, self.y, self.dir, self.hp = x, y, d, hp
        self.cooldown = 0
        self.damage_dealt = 0
        self.shots_fired = 0

    @property
    def alive(self) -> bool:
        return self.hp > 0


class Bullet:
    __slots__ = ("x", "y", "dir", "owner")

    def __init__(self, x: int, y: int, d: int, owner: int):
        self.x, self.y, self.dir, self.owner = x, y, d, owner


def new_events() -> dict:
    """Per-player events of one step. Passed to the reward function."""
    return {
        "shots_fired": 0,
        "hits_dealt": 0,
        "hits_taken": 0,
        "bricks_destroyed": 0,
        "bumped": 0,           # tried to drive into something
        "bullets_cancelled": 0,
        "result": None,        # None while playing, then "win" / "lose" / "draw"
        "end_reason": None,    # "destroyed", "double_ko", "timeout"
    }


# --------------------------------------------------------------------------
# Game
# --------------------------------------------------------------------------
class TanksGame:
    """Two tanks, simultaneous moves."""

    def __init__(self, game_map: GameMap, config: Optional[GameConfig] = None):
        self.map = game_map
        self.cfg = GameConfig.from_dict(config) if not isinstance(config, GameConfig) else config
        self._cfg_dict = self.cfg.to_dict()
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.tiles = self.map.tiles.copy()
        self.brick_hp = np.where(self.tiles == BRICK, self.cfg.brick_hp, 0).astype(np.int8)
        self.tanks = [Tank(x, y, d, self.cfg.max_hp) for (x, y, d) in self.map.spawns]
        self.bullets: List[Bullet] = []
        self.step_count = 0
        self.done = False
        self.winner: Optional[int] = None   # 0, 1 or None (draw / not finished)
        self.end_reason: Optional[str] = None
        self.action_history: List[List[int]] = []

    @property
    def width(self) -> int:
        return self.map.width

    @property
    def height(self) -> int:
        return self.map.height

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def drivable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.tiles[y, x] in DRIVABLE

    # ------------------------------------------------------------------- step
    def step(self, actions: Sequence[int]) -> List[dict]:
        """Advance one step. `actions[i]` is the action of player i. Returns events per player."""
        if self.done:
            raise RuntimeError("Game is over, call reset()")
        acts = [int(a) if 0 <= int(a) < N_ACTIONS else NOOP for a in actions]
        self.action_history.append(acts)
        events = [new_events(), new_events()]

        # 1) cooldowns
        for t in self.tanks:
            if t.cooldown > 0:
                t.cooldown -= 1

        # 2) rotations
        for t, a in zip(self.tanks, acts):
            if a == TURN_LEFT:
                t.dir = (t.dir - 1) % 4
            elif a == TURN_RIGHT:
                t.dir = (t.dir + 1) % 4

        # 3) simultaneous movement
        self._move_tanks(acts, events)

        # 4) firing (the new bullet starts on the tank cell and moves in phase 5)
        for i, (t, a) in enumerate(zip(self.tanks, acts)):
            if a == FIRE and t.cooldown == 0:
                self.bullets.append(Bullet(t.x, t.y, t.dir, i))
                t.cooldown = self.cfg.fire_cooldown
                t.shots_fired += 1
                events[i]["shots_fired"] += 1

        # 5) bullets
        self._move_bullets(events)

        # 6) end of game
        self.step_count += 1
        self._check_end(events)
        return events

    def _move_tanks(self, acts: Sequence[int], events: List[dict]) -> None:
        targets = []
        for i, (t, a) in enumerate(zip(self.tanks, acts)):
            tx, ty = t.x, t.y
            if a in (FORWARD, BACKWARD):
                d = t.dir if a == FORWARD else (t.dir + 2) % 4
                nx, ny = t.x + DX[d], t.y + DY[d]
                if self.drivable(nx, ny):
                    tx, ty = nx, ny
                else:
                    events[i]["bumped"] += 1
            targets.append((tx, ty))

        cur = [(t.x, t.y) for t in self.tanks]
        # same target cell or swapping cells -> nobody moves
        if targets[0] == targets[1] or (targets[0] == cur[1] and targets[1] == cur[0]):
            for i in (0, 1):
                if targets[i] != cur[i]:
                    events[i]["bumped"] += 1
            targets = list(cur)
        # moving into a tank that stays -> blocked
        for _ in range(2):
            for i, j in ((0, 1), (1, 0)):
                if targets[i] != cur[i] and targets[i] == targets[j]:
                    targets[i] = cur[i]
                    events[i]["bumped"] += 1
        for t, (x, y) in zip(self.tanks, targets):
            t.x, t.y = x, y

    def _hit(self, bullet: Bullet, victim: int, events: List[dict]) -> None:
        self.tanks[victim].hp -= 1
        self.tanks[bullet.owner].damage_dealt += 1
        events[bullet.owner]["hits_dealt"] += 1
        events[victim]["hits_taken"] += 1

    def _move_bullets(self, events: List[dict]) -> None:
        # a tank that drove onto a bullet is hit
        survivors = []
        for b in self.bullets:
            victim = self._tank_at(b.x, b.y, exclude=b.owner)
            if victim is not None:
                self._hit(b, victim, events)
            else:
                survivors.append(b)
        self.bullets = survivors

        for _ in range(self.cfg.bullet_speed):
            if not self.bullets:
                break
            prev = [(b.x, b.y) for b in self.bullets]
            for b in self.bullets:
                b.x += DX[b.dir]
                b.y += DY[b.dir]

            removed = [False] * len(self.bullets)
            # leaving the map / walls
            for k, b in enumerate(self.bullets):
                if not self.in_bounds(b.x, b.y) or self.tiles[b.y, b.x] == WALL:
                    removed[k] = True
            # bullet vs bullet (same cell or crossing each other)
            n = len(self.bullets)
            for k in range(n):
                for m in range(k + 1, n):
                    if removed[k] or removed[m]:
                        continue
                    bk, bm = self.bullets[k], self.bullets[m]
                    same = (bk.x, bk.y) == (bm.x, bm.y)
                    crossed = (bk.x, bk.y) == prev[m] and (bm.x, bm.y) == prev[k]
                    if same or crossed:
                        removed[k] = removed[m] = True
                        events[bk.owner]["bullets_cancelled"] += 1
                        events[bm.owner]["bullets_cancelled"] += 1
            # bricks and tanks
            for k, b in enumerate(self.bullets):
                if removed[k]:
                    continue
                if self.tiles[b.y, b.x] == BRICK:
                    self.brick_hp[b.y, b.x] -= 1
                    if self.brick_hp[b.y, b.x] <= 0:
                        self.tiles[b.y, b.x] = EMPTY
                        self.brick_hp[b.y, b.x] = 0
                        events[b.owner]["bricks_destroyed"] += 1
                    removed[k] = True
                    continue
                victim = self._tank_at(b.x, b.y, exclude=b.owner)
                if victim is not None:
                    self._hit(b, victim, events)
                    removed[k] = True
            self.bullets = [b for b, r in zip(self.bullets, removed) if not r]

    def _tank_at(self, x: int, y: int, exclude: int = -1) -> Optional[int]:
        for i, t in enumerate(self.tanks):
            if i != exclude and t.alive and t.x == x and t.y == y:
                return i
        return None

    def _check_end(self, events: List[dict]) -> None:
        a0, a1 = self.tanks[0].alive, self.tanks[1].alive
        if not a0 and not a1:
            self._finish(None, "double_ko", events)
        elif not a0:
            self._finish(1, "destroyed", events)
        elif not a1:
            self._finish(0, "destroyed", events)
        elif self.step_count >= self.cfg.max_steps:
            t0, t1 = self.tanks
            if t0.hp != t1.hp:
                winner = 0 if t0.hp > t1.hp else 1
            elif t0.damage_dealt != t1.damage_dealt:
                winner = 0 if t0.damage_dealt > t1.damage_dealt else 1
            else:
                winner = None
            self._finish(winner, "timeout", events)

    def _finish(self, winner: Optional[int], reason: str, events: List[dict]) -> None:
        self.done = True
        self.winner = winner
        self.end_reason = reason
        for i in (0, 1):
            events[i]["end_reason"] = reason
            if winner is None:
                events[i]["result"] = "draw"
            else:
                events[i]["result"] = "win" if winner == i else "lose"

    # ------------------------------------------------------------------ state
    def enemy_visible(self, viewer: int) -> bool:
        me, en = self.tanks[viewer], self.tanks[1 - viewer]
        if self.tiles[en.y, en.x] != BUSH:
            return True
        return max(abs(me.x - en.x), abs(me.y - en.y)) <= self.cfg.bush_reveal_distance

    def get_state(self, player: int) -> dict:
        """
        Egocentric state for `player`: you are always "me" and the map is rotated
        180 degrees for player 1, so both players see the game from the same side.
        (Maps are point-symmetric, so a policy works on either side.)

        Keys:
          step, max_steps, width, height
          tiles      (H, W) int8 array, see EMPTY/WALL/BRICK/WATER/BUSH
          brick_hp   (H, W) int8 array
          me         dict(x, y, dir, hp, cooldown)
          enemy      dict(visible, x, y, dir, hp, cooldown)   x/y/dir are None when hidden in a bush
          bullets    list of dict(x, y, dir, mine)
          config     dict with the GameConfig values
        """
        flip = player == 1
        w, h = self.width, self.height

        def tx(x: int) -> int:
            return w - 1 - x if flip else x

        def ty(y: int) -> int:
            return h - 1 - y if flip else y

        def td(d: int) -> int:
            return (d + 2) % 4 if flip else d

        me, en = self.tanks[player], self.tanks[1 - player]
        tiles = self.tiles[::-1, ::-1].copy() if flip else self.tiles.copy()
        bhp = self.brick_hp[::-1, ::-1].copy() if flip else self.brick_hp.copy()
        visible = self.enemy_visible(player)
        return {
            "step": self.step_count,
            "max_steps": self.cfg.max_steps,
            "width": w,
            "height": h,
            "tiles": tiles,
            "brick_hp": bhp,
            "me": {"x": tx(me.x), "y": ty(me.y), "dir": td(me.dir), "hp": me.hp, "cooldown": me.cooldown},
            "enemy": {
                "visible": visible,
                "x": tx(en.x) if visible else None,
                "y": ty(en.y) if visible else None,
                "dir": td(en.dir) if visible else None,
                "hp": en.hp,
                "cooldown": en.cooldown if visible else None,
            },
            "bullets": [
                {"x": tx(b.x), "y": ty(b.y), "dir": td(b.dir), "mine": b.owner == player}
                for b in self.bullets
            ],
            "config": dict(self._cfg_dict),
        }


# --------------------------------------------------------------------------
# Helpers that work on an egocentric state (usable by heuristics and students)
# --------------------------------------------------------------------------
def valid_actions(state: dict) -> np.ndarray:
    """Boolean mask of length N_ACTIONS: actions that are not obviously useless.

    FORWARD/BACKWARD are invalid when the target cell is not drivable or holds the
    (visible) enemy, FIRE is invalid while on cooldown. NOOP and turns are always valid.
    Use it to write your own action masking or safety filters.
    """
    tiles = state["tiles"]
    h, w = tiles.shape
    me, en = state["me"], state["enemy"]
    mask = np.ones(N_ACTIONS, dtype=bool)
    for a, d in ((FORWARD, me["dir"]), (BACKWARD, (me["dir"] + 2) % 4)):
        nx, ny = me["x"] + DX[d], me["y"] + DY[d]
        ok = 0 <= nx < w and 0 <= ny < h and tiles[ny, nx] in DRIVABLE
        if ok and en["visible"] and (nx, ny) == (en["x"], en["y"]):
            ok = False
        mask[a] = ok
    mask[FIRE] = me["cooldown"] == 0
    return mask


def clear_line(tiles: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
    """True if (x0,y0) and (x1,y1) share a row/column and no wall/brick lies strictly between."""
    if x0 != x1 and y0 != y1:
        return False
    if x0 == x1:
        lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
        cells = tiles[lo + 1:hi, x0]
    else:
        lo, hi = (x0, x1) if x0 < x1 else (x1, x0)
        cells = tiles[y0, lo + 1:hi]
    for c in cells.tolist():
        if c == WALL or c == BRICK:
            return False
    return True


def direction_to(x0: int, y0: int, x1: int, y1: int) -> Optional[int]:
    """Cardinal direction from (x0,y0) to (x1,y1) if they are aligned, else None."""
    if x0 == x1 and y0 != y1:
        return DOWN if y1 > y0 else UP
    if y0 == y1 and x0 != x1:
        return RIGHT if x1 > x0 else LEFT
    return None
