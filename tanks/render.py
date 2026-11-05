# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""Pygame renderer (window, rgb arrays) and GIF export. Never used inside the training loop
unless render_mode is set."""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

from tanks.game import BRICK, BUSH, DX, DY, WALL, WATER, TanksGame

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

COLORS = {
    "bg": (30, 30, 34),
    "grid": (40, 40, 46),
    WALL: (120, 120, 130),
    BRICK: (170, 90, 50),
    "brick_damaged": (120, 65, 40),
    WATER: (40, 90, 170),
    BUSH: (40, 130, 60),
    "p0": (70, 150, 255),
    "p1": (240, 80, 80),
    "bullet": (255, 230, 120),
    "text": (230, 230, 230),
    "hud": (20, 20, 24),
}


class Renderer:
    def __init__(self, cell: int = 30, hud: int = 44, title: str = "Tanks"):
        import pygame
        self.pg = pygame
        pygame.init()
        self.cell, self.hud, self.title = cell, hud, title
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 16)
        self.window = None
        self.clock = None
        self.closed = False

    def draw(self, game: TanksGame, names: Sequence[str] = ("P0", "P1"), extra: str = ""):
        pg, c = self.pg, self.cell
        w, h = game.width * c, game.height * c + self.hud
        surf = pg.Surface((w, h))
        surf.fill(COLORS["bg"])
        off = self.hud

        for y in range(game.height):
            for x in range(game.width):
                rect = pg.Rect(x * c, off + y * c, c, c)
                t = int(game.tiles[y, x])
                if t == WALL:
                    pg.draw.rect(surf, COLORS[WALL], rect)
                elif t == BRICK:
                    col = COLORS[BRICK] if game.brick_hp[y, x] >= game.cfg.brick_hp else COLORS["brick_damaged"]
                    pg.draw.rect(surf, col, rect)
                    pg.draw.line(surf, COLORS["bg"], rect.midleft, rect.midright, 1)
                elif t == WATER:
                    pg.draw.rect(surf, COLORS[WATER], rect)
                else:
                    pg.draw.rect(surf, COLORS["grid"], rect, 1)

        for i, t in enumerate(game.tanks):
            if not t.alive:
                continue
            col = COLORS["p0" if i == 0 else "p1"]
            cx, cy = t.x * c + c // 2, off + t.y * c + c // 2
            body = pg.Rect(0, 0, int(c * 0.8), int(c * 0.8))
            body.center = (cx, cy)
            pg.draw.rect(surf, col, body, border_radius=4)
            dark = tuple(int(v * 0.55) for v in col)
            pg.draw.line(surf, (235, 235, 235), (cx, cy),
                         (cx + DX[t.dir] * int(c * 0.62), cy + DY[t.dir] * int(c * 0.62)), 4)
            pg.draw.circle(surf, dark, (cx, cy), c // 4)

        for b in game.bullets:
            cx, cy = b.x * c + c // 2, off + b.y * c + c // 2
            pg.draw.circle(surf, COLORS["bullet"], (cx, cy), max(3, c // 7))

        # bushes are drawn on top (semi-transparent) so hidden tanks are hard to see
        bush = pg.Surface((c, c), pg.SRCALPHA)
        bush.fill((*COLORS[BUSH], 215))
        for y, x in np.argwhere(game.tiles == BUSH):
            surf.blit(bush, (x * c, off + y * c))

        # HUD
        pg.draw.rect(surf, COLORS["hud"], pg.Rect(0, 0, w, self.hud))
        for i, t in enumerate(game.tanks):
            col = COLORS["p0" if i == 0 else "p1"]
            label = f"{names[i][:18]}  HP {'#' * max(t.hp, 0)}{'.' * (game.cfg.max_hp - max(t.hp, 0))}"
            txt = self.font.render(label, True, col)
            surf.blit(txt, (8, 4) if i == 0 else (w - txt.get_width() - 8, 4))
        status = f"step {game.step_count}/{game.cfg.max_steps}  {game.map.name}"
        if game.done:
            status += "  |  " + ("DRAW" if game.winner is None else f"WINNER: {names[game.winner][:18]}")
            status += f" ({game.end_reason})"
        if extra:
            status += "  " + extra
        txt = self.font.render(status, True, COLORS["text"])
        surf.blit(txt, (8, 24))
        return surf

    def show(self, surface, fps: float = 8) -> bool:
        """Show a surface in a window. Returns False if the window was closed."""
        pg = self.pg
        if self.window is None:
            self.window = pg.display.set_mode(surface.get_size())
            pg.display.set_caption(self.title)
            self.clock = pg.time.Clock()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.closed = True
        if self.closed:
            return False
        self.window.blit(surface, (0, 0))
        pg.display.flip()
        if fps:
            self.clock.tick(fps)
        return True

    def to_array(self, surface) -> np.ndarray:
        return np.transpose(self.pg.surfarray.array3d(surface), (1, 0, 2)).copy()

    def close(self) -> None:
        if self.window is not None:
            self.pg.display.quit()
            self.window = None


def save_gif(frames: List[np.ndarray], path: str, fps: float = 8) -> None:
    from PIL import Image
    if not frames:
        raise ValueError("No frames to save")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    print(f"GIF saved to {path} ({len(frames)} frames)")
