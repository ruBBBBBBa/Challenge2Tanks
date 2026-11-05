# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Play against any agent with the keyboard (you are the blue tank, player 0).

    python tools/play_human.py --opponent basic
    python tools/play_human.py --opponent models/tanks_ppo.zip --map arena_04_maze --fps 5

Controls:  W / Up = forward    S / Down = backward
           A / Left = turn left    D / Right = turn right
           Space = fire    Esc = quit
The game advances at --fps steps per second; the last key pressed in each tick is used.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # challenge root

from tanks.agents.base import load_agent  # noqa: E402
from tanks.game import BACKWARD, FIRE, FORWARD, NOOP, TURN_LEFT, TURN_RIGHT, TanksGame  # noqa: E402
from tanks.maps import list_maps, load_map  # noqa: E402
from tanks.render import Renderer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opponent", default="basic")
    ap.add_argument("--map", default="random", help=f"official: {list_maps()}")
    ap.add_argument("--fps", type=float, default=6)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    opponent = load_agent(args.opponent)
    renderer = Renderer(title="Tanks - you are BLUE")
    pg = renderer.pg
    keymap = {
        pg.K_w: FORWARD, pg.K_UP: FORWARD, pg.K_s: BACKWARD, pg.K_DOWN: BACKWARD,
        pg.K_a: TURN_LEFT, pg.K_LEFT: TURN_LEFT, pg.K_d: TURN_RIGHT, pg.K_RIGHT: TURN_RIGHT,
        pg.K_SPACE: FIRE,
    }
    score = [0, 0, 0]
    while True:
        game = TanksGame(load_map(args.map, rng))
        opponent.reset(int(rng.integers(2**31 - 1)))
        names = ["YOU", opponent.name]
        renderer.show(renderer.draw(game, names, "W/A/S/D + Space"), args.fps)
        while not game.done:
            action = NOOP
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    print(f"Final score  you {score[0]} - {score[1]} {opponent.name}  (draws {score[2]})")
                    return
                if event.type == pg.KEYDOWN and event.key in keymap:
                    action = keymap[event.key]
            if action == NOOP:   # holding a key also works
                pressed = pg.key.get_pressed()
                for key, a in keymap.items():
                    if pressed[key]:
                        action = a
                        break
            game.step([action, opponent.act(game.get_state(1))])
            renderer.show(renderer.draw(game, names), args.fps)
        score[2 if game.winner is None else game.winner] += 1
        print(f"{'Draw' if game.winner is None else ('You win!' if game.winner == 0 else 'You lose')} "
              f"| score you {score[0]} - {score[1]} {opponent.name} (draws {score[2]}) | press any key")
        renderer.show(renderer.draw(game, names, "press a key: next game"), args.fps)
        waiting = True
        while waiting:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    return
                if event.type == pg.KEYDOWN:
                    waiting = False
            pg.time.wait(30)


if __name__ == "__main__":
    main()
