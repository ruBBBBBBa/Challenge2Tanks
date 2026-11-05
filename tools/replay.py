# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Watch a saved replay or export it as a GIF.

    python tools/replay.py replays/game_1.json
    python tools/replay.py replays/game_1.json --gif out/game_1.gif --no-render
"""
import argparse

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # challenge root

from tanks.match import load_replay, replay_game  # noqa: E402
from tanks.render import Renderer, save_gif  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    data = load_replay(args.file)
    names = data.get("names", ["P0", "P1"])
    renderer = Renderer(title=f"Replay - {args.file}")
    frames = []
    game = None
    for game in replay_game(data):
        surf = renderer.draw(game, names)
        if args.gif:
            frames.append(renderer.to_array(surf))
        if not args.no_render and not renderer.show(surf, args.fps):
            break
    if game is not None:
        who = "draw" if game.winner is None else names[game.winner]
        print(f"{names[0]} vs {names[1]} on {game.map.name}: winner={who} ({game.end_reason}), "
              f"{game.step_count} steps")
    if args.gif:
        save_gif(frames, args.gif, args.fps)
    if not args.no_render:
        for _ in range(int(args.fps * 2)):
            if not renderer.show(renderer.draw(game, names), args.fps):
                break
        renderer.close()


if __name__ == "__main__":
    main()
