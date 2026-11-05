# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Watch a game between two agents.

Examples:
    python tools/play_match.py --p0 basic --p1 aggressive
    python tools/play_match.py --p0 models/tanks_ppo.zip --p1 baseline:easy --map arena_03_river
    python tools/play_match.py --p0 defensive --p1 "param:aggression=0.9,caution=0.1" --gif out/game.gif --no-render
    python tools/play_match.py --p0 models/tanks_ppo.zip --p1 basic --games 5 --save-replay replays/

Agent specs: random, basic, aggressive, defensive, param:..., baseline:<level>,
             model:<path.zip> (or just <path.zip>), my:<bot>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # challenge root

from tanks.match import play_match, save_replay  # noqa: E402
from tanks.maps import list_maps  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", default="basic", help="agent spec for player 0 (blue)")
    ap.add_argument("--p1", default="aggressive", help="agent spec for player 1 (red)")
    ap.add_argument("--map", default="random", help=f"map name, path or 'random'. Official: {list_maps()}")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--no-render", action="store_true", help="do not open a window")
    ap.add_argument("--gif", default=None, help="save a GIF (with --games > 1 a number is appended)")
    ap.add_argument("--save-replay", default=None, help="directory or .json file for replays")
    args = ap.parse_args()

    wins = [0, 0, 0]
    for g in range(args.games):
        seed = None if args.seed is None else args.seed + g
        res = play_match(args.p0, args.p1, args.map, seed=seed, render=not args.no_render,
                         fps=args.fps, record_frames=args.gif is not None)
        w = res["winner"]
        wins[2 if w is None else w] += 1
        who = "draw" if w is None else res["names"][w]
        print(f"game {g + 1}: map={res['map_name']:22s} winner={who:25s} reason={res['end_reason']:9s} "
              f"steps={res['steps']:3d} hp={res['hp']} damage={res['damage']}")
        if args.gif:
            from tanks.render import save_gif
            path = args.gif if args.games == 1 else args.gif.replace(".gif", f"_{g + 1}.gif")
            save_gif(res["frames"], path, args.fps)
        if args.save_replay:
            path = args.save_replay
            if not path.endswith(".json"):
                path = os.path.join(path, f"game_{g + 1}.json")
            save_replay(res, path)
            print(f"replay saved to {path}")
    if args.games > 1:
        print(f"\nTotal: P0 {wins[0]} - P1 {wins[1]} - draws {wins[2]}")


if __name__ == "__main__":
    main()
