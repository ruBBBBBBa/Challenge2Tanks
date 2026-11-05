# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Win rate of an agent against the official bots (and the baselines, if available).

    python tools/evaluate.py --agent models/tanks_ppo.zip
    python tools/evaluate.py --agent models/tanks_ppo.zip --games 40 --opponents basic baseline:medium
    python tools/evaluate.py --agent "param:aggression=0.9,caution=0.3"

Sides alternate every game and the maps cycle through the official maps + random ones.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # challenge root

from tanks.evaluation import evaluate_agent  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="agent spec, e.g. models/tanks_ppo.zip")
    ap.add_argument("--opponents", nargs="*", default=None)
    ap.add_argument("--games", type=int, default=20, help="games per opponent")
    ap.add_argument("--maps", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"Evaluating {args.agent} ({args.games} games per opponent)\n")
    res = evaluate_agent(args.agent, args.opponents, args.games, args.maps, args.seed)
    total = sum(r["win"] for r in res.values()) / max(1, sum(args.games for _ in res))
    print(f"\nOverall win rate: {100 * total:.1f}%")


if __name__ == "__main__":
    main()
