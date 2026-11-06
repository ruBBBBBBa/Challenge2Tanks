"""
TensorBoard curves -> CSV + comparison PNGs for the report.

    python student/export_tb.py                       # results/tb/<ID>/ -> results/curves/
    python student/export_tb.py --runs A-g1 B-g1 --smooth 0.95

One CSV per training (results/curves/<ID>.csv, one column per tag), one PNG per metric comparing
all runs (EMA-smoothed line over the faint raw curve) and, per run with PFSP, a PNG of pfsp/wr/*.
"""
import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ["rollout/success_rate", "rollout/ep_rew_mean", "rollout/ep_len_mean", "train/entropy_loss"]


def read_run(path):
    """{tag: [(step, value), ...]} for METRICS and pfsp/wr/*, merging every event file under path."""
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "**", "events.out.tfevents.*"), recursive=True)):
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if tag in METRICS or tag.startswith("pfsp/wr/"):
                out.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    return {t: sorted(dict(v).items()) for t, v in out.items()}


def ema(values, alpha):
    out, m = [], None
    for v in values:
        m = v if m is None else alpha * m + (1 - alpha) * v
        out.append(m)
    return out


def write_csv(path, run):
    tags = sorted(run)
    steps = sorted({s for t in tags for s, _ in run[t]})
    cols = {t: dict(run[t]) for t in tags}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step"] + tags)
        for s in steps:
            w.writerow([s] + [cols[t].get(s, "") for t in tags])


def plot(ax, pts, alpha, label, color=None):
    xs, ys = zip(*pts)
    line, = ax.plot(xs, ema(ys, alpha), label=label, color=color, lw=1.6)
    ax.plot(xs, ys, color=line.get_color(), alpha=0.2, lw=0.8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tb-dir", default="results/tb")
    ap.add_argument("--out", default="results/curves")
    ap.add_argument("--runs", nargs="*", default=None, help="run IDs (default: every folder in --tb-dir)")
    ap.add_argument("--smooth", type=float, default=0.9, help="EMA factor (0 = raw)")
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    ids = args.runs or sorted(d for d in os.listdir(args.tb_dir) if os.path.isdir(os.path.join(args.tb_dir, d)))
    runs = {i: read_run(os.path.join(args.tb_dir, i)) for i in ids}
    runs = {i: r for i, r in runs.items() if r}
    if not runs:
        sys.exit(f"No TensorBoard data in {args.tb_dir}")
    os.makedirs(args.out, exist_ok=True)

    for i, r in runs.items():
        write_csv(os.path.join(args.out, f"{i}.csv"), r)
        wr = {t: v for t, v in r.items() if t.startswith("pfsp/wr/")}
        if wr:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for t, v in sorted(wr.items()):
                plot(ax, v, args.smooth, t[len("pfsp/wr/"):])
            ax.set(title=f"{i}: win rate por rival (PFSP)", xlabel="pasos", ylabel="win rate", ylim=(0, 1))
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"{i}_pfsp_wr.png"), dpi=130)
            plt.close(fig)

    for tag in METRICS:
        have = {i: r[tag] for i, r in runs.items() if r.get(tag)}
        if not have:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, v in have.items():
            plot(ax, v, args.smooth, i)
        ax.set(title=tag, xlabel="pasos")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, tag.replace("/", "_") + ".png"), dpi=130)
        plt.close(fig)
    print(f"{len(runs)} runs -> {args.out}: " + ", ".join(f"{i} ({len(r)} tags)" for i, r in runs.items()))


if __name__ == "__main__":
    main()
