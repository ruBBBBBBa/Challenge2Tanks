"""
Watchdog for a long training run (PC2, Windows).

    python student/watchdog.py --name D2-g1 --script logs/run_D2-g1.ps1

Every --every s it checks the PID in logs/<name>.pid. If the process is gone:
  - finished normally ("Saved models/<name>.zip" in logs/<name>.out) -> notify and exit;
  - logs/<name>.stop exists (stopped on purpose) -> exit without relaunching;
  - crashed -> notify (Windows popup + logs/watchdog_<name>.log, with the end of logs/<name>.err) and relaunch
    the script ONCE with RESUME_FROM = newest models/checkpoints_<name>/<name>_<N>_steps.zip (the script adds
    --resume, so the step count and snapshot names continue). A second crash is only reported.
To stop the training on purpose: create logs/<name>.stop first (or kill this watchdog first).
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def read(path, tail=None):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            s = f.read()
    except OSError:
        return ""
    return s[-tail:] if tail else s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--script", required=True, help="launch .ps1; must honour $env:RESUME_FROM and write the pid file")
    ap.add_argument("--every", type=int, default=120)
    ap.add_argument("--max-restarts", type=int, default=1)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    pidf, outf, errf = (f"logs/{args.name}.{e}" for e in ("pid", "out", "err"))
    stopf, logf = f"logs/{args.name}.stop", f"logs/watchdog_{args.name}.log"

    def log(msg, popup=False):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if popup:  # Windows message box on the console session; never blocks (msg returns at once)
            subprocess.run(["msg", os.environ.get("USERNAME", "*"), "/TIME:86400", f"{args.name}: {msg[:250]}"],
                           capture_output=True)

    restarts = 0
    log(f"watching {args.name} (pid file {pidf}, max {args.max_restarts} restart)")
    while True:
        time.sleep(args.every)
        try:
            pid = int(read(pidf).strip())
        except ValueError:
            continue
        if alive(pid):
            continue
        if f"Saved models/{args.name}.zip" in read(outf):
            log("training finished normally", popup=True)
            return
        if os.path.exists(stopf):
            log(f"pid {pid} gone and {stopf} exists: stopped on purpose, not relaunching")
            return
        err = read(errf, 1500).strip().splitlines()
        steps = re.findall(r"total_timesteps\s*\|\s*(\d+)", read(outf))
        log(f"CRASH: pid {pid} gone at {steps[-1] if steps else '?'} steps. Last error line: "
            f"{err[-1] if err else '(empty .err)'}", popup=True)
        log("--- end of .err ---\n" + "\n".join(err[-15:]))
        if restarts >= args.max_restarts:
            log(f"already relaunched {restarts} time(s): not relaunching again", popup=True)
            return
        ckpts = glob.glob(f"models/checkpoints_{args.name}/{args.name}_*_steps.zip")
        ckpts.sort(key=lambda p: int(re.search(r"_(\d+)_steps\.zip$", p).group(1)))
        env = dict(os.environ)
        if ckpts:
            env["RESUME_FROM"] = ckpts[-1].replace("\\", "/")
        else:
            env.pop("RESUME_FROM", None)
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", args.script],
                           env=env, capture_output=True, text=True)
        restarts += 1
        log(f"relaunched ({'from ' + ckpts[-1] if ckpts else 'from scratch, no checkpoint'}): {r.stdout.strip()} "
            f"{r.stderr.strip()[-300:]}", popup=True)


if __name__ == "__main__":
    main()
