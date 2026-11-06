"""
Model exchange between computers through GitHub (models/league/ of the repo).

    python student/sync_models.py --once                 # download only
    python student/sync_models.py --once --publish A     # download + publish models/outbox/A-*.zip
    nohup python student/sync_models.py --loop --publish A > logs/sync.out 2>&1 &

All git commands run in a separate clean clone, ../Challenge2Tanks-sync (cloned from our origin if
missing); never in our working folder.
 - download: git pull --rebase in the clone, new/changed models/league/*.zip -> our models/shared/
 - publish:  models/outbox/<P>-*.zip (P in --publish) -> clone models/league/, git add of that file
             only, commit, push (on failure: pull --rebase and retry, 3 times) -> models/outbox/sent/
Never publishes R1-*, anything from models/reserved/, or files over 50 MB.
Log: logs/sync.log. A lock file keeps a --loop and a --once (from pipeline.py) from overlapping.
"""
import argparse
import configparser
import datetime
import fcntl
import filecmp
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNC = os.path.join(os.path.dirname(ROOT), "Challenge2Tanks-sync")
SHARED = os.path.join(ROOT, "models", "shared")
OUTBOX = os.path.join(ROOT, "models", "outbox")
RESERVED = os.path.join(ROOT, "models", "reserved")
MAX_BYTES = 50 * 1024 * 1024


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "sync.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def git(*args, check=True):
    r = subprocess.run(["git", "-C", SYNC, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return r


def origin_url():
    """Read our origin URL from .git/config (no git command on the working folder)."""
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, ".git", "config"))
    return cfg['remote "origin"']["url"]


def ensure_clone(url):
    if os.path.isdir(os.path.join(SYNC, ".git")):
        return
    log(f"cloning {url} -> {SYNC}")
    subprocess.run(["git", "clone", "-q", url, SYNC], check=True)


def download():
    git("pull", "--rebase", "-q")
    os.makedirs(SHARED, exist_ok=True)
    new = []
    for src in sorted(glob.glob(os.path.join(SYNC, "models", "league", "*.zip"))):
        dst = os.path.join(SHARED, os.path.basename(src))
        if not os.path.exists(dst) or not filecmp.cmp(src, dst, shallow=False):
            shutil.copy2(src, dst)
            new.append(os.path.basename(src))
    log(f"download: {len(new)} new/changed -> models/shared/" + (f": {', '.join(new)}" if new else ""))
    return new


def publishable(path, prefixes):
    name = os.path.basename(path)
    real = os.path.realpath(path)
    if name.startswith("R1"):
        return "R1 is never published"
    if real.startswith(os.path.realpath(RESERVED) + os.sep):
        return "file from models/reserved/"
    if os.path.getsize(real) > MAX_BYTES:
        return f"larger than {MAX_BYTES // 2**20} MB"
    if not any(name.startswith(p + "-") for p in prefixes):
        return "prefix not in --publish"
    return None


def push_with_retry(tries=3):
    for k in range(1, tries + 1):
        if git("push", "-q", check=False).returncode == 0:
            return True
        log(f"  push failed (attempt {k}/{tries}): pull --rebase and retry")
        git("pull", "--rebase", "-q", check=False)
    return False


def publish(prefixes):
    os.makedirs(os.path.join(OUTBOX, "sent"), exist_ok=True)
    for path in sorted(glob.glob(os.path.join(OUTBOX, "*.zip"))):
        name = os.path.basename(path)
        why = publishable(path, prefixes)
        if why:
            log(f"publish: skip {name} ({why})")
            continue
        rel = f"models/league/{name}"
        os.makedirs(os.path.join(SYNC, "models", "league"), exist_ok=True)
        shutil.copy2(path, os.path.join(SYNC, rel))
        git("add", "--", rel)
        if git("diff", "--cached", "--quiet", "--", rel, check=False).returncode == 0:
            log(f"publish: {name} already identical on origin")
        else:
            git("commit", "-q", "-m", f"sync: {name} (PC1, student/sync_models.py)", "--", rel)
            if not push_with_retry():
                log(f"publish: ERROR, could not push {name}; it stays in models/outbox/ (the commit is kept "
                    f"in the sync clone and goes out with the next push)")
                continue
            log(f"publish: pushed {rel}")
        shutil.move(path, os.path.join(OUTBOX, "sent", name))


def run_once(args):
    lock = open(os.path.join(os.path.dirname(SYNC), ".Challenge2Tanks-sync.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)          # waits for a concurrent run
    try:
        ensure_clone(args.url or origin_url())
        download()
        if args.publish:
            publish(args.publish)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--loop", action="store_true")
    ap.add_argument("--publish", nargs="*", default=[], help="branch prefixes to publish from models/outbox/ (e.g. A)")
    ap.add_argument("--every", type=int, default=15 * 60, help="seconds between --loop runs")
    ap.add_argument("--url", default=None, help="repo URL for the first clone (default: our origin)")
    args = ap.parse_args(argv)
    if "R1" in args.publish:
        sys.exit("R1 is never published")
    while True:
        try:
            run_once(args)
        except Exception as e:   # keep looping: a network error must not kill --loop
            log(f"ERROR: {type(e).__name__}: {e}")
            if args.once:
                sys.exit(1)
        if args.once:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
