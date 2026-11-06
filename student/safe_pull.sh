#!/usr/bin/env bash
# git pull without fighting over shared files (models published by other computers, BITACORA.md).
#
#   bash student/safe_pull.sh
#
# 1. git fetch.
# 2. Each file the pull would add that already exists here untracked: identical to origin's blob
#    (git hash-object) -> local copy removed; different -> renamed to <name>.local (warning).
# 3. git pull --rebase --autostash.
# 4. If the only conflict is BITACORA.md: keep the lines of both sides (drop the <<<<<<< ======= >>>>>>>
#    markers), git add it, then continue the rebase or drop the autostash entry. A conflict in any other
#    file: stop and report, touching nothing.
# Never: git add ., plain git stash, git reset --hard.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1

say() { echo "[safe_pull] $*"; }
in_rebase() { [ -d "$(git rev-parse --git-path rebase-merge)" ] || [ -d "$(git rev-parse --git-path rebase-apply)" ]; }
unmerged() { git diff --name-only --diff-filter=U; }

resolve_bitacora() {   # keep both sides; drop the base section of diff3-style conflicts
    awk '/^<<<<<<< /{base=0; next} /^\|\|\|\|\|\|\| /{base=1; next} /^=======$/{base=0; next}
         /^>>>>>>> /{next} !base' BITACORA.md > BITACORA.md.tmp && mv BITACORA.md.tmp BITACORA.md
    git add -- BITACORA.md
}

# 1-2 --------------------------------------------------------------------------------------------
git fetch -q || { say "git fetch failed"; exit 1; }
upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}') || { say "no upstream branch"; exit 1; }
warned=0
while IFS= read -r f; do
    [ -e "$f" ] || continue
    git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 && continue      # tracked: git handles it
    if [ "$(git hash-object -- "$f")" = "$(git rev-parse "$upstream:$f")" ]; then
        rm -- "$f" && say "removed local untracked copy (identical to $upstream): $f"
    else
        mv -- "$f" "$f.local" && say "WARNING: $f differs from $upstream -> kept as $f.local"
        warned=1
    fi
done < <(git diff --name-only --diff-filter=A HEAD "$upstream")

# 3-4 --------------------------------------------------------------------------------------------
git pull --rebase --autostash -q
for _ in 1 2 3 4 5; do
    conflicts=$(unmerged)
    [ -z "$conflicts" ] && break
    if [ "$conflicts" != "BITACORA.md" ]; then
        say "CONFLICT in files other than BITACORA.md, nothing touched:"
        echo "$conflicts" | sed 's/^/    /'
        in_rebase && say "a rebase is in progress: resolve it, or 'git rebase --abort' (autostash is restored)"
        exit 2
    fi
    resolve_bitacora
    if in_rebase; then
        say "BITACORA.md conflict in the rebase: kept both sides, continuing"
        GIT_EDITOR=true git rebase --continue >/dev/null 2>&1   # may stop again (e.g. the autostash)
    else
        say "BITACORA.md conflict when re-applying the autostash: kept both sides"
        if git stash list -n 1 | grep -q autostash; then
            git stash drop -q && say "dropped the autostash entry (its changes are in the working tree)"
        fi
    fi
done
if [ -n "$(unmerged)" ] || in_rebase; then
    say "could not finish the pull cleanly, see git status"
    exit 2
fi
say "OK: $(git log --oneline -1)"
[ "$warned" = 1 ] && say "check the *.local files listed above"
exit 0
