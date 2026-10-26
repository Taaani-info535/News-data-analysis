#!/usr/bin/env bash
# Restore persisted state (SQLite DB, models, reports) from the orphan branch `uamis-state` into ./state.
# Uses the credentials actions/checkout already configured; nothing secret is printed.
set -euo pipefail
BRANCH="${STATE_BRANCH:-uamis-state}"
mkdir -p state
if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git fetch --depth=1 origin "$BRANCH:refs/remotes/origin/$BRANCH"
  git archive "origin/$BRANCH" | tar -x -C state
  echo "state restored from branch $BRANCH"
else
  echo "no state branch yet - starting fresh"
fi
# fingerprint so we only push when something changed
find state -type f -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1 > /tmp/state_fingerprint_before || true
