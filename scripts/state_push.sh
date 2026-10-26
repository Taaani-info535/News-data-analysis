#!/usr/bin/env bash
# Publish ./state to the orphan branch as a SINGLE commit (force-push) so repository history never grows.
# Workflows share one concurrency group, so pushes are serialised.
set -euo pipefail
BRANCH="${STATE_BRANCH:-uamis-state}"
[ -d state ] || { echo "no state dir"; exit 0; }
after=$(find state -type f -print0 | sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1)
before=$(cat /tmp/state_fingerprint_before 2>/dev/null || echo none)
if [ "$after" = "$before" ]; then echo "state unchanged - nothing to push"; exit 0; fi
URL=$(git remote get-url origin)
AUTH=$(git config --get http.https://github.com/.extraheader || true)
rm -rf /tmp/statepush && mkdir /tmp/statepush && cp -r state/. /tmp/statepush/
cd /tmp/statepush
git init -q && git checkout -q -b "$BRANCH"
git config user.name "uamis-bot" && git config user.email "uamis-bot@users.noreply.github.com"
git add -A && git commit -q -m "state $(date -u +%Y-%m-%dT%H:%MZ)"
git remote add origin "$URL"
if [ -n "$AUTH" ]; then git -c http.extraheader="$AUTH" push -q --force origin "$BRANCH"; else git push -q --force origin "$BRANCH"; fi
echo "state pushed to $BRANCH"
