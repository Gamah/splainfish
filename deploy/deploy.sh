#!/usr/bin/env bash
#
# Deploy the splainfish web app to the notadomain.lol nginx host.
#
# Copies web/ (only the browser app — not the Python CLI, tests, or vendor dir)
# to /var/www/notadomain.lol/chess/ over SSH with rsync.
#
# Usage:
#   SSH_TARGET=you@notadomain.lol ./deploy/deploy.sh
#
# Override the remote path if yours differs:
#   SSH_TARGET=you@host REMOTE_DIR=/srv/www/site/chess ./deploy/deploy.sh
#
# Requires: rsync and ssh, with key-based access to the host.

set -euo pipefail

SSH_TARGET="${SSH_TARGET:?set SSH_TARGET, e.g. you@notadomain.lol}"
REMOTE_DIR="${REMOTE_DIR:-/var/www/notadomain.lol/chess}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/web/"

if [ ! -f "$SRC/index.html" ]; then
  echo "error: $SRC/index.html not found — run from a full checkout" >&2
  exit 1
fi

echo "→ Ensuring $REMOTE_DIR exists on $SSH_TARGET"
ssh "$SSH_TARGET" "mkdir -p '$REMOTE_DIR'"

echo "→ Syncing web/ → $SSH_TARGET:$REMOTE_DIR"
# --delete keeps the remote a clean mirror of web/. The .nnue is large but only
# transfers when changed (rsync checksums), so redeploys after code edits are
# fast.
rsync -avz --delete \
  --exclude '.DS_Store' \
  "$SRC" "$SSH_TARGET:$REMOTE_DIR/"

echo "✓ Deployed. Ensure deploy/nginx-chess.conf is included in the"
echo "  notadomain.lol server block, then: sudo nginx -t && sudo systemctl reload nginx"
echo "  Live at: https://notadomain.lol/chess/"
