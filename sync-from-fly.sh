#!/usr/bin/env bash
# Pull memory/ from Fly to local (one-time bootstrap).
#
# Use this to seed your local memory/ directory from Fly's existing
# data. After that, ordinal_sync runs locally and you push results
# back to Fly with push-to-fly.sh.
#
# Usage:
#   ./sync-from-fly.sh              # pull everything
#   ./sync-from-fly.sh trimble-mark # pull one client only

set -euo pipefail

APP="amphoreus"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_MEMORY="$SCRIPT_DIR/memory"
REMOTE_ARCHIVE="/tmp/_memory_sync.tar.gz"

mkdir -p "$LOCAL_MEMORY"

if [ $# -gt 0 ]; then
    CLIENT="$1"
    TAR_PATH="memory/$CLIENT/"
    echo "⟳ Pulling $CLIENT from Fly ($APP)..."
else
    TAR_PATH="memory/"
    echo "⟳ Pulling all memory from Fly ($APP)..."
fi

# Create archive on Fly (single command, no && chaining)
echo "  → Creating archive on Fly..."
fly ssh console -a "$APP" -q -C "tar czf $REMOTE_ARCHIVE -C /data $TAR_PATH" 2>/dev/null

# Download
echo "  → Downloading..."
fly sftp get -a "$APP" "$REMOTE_ARCHIVE" "$SCRIPT_DIR/_memory_sync.tar.gz" 2>/dev/null

# Extract locally
echo "  → Extracting..."
tar xzf "$SCRIPT_DIR/_memory_sync.tar.gz" -C "$SCRIPT_DIR"

# Cleanup
rm -f "$SCRIPT_DIR/_memory_sync.tar.gz"
fly ssh console -a "$APP" -q -C "rm -f $REMOTE_ARCHIVE" 2>/dev/null || true

echo "✓ Synced to $LOCAL_MEMORY/"
echo ""
echo "Local memory contents:"
ls -1 "$LOCAL_MEMORY/" | head -30
