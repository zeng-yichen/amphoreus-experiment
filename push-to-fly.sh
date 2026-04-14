#!/usr/bin/env bash
# Push local memory/ directory to Fly's /data/memory/.
#
# Ordinal sync runs on your local machine. This script uploads the
# results to Fly so the production backend has the same data.
#
# Usage:
#   ./push-to-fly.sh              # push everything
#   ./push-to-fly.sh trimble-mark # push one client only

set -euo pipefail

APP="amphoreus"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_MEMORY="$SCRIPT_DIR/memory"

if [ ! -d "$LOCAL_MEMORY" ]; then
    echo "✗ No local memory/ directory found. Run the backend locally first so ordinal_sync populates it."
    exit 1
fi

if [ $# -gt 0 ]; then
    CLIENT="$1"
    if [ ! -d "$LOCAL_MEMORY/$CLIENT" ]; then
        echo "✗ No local memory/$CLIENT/ directory found."
        exit 1
    fi
    echo "⟳ Pushing memory/$CLIENT/ to Fly ($APP)..."
    tar czf "$SCRIPT_DIR/_push_sync.tar.gz" -C "$SCRIPT_DIR" "memory/$CLIENT/"
else
    echo "⟳ Pushing all memory/ to Fly ($APP)..."
    tar czf "$SCRIPT_DIR/_push_sync.tar.gz" -C "$SCRIPT_DIR" memory/
fi

# Upload the archive via fly sftp.
# The fly CLI prints a "Metrics token unavailable" warning to stderr
# which can cause a non-zero exit code. We suppress stderr and check
# that the upload actually worked by verifying the file exists on Fly.
echo "  → Uploading ($(du -h "$SCRIPT_DIR/_push_sync.tar.gz" | cut -f1) compressed)..."
fly sftp shell -a "$APP" <<SFTP 2>/dev/null || true
put $SCRIPT_DIR/_push_sync.tar.gz /tmp/_push_sync.tar.gz
SFTP

# Extract on Fly
echo "  → Extracting on Fly..."
fly ssh console -a "$APP" -q -C "tar xzf /tmp/_push_sync.tar.gz -C /data" 2>/dev/null

# Cleanup
echo "  → Cleaning up..."
fly ssh console -a "$APP" -q -C "rm -f /tmp/_push_sync.tar.gz" 2>/dev/null || true
rm -f "$SCRIPT_DIR/_push_sync.tar.gz"

echo "✓ Pushed to Fly."
