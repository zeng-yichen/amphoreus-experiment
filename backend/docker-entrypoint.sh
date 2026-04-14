#!/bin/sh
#
# Amphoreus backend entrypoint.
#
# Purpose: bridge the Fly persistent volume (mounted at /data) into the paths
# the application expects on disk. The existing code uses Path(__file__) to
# resolve PROJECT_ROOT, which on Fly resolves to /app — so we symlink the
# persistent subdirs in place rather than refactoring every `memory/...`
# reference across 30+ Python files.
#
# SQLite: uses the SQLITE_PATH env var (see fly.toml).
# Everything else: symlinked from /data to the expected /app/* paths.
#
# Safe to run multiple times (ln -sfn overwrites).
#
set -eu

VOLUME_ROOT="${VOLUME_ROOT:-/data}"

# Create persistent subdirs on the volume (idempotent).
mkdir -p \
    "${VOLUME_ROOT}/memory" \
    "${VOLUME_ROOT}/sqlite" \
    "${VOLUME_ROOT}/backend_data" \
    "${VOLUME_ROOT}/images" \
    "${VOLUME_ROOT}/products" \
    "${VOLUME_ROOT}/output"

# Symlink the in-image paths to the volume.
# If the in-image path already exists as a non-symlink (i.e. from the build),
# remove it first so ln -sfn can take over.
link() {
    target="$1"; src="$2"
    if [ -e "$src" ] && [ ! -L "$src" ]; then
        rm -rf "$src"
    fi
    # ensure parent dir exists for nested targets (e.g. /app/backend/static/images)
    mkdir -p "$(dirname "$src")"
    ln -sfn "$target" "$src"
}

link "${VOLUME_ROOT}/memory"       /app/memory
link "${VOLUME_ROOT}/backend_data" /app/backend/data
link "${VOLUME_ROOT}/images"       /app/backend/static/images
link "${VOLUME_ROOT}/products"     /app/products
link "${VOLUME_ROOT}/output"       /app/output

# SQLite path + DATA_DIR: default to the volume so that forgetting to set
# these in fly.toml or docker run can't silently fall back to ephemeral
# in-container storage. Explicit env values from the deploy config still win.
export DATA_DIR="${DATA_DIR:-${VOLUME_ROOT}/sqlite}"
export SQLITE_PATH="${SQLITE_PATH:-${DATA_DIR}/amphoreus.db}"
mkdir -p "$(dirname "${SQLITE_PATH}")"

# Stage 2 auth: ensure /data/acl.json and /data/audit.log exist on the volume.
# acl.json is seeded with a commented-out example on first boot so a fresh
# deploy doesn't immediately 403 everyone. Admins should edit this file via
# `flyctl ssh console -a amphoreus` and re-run the backend (or just wait —
# the ACL reader mtime-caches and picks up changes live).
export ACL_PATH="${ACL_PATH:-${VOLUME_ROOT}/acl.json}"
export AUDIT_LOG_PATH="${AUDIT_LOG_PATH:-${VOLUME_ROOT}/audit.log}"
if [ ! -f "${ACL_PATH}" ]; then
    echo "[entrypoint] seeding default ACL at ${ACL_PATH}"
    cat > "${ACL_PATH}" <<'JSON'
{
  "admins": [
    "yzeng@berkeley.edu"
  ],
  "users": {}
}
JSON
fi
touch "${AUDIT_LOG_PATH}"

echo "[entrypoint] volume=${VOLUME_ROOT}"
echo "[entrypoint] SQLITE_PATH=${SQLITE_PATH:-<unset>}"
echo "[entrypoint] DATA_DIR=${DATA_DIR}"
echo "[entrypoint] ACL_PATH=${ACL_PATH}"
echo "[entrypoint] AUDIT_LOG_PATH=${AUDIT_LOG_PATH}"
echo "[entrypoint] exec: $*"

exec "$@"
