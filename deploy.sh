#!/usr/bin/env bash
#
# One-command push-to-Fly for both the backend (amphoreus) and frontend
# (cyrene) apps. Sources whatever is in your local working tree — including
# uncommitted changes. No git gate.
#
# Usage:
#   ./deploy.sh                  # deploys both (backend first, then frontend)
#   ./deploy.sh backend
#   ./deploy.sh frontend
#   ./deploy.sh both             # same as no arg
#
# Notes:
# - Fly CLI must be installed and authenticated (`fly auth login`).
# - Backend deploy uses backend/fly.toml + backend/Dockerfile and ships the
#   repo root as build context (because the Dockerfile COPYs from ./backend).
# - Frontend deploy runs from the frontend/ directory so its Dockerfile sees
#   package.json at the build-context root.
# - --remote-only uses Fly's Depot builder (no local Docker daemon needed).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

target="${1:-both}"

deploy_backend() {
    echo "==> Deploying backend (amphoreus)"
    fly deploy \
        --config backend/fly.toml \
        --dockerfile backend/Dockerfile \
        --remote-only
}

deploy_frontend() {
    echo "==> Deploying frontend (cyrene)"
    (cd frontend && fly deploy --remote-only)
}

case "$target" in
    backend)  deploy_backend ;;
    frontend) deploy_frontend ;;
    both)     deploy_backend && deploy_frontend ;;
    *)
        echo "usage: $0 [backend|frontend|both]" >&2
        exit 1
        ;;
esac

echo "==> Done."
