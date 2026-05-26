#!/usr/bin/env bash
# start.sh — launch the TTS server with correct PYTHONPATH for qwen_megakernel.
#
# Usage (from repo root):
#   bash start.sh
#   PORT=9090 bash start.sh
#
# qwen_megakernel must be cloned as a sibling directory:
#   git clone https://github.com/AlpinDale/qwen_megakernel.git ../qwen_megakernel
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGAKERNEL_DIR="$(realpath "$REPO_ROOT/../qwen_megakernel")"

if [[ ! -d "$MEGAKERNEL_DIR" ]]; then
    echo "ERROR: qwen_megakernel not found at $MEGAKERNEL_DIR" >&2
    echo "  git clone https://github.com/AlpinDale/qwen_megakernel.git $MEGAKERNEL_DIR" >&2
    exit 1
fi

export PYTHONPATH="$MEGAKERNEL_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH=$PYTHONPATH"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

# Kill any existing listener on the port (ignore errors if nothing is running)
fuser -k "${PORT}/tcp" 2>/dev/null || true

exec /venv/main/bin/uvicorn server.app:app \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
