#!/usr/bin/env bash
# Clone Kev (once) and serve a checkpoint locally with the Jev-compatible API.
# Usage: scripts/start_kev.sh [checkpoint] [port]
set -euo pipefail
CHECKPOINT="${1:-jaredpalmer/kev-0.8b}"
PORT="${2:-8009}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -d "$ROOT/.kev" ]; then
  git clone --depth 1 https://github.com/jaredpalmer/kev.git "$ROOT/.kev"
fi
cd "$ROOT/.kev"
uv sync --extra serve
exec uv run --extra serve python -m kev.serve --run "$CHECKPOINT" --port "$PORT"
