#!/usr/bin/env bash
# Run post-training with configs/final_posttrain_from_tdeed.example.json under PM2.
#
# --- PM2 (survives SSH disconnect; centralized logs) ---
#
# From the repo root (custom-spotting/):
#
#   chmod +x scripts/run-posttrain-pm2.sh
#   pm2 start scripts/run-posttrain-pm2.sh --name actionspot-posttrain --no-autorestart
#
# Logs:
#   pm2 logs actionspot-posttrain
#
# Training exits once; PM2 leaves the job stopped (--no-autorestart). Remove:
#   pm2 delete actionspot-posttrain
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${REPO_ROOT}/configs/final_posttrain_from_tdeed.example.json"
export PYTHONUNBUFFERED=1

if [[ -x "${REPO_ROOT}/.venv/bin/custom-spotting" ]]; then
  exec "${REPO_ROOT}/.venv/bin/custom-spotting" posttrain --config "${CONFIG}"
elif command -v custom-spotting >/dev/null 2>&1; then
  exec custom-spotting posttrain --config "${CONFIG}"
else
  echo "custom-spotting not found. Activate .venv or: pip install -e ." >&2
  exit 1
fi
