#!/usr/bin/env bash
# Run a real-GPU smoke that exercises the all-turn NCU gating.
#
# Invokes the workspace launcher with overrides that produce a tiny
# 3-turn rollout on a single problem, then runs verify_trajectory.py
# against the resulting trainer log.
#
# Required env var:
#   DRKERNEL_WORKSPACE  — absolute path to the parent workspace where
#                         baseline/launch_drkernel_8b_rl_baseline_ncu.sh
#                         and logs/drkernel_baseline_ncu_rl/ live.
#
# Optional env vars:
#   SMOKE_MAX_TURN      — default 3
#   SMOKE_N_TRAIN       — default 1 (training steps after val_before_train)
#
# Exit code: 0 = gating pass, non-zero = verifier or launcher failure.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_DIR="${REPO_DIR}/tests/gpu"

: "${DRKERNEL_WORKSPACE:?Set DRKERNEL_WORKSPACE to the workspace root (the dir containing baseline/ and logs/).}"
: "${SMOKE_MAX_TURN:=3}"
: "${SMOKE_N_TRAIN:=1}"

LAUNCHER="${DRKERNEL_WORKSPACE}/baseline/launch_drkernel_8b_rl_baseline_ncu.sh"
TRAIN_LOG_DIR="${DRKERNEL_WORKSPACE}/logs/drkernel_baseline_ncu_rl"
SMOKE_DATASET="${SMOKE_DIR}/fixtures/smoke_1problem.parquet"

if [[ ! -x "${LAUNCHER}" ]]; then
  echo "ERROR: launcher not found or not executable: ${LAUNCHER}" >&2
  exit 2
fi
if [[ ! -f "${SMOKE_DATASET}" ]]; then
  echo "ERROR: smoke fixture missing: ${SMOKE_DATASET}" >&2
  exit 2
fi

echo "[smoke] max_turn=${SMOKE_MAX_TURN} n_train_steps=${SMOKE_N_TRAIN}"
echo "[smoke] dataset=${SMOKE_DATASET}"
echo "[smoke] launcher=${LAUNCHER}"
echo "[smoke] running launcher in foreground..."

# Snapshot the existing log timestamps so we can identify the new one.
mkdir -p "${TRAIN_LOG_DIR}"
PRE_LOGS_LIST="$(mktemp)"
ls -1 "${TRAIN_LOG_DIR}"/drkernel_8b_rl_*.log 2>/dev/null > "${PRE_LOGS_LIST}" || true

VAL_MAX_TURN="${SMOKE_MAX_TURN}" TRAIN_FOREGROUND=1 "${LAUNCHER}" \
  --max_turn "${SMOKE_MAX_TURN}" \
  --val_before_train True \
  --train_batch_size 1 \
  --n_val 1 \
  --total_epochs "${SMOKE_N_TRAIN}" \
  --train_dataset "${SMOKE_DATASET}"

echo "[smoke] launcher finished. Locating new training log..."

# Find the log file that did not exist before launch.
NEW_LOG=""
for f in "${TRAIN_LOG_DIR}"/drkernel_8b_rl_*.log; do
  if ! grep -qxF "${f}" "${PRE_LOGS_LIST}"; then
    NEW_LOG="${f}"
    break
  fi
done
rm -f "${PRE_LOGS_LIST}"

if [[ -z "${NEW_LOG}" ]]; then
  echo "ERROR: could not identify the smoke's training log under ${TRAIN_LOG_DIR}" >&2
  exit 3
fi

echo "[smoke] training log: ${NEW_LOG}"
echo "[smoke] running verifier..."

exec python3 "${SMOKE_DIR}/verify_trajectory.py" "${NEW_LOG}" --max-turns "${SMOKE_MAX_TURN}"
