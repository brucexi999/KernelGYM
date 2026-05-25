#!/usr/bin/env bash
# Run a real-GPU smoke that exercises the all-turn NCU gating during a
# real training step.
#
# Invokes the workspace launcher with overrides that:
#   - skip val_before_train (we have a separate proof of gating in val)
#   - run exactly one training step over a 4-problem fixture
#     (train_batch_size=4 must divide evenly across n_gpus_per_node=4)
# Captures launcher stdout, then runs verify_trajectory.py over it.
#
# Required env var:
#   DRKERNEL_WORKSPACE  — absolute path to the parent workspace where
#                         baseline/launch_drkernel_8b_rl_baseline_ncu.sh
#                         lives.
#
# Optional env vars:
#   SMOKE_MAX_TURN      — default 3
#
# Exit code: 0 = gating pass, non-zero = verifier or fixture failure.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_DIR="${REPO_DIR}/tests/gpu"

: "${DRKERNEL_WORKSPACE:?Set DRKERNEL_WORKSPACE to the workspace root (the dir containing baseline/ and logs/).}"
: "${SMOKE_MAX_TURN:=3}"

LAUNCHER="${DRKERNEL_WORKSPACE}/baseline/launch_drkernel_8b_rl_baseline_ncu.sh"
SMOKE_DATASET="${SMOKE_DIR}/fixtures/smoke_4problems.parquet"

if [[ ! -x "${LAUNCHER}" ]]; then
  echo "ERROR: launcher not found or not executable: ${LAUNCHER}" >&2
  exit 2
fi
if [[ ! -f "${SMOKE_DATASET}" ]]; then
  echo "ERROR: smoke fixture missing: ${SMOKE_DATASET}" >&2
  exit 2
fi

# Capture launcher stdout/stderr into a file the verifier can scan.
# In foreground mode the launcher does not tee to the nominal
# logs/drkernel_baseline_ncu_rl/drkernel_8b_rl_*.log file, so we own
# the capture here.
CAPTURE_LOG="/tmp/ncu_all_turn_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[smoke] max_turn=${SMOKE_MAX_TURN}"
echo "[smoke] dataset=${SMOKE_DATASET}"
echo "[smoke] launcher=${LAUNCHER}"
echo "[smoke] capture=${CAPTURE_LOG}"
echo "[smoke] running launcher (1 training step on 4 problems, no val)..."

LAUNCHER_RC=0
VAL_MAX_TURN="${SMOKE_MAX_TURN}" TRAIN_FOREGROUND=1 "${LAUNCHER}" \
  --max_turn "${SMOKE_MAX_TURN}" \
  --val_before_train False \
  --train_batch_size 4 \
  --total_epochs 1 \
  --train_dataset "${SMOKE_DATASET}" \
  > "${CAPTURE_LOG}" 2>&1 || LAUNCHER_RC=$?

echo "[smoke] launcher exited with rc=${LAUNCHER_RC} (any non-zero exit after training rollouts complete is tolerable; the verifier reads the captured trajectory)."
echo "[smoke] running verifier on ${CAPTURE_LOG}..."

exec python3 "${SMOKE_DIR}/verify_trajectory.py" "${CAPTURE_LOG}" --max-turns "${SMOKE_MAX_TURN}"
