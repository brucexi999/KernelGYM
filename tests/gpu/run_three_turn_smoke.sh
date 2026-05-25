#!/usr/bin/env bash
# Run a real-GPU smoke that exercises the all-turn NCU gating end-to-end.
#
# Invokes the workspace launcher with overrides that:
#   - val_before_train=True (the val phase produces the wandb val table
#     with cleanly-rendered multi-turn trajectories; the training step
#     produces interleaved Ray output that's harder to attribute per-
#     rollout). Both phases emit [NCU-GATE] lines, which the verifier
#     scans regardless of phase.
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

# Clear any prior smoke checkpoint. Otherwise the trainer resumes from
# global_step_1 and immediately exits because --total_epochs 1 is
# already satisfied, leaving zero training rollouts in the capture.
CKPT_ROOT="${DRKERNEL_WORKSPACE}/checkpoints/drkernel_baseline_ncu"
for d in "${CKPT_ROOT}"/*smoke_4problems*; do
  if [[ -d "${d}" ]]; then
    echo "[smoke] removing stale checkpoint: ${d}"
    rm -rf "${d}"
  fi
done

echo "[smoke] max_turn=${SMOKE_MAX_TURN}"
echo "[smoke] dataset=${SMOKE_DATASET}"
echo "[smoke] launcher=${LAUNCHER}"
echo "[smoke] capture=${CAPTURE_LOG}"
echo "[smoke] running launcher (val_before_train + 1 training step on 4 problems)..."

LAUNCHER_RC=0
VAL_MAX_TURN="${SMOKE_MAX_TURN}" TRAIN_FOREGROUND=1 "${LAUNCHER}" \
  --max_turn "${SMOKE_MAX_TURN}" \
  --val_before_train True \
  --train_batch_size 4 \
  --total_epochs 1 \
  --train_dataset "${SMOKE_DATASET}" \
  > "${CAPTURE_LOG}" 2>&1 || LAUNCHER_RC=$?

echo "[smoke] launcher exited with rc=${LAUNCHER_RC} (any non-zero exit after training rollouts complete is tolerable; the verifier reads the captured trajectory)."
echo "[smoke] running verifier on ${CAPTURE_LOG}..."

exec python3 "${SMOKE_DIR}/verify_trajectory.py" "${CAPTURE_LOG}" --max-turns "${SMOKE_MAX_TURN}"
