# GPU smoke test — NCU all-turn gating

Real-hardware verification that NCU profiles every non-final rollout
turn and skips the last turn. Run on a host with the full DR.Kernel
workspace (CUDA, `ncu`, the 8B model checkpoint, the existing
KernelGYM + Redis setup).

## What it does

1. `run_three_turn_smoke.sh` invokes the workspace launcher with
   overrides that produce a 3-turn rollout (`--max_turn 3`) on a single
   problem (`--train_batch_size 1`, `--n_val 1`, the 1-row fixture
   under `fixtures/smoke_1problem.parquet`). Launcher stdout/stderr is
   captured into `/tmp/ncu_all_turn_smoke_<ts>.log`.
2. The val_before_train phase runs the full 3-turn rollout over the
   validation set and emits the `[NCU-GATE]` lines that the verifier
   needs. The subsequent training step is expected to crash with
   "AssertionError: only support equal chunk" because
   `train_batch_size=1` can't divide evenly across the 4 training GPUs
   — the smoke script tolerates this; the val output is what we test.
3. It then runs `verify_trajectory.py` against the captured log.

Expected gating decisions for `max_turns=3`:

| turn_idx | enable_ncu | reason |
|---|---|---|
| 0 | True  | non-final; NCU summary feeds turn-1 prompt |
| 1 | True  | non-final; NCU summary feeds turn-2 prompt |
| 2 | False | last turn; nowhere to send the summary |

The verifier asserts this exactly: any deviation (e.g. turn 1 = False
because old first-turn-only code is still on the path, or turn 2 = True
because a stale gate let it through) exits non-zero with a clear diff.

## Prerequisites

- DR.Kernel workspace at a known path (the dir containing `baseline/`,
  `logs/`, `models/`, `envs/`, `third_party/`).
- Workspace KernelGYM brought up beforehand by the launcher itself —
  the launcher handles boot.
- `KERNELGYM_ENABLE_NCU=1` set inside the launcher (already wired in
  `launch_drkernel_8b_rl_baseline_ncu.sh`).
- Sufficient GPU layout: 4 GPUs for KGym (0,1,2,3) + 4 for training
  (4,5,6,7), as documented in the top-level `README_NCU.md`.

## Running

```bash
cd /path/to/KernelGYM-upstream-sync-ncu
export DRKERNEL_WORKSPACE=/path/to/the/parent/workspace
bash tests/gpu/run_three_turn_smoke.sh
```

Expected wallclock: ~20 min on H100s. Most of it is the val_before_train
phase (vLLM startup + 100 val problems × 3 turns × KGym eval). The
training step that follows crashes ~5 s in (by design); the smoke
script swallows that exit code and runs the verifier on the captured
trajectory.

Override knobs (env vars):

- `SMOKE_MAX_TURN` — default `3`. Use `4` to verify that turns 0,1,2
  enable NCU and turn 3 skips it.

## Verifier-only mode

If you already have a captured runner log from a recent smoke (under
`/tmp/ncu_all_turn_smoke_<ts>.log`), skip the launcher and verify:

```bash
python3 tests/gpu/verify_trajectory.py /tmp/ncu_all_turn_smoke_<ts>.log --max-turns 3
```

The verifier strips Ray's ANSI color codes from `[NCU-GATE]` lines
before parsing, so it works on raw captured stdout from Ray workers.

The verifier also reports a tally of `Env Result` lines with non-empty
`ncu_summary` — secondary signal that the server-side gate is honoring
the trainer's decision.

## Common failure modes

| Symptom | Likely cause |
|---|---|
| "No [NCU-GATE] lines found" | Log predates the all-turn change (old format had no `max_turns=` field). Re-run the smoke against new code. |
| "turn_idx=2 ... must always be enable_ncu=False, but saw True=N" | Trainer-side predicate regression — `decide_enable_ncu` returned True on the final turn. Check `drkernel/kernel/rewards/ncu_gate.py`. |
| "turn_idx=1 must always be enable_ncu=True, but saw False=N" | Old first-turn-only logic still in play. Check that `vllm_async_engine.py` passes `max_turns` in `reward_kwargs`. |
| `Env Result with non-empty ncu_summary` count is 0 | Gating decisions correct, but model never produced a correct kernel on a non-final turn — change the fixture row or accept that this run's model couldn't compile. The gating check still passes. |

## Why this is not in CI

Real `ncu` requires `sudo`, a CUDA GPU, the 8B checkpoint, and
multi-process orchestration that exceeds GitHub free-runner budgets.
Run this manually on the host that has the full workspace, gate
releases on it passing.
