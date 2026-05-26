# GPU smoke test — NCU all-turn gating

Real-hardware verification that NCU profiles every non-final rollout
turn and skips the last turn. Run on a host with the full DR.Kernel
workspace (CUDA, `ncu`, the 8B model checkpoint, the existing
KernelGYM + Redis setup).

## What it does

1. `run_three_turn_smoke.sh` invokes the workspace launcher with
   overrides that run a single training step on a 4-problem fixture
   (`--max_turn 3 --train_batch_size 4 --total_epochs 1
   --val_before_train False`). Launcher stdout/stderr is captured into
   `/tmp/ncu_all_turn_smoke_<ts>.log`.
2. The training step drives a full 3-turn rollout per sample with
   `rollout.n=16` generations per problem, so 4 × 16 = 64 trajectories
   per step. Each non-final turn of each correct kernel triggers a
   real NCU profile pass against the running KGym server. The training
   step's `[NCU-GATE]` lines + `Env Result` blocks land in the
   captured log.
3. It then runs `verify_trajectory.py` against the captured log,
   asserting the all-turn policy (`enable_ncu=True` on turns 0,1;
   `=False` on turn 2).

`train_batch_size=4` is required because the rollout chunks across
`n_gpus_per_node=4`; any batch size that doesn't divide evenly will
trigger `AssertionError: only support equal chunk`.

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

Expected wallclock: ~25-35 min on H100s, dominated by vLLM startup
(~5 min) and the 64-trajectory training rollout with NCU passes on
non-final turns (~20-25 min depending on how many kernels compile
successfully). The launcher may still exit non-zero after the step
completes (logprob recompute / optimizer step / checkpoint save can
hit unrelated edges with a 4-problem fixture) — the smoke script
tolerates this and runs the verifier on the captured trajectory
regardless.

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

## Live monitor for a running training job

`training_monitor.py` tails a `drkernel_8b_rl_*.log` and prints
structured events as the trainer progresses. Useful when babysitting a
real (multi-hour) RL training run.

```bash
# Auto-discover the latest log under
# /home/ubuntu/z84318463/logs/drkernel_baseline_ncu_rl/
python3 tests/gpu/training_monitor.py

# Or point at an explicit file:
python3 tests/gpu/training_monitor.py /path/to/drkernel_8b_rl_<ts>.log
```

What it prints:

- **`[TRAIN step=N | step_time=Xs | correctness_mean=Y | critic/score/mean=Z | ...]`**
  one line per training step.
- **`[EVAL step=N | best_by_turn_3: correctness=A | fast@1=B | fast@1.2=C | fast@1.5=D | mean_perf=E | max_perf=F | n=G]`**
  one line per eval step, sourced from `val/kernel/best_by_turn_3/*` scalars
  (the same metrics `plot_baseline_vs_ncu.py` plots).
- **`[ALERT] ...`** on:
  - `kernel.main_kernel` process disappearing (training driver dead),
  - 15+ min of log silence (hang),
  - OOM / CUDA error / NCCL error / Ray task error markers in the log.

Defaults: `--hang-sec 900` (15 min silence triggers alert),
`--log-dir /home/ubuntu/z84318463/logs/drkernel_baseline_ncu_rl/` for
auto-discovery.

Usage pattern for a babysat run:
```bash
bash /home/ubuntu/z84318463/baseline/launch_drkernel_8b_rl_baseline_ncu.sh --save_freq 60
# launcher detaches into background; then in another shell:
python3 tests/gpu/training_monitor.py | tee /tmp/monitor.log
```

## Why this is not in CI

Real `ncu` requires `sudo`, a CUDA GPU, the 8B checkpoint, and
multi-process orchestration that exceeds GitHub free-runner budgets.
Run this manually on the host that has the full workspace, gate
releases on it passing.
