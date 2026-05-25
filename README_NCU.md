# NCU integration (this fork)

This fork adds **Nsight Compute (NCU) profiling on every non-final
rollout turn** to the sync RL pipeline. NCU runs after the timed perf
step, profiles the generated kernel(s), and feeds a short
SM/DRAM/TensorCore/occupancy summary back into the next-turn prompt so
the model can reason about where time is actually going.

Two cooperating processes (KernelGYM evaluator + VERL trainer) are
joined by one extra field, `enable_ncu`, threaded end-to-end.

> **Path conventions.** Paths under `kernelgym/...` and `drkernel/...`
> are inside this repo. Paths like `baseline/launch_drkernel_8b_rl_baseline_ncu.sh`,
> `logs/drkernel_baseline_ncu_rl/`, `envs/drkernel310/` refer to the
> broader workspace this fork is consumed by (the launcher scripts and
> training-artifact dirs live there, not in this repo).

Launched via `baseline/launch_drkernel_8b_rl_baseline_ncu.sh` →
`baseline/run_8b_rl_from_8bsft_baseline_ncu.sh` in that workspace.

## Why NCU runs on every non-final turn

NCU adds ~2–5 s per kernel and locks GPU clocks (which would distort the
timed perf pass that drives Fast@p). The design choices:

1. NCU runs **after** the timed perf step, never replacing it.
2. NCU runs on **every turn except the last configured turn** of a
   multi-turn rollout. Each turn's NCU summary is embedded in the
   next-turn user prompt, so a summary on turn `max_turns - 1` would
   have nowhere to go. Concretely with `max_turns=3`: NCU runs on
   turns 0 and 1; turn 2 (the last turn) is skipped.
3. NCU runs only on **correct** kernels — profiling a broken kernel is
   meaningless.

The trainer-side predicate is
`decide_enable_ncu(turn_idx, max_turns) -> (turn_idx < max_turns - 1)`,
extracted into `drkernel/kernel/rewards/ncu_gate.py` so it can be
unit-tested in isolation.

Three gates enforce the final go/no-go (any False stops the run):

- Trainer's per-call `enable_ncu` flag (True = run-if-env-on,
  False = skip, None = fall back to env).
- Env var `KERNELGYM_ENABLE_NCU=1` (global on/off).
- `kernel_exec_result.correctness == True`.

The server-side AND of these gates is
`should_run_ncu(enable_ncu_flag, env_enabled, is_correct)` in
`kernelgym/toolkit/kernelbench/ncu_gate.py`.

> **Early termination caveat.** If the model produces a final answer
> before `max_turns` (no further tool call), that turn is *effectively*
> the last turn, but we can't know that at gating time. NCU will run
> and its summary will be discarded. Cost: ~2-5s wasted per
> early-stop. The alternative (look-ahead detection) isn't reliable, so
> we accept the waste.

## KernelGYM side — runs `ncu` and attaches a summary

| File | Change |
|---|---|
| `kernelgym/toolkit/kernelbench/ncu_profile.py` | **New.** Builds a self-contained runner script (kernel + reference source base64-embedded, imported as modules to dodge pickling `ModelNew`), invokes `sudo -n /usr/local/cuda/bin/ncu --csv --section SpeedOfLight --metrics ...` against `envs/drkernel310/bin/python`, parses CSV → `{kernel_id: {name, metrics}}`. `build_ncu_summary()` renders top-5 kernels with roofline label, SM/DRAM/TensorCore %, occupancy, regs/thread, smem, L1/L2 hit, and a rule-based `_hint()` (e.g. "no tensor-core activity — consider tl.dot"). 120 s subprocess timeout. |
| `kernelgym/toolkit/kernelbench/pipeline.py` | New `_maybe_run_ncu_profile(...)`. Applies the three gates above. On success writes `metadata.ncu_summary`, `metadata.ncu_overhead_sec`, `metadata.ncu_num_kernels`. Wired into `eval_kernel_against_ref` **after** the perf step. |
| `kernelgym/toolkit/kernelbench/toolkit.py` | Passes `enable_ncu=task.enable_ncu` through to the pipeline (two call sites). |
| `kernelgym/workflow/kernelbench_helpers.py` | Same: forwards `task.enable_ncu`. |
| `kernelgym/schema/task.py` | Adds `enable_ncu: Optional[bool] = None` to both task dataclasses. |
| `kernelgym/server/api/models.py` | Adds `enable_ncu` to the HTTP request model. `None` → fall back to env var; `True/False` overrides per-call. |

### Why source-string input (not callable + pickle)

Pickling dynamically-defined classes like `ModelNew` fails on the receiving
side; the runner process can't import the original module. The runner
script writes the source to a temp file and imports it under a known name
inside the child process.

### Metric set (single SpeedOfLight pass)

```
sm__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active
sm__warps_active.avg.pct_of_peak_sustained_active
launch__registers_per_thread
launch__shared_mem_per_block_static
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
gpu__time_duration.sum
```

### Roofline label rules (`_roofline_label`)

| SM% | DRAM% | Label |
|---|---|---|
| ≥ 60 | — | compute-bound |
| < 60 | ≥ 60 | memory-bound — well-utilized |
| < 30 | < 30 | latency-bound or under-utilized |
| else | else | memory-bound — under-utilized |

### Hint rules (`_hint`)

| Condition | Hint |
|---|---|
| `tc < 1%` and `sm < 50%` | "no tensor-core activity — consider tl.dot for matmul-shaped reductions" |
| `occ < 30%` and `regs ≥ 64` | "low occupancy likely register-bound — reduce per-thread registers or BLOCK size" |
| `dram < 30%` and `sm < 30%` | "kernel under-utilized — increase work per program or check coalescing" |
| `dram > 60%` and `sm < 30%` | "bandwidth-saturated — fuse adjacent ops or reduce redundant reads" |
| else | "looks balanced; further gains likely need algorithmic changes" |

## Training side — non-final-turn gating

| File | Change |
|---|---|
| `drkernel/kernel/workers/rollout/vllm_rollout/vllm_async_engine.py` | Captures `turn_idx_for_ncu = req.get_num_turns()` and the surrounding `actual_max_turns`, then puts both in `reward_kwargs`. `get_num_turns()` is 0-indexed and reflects turns completed **before** this turn, so the turn that just generated the response has index == `get_num_turns()`. |
| `drkernel/kernel/workers/reward_manager/kernel_async.py` | `execute_env` now accepts `turn_idx` and `max_turns`; `_process_single_turn` forwards them from kwargs. Both pass through to `compute_score`. |
| `drkernel/kernel/rewards/kernel_reward.py` | Calls `decide_enable_ncu(turn_idx, max_turns)` (in `ncu_gate.py`). Stamps the result into the KGym task. Prints `[NCU-GATE] batch_size=... turn_idx=... max_turns=... enable_ncu=...` for greppable trainer logs. |
| `drkernel/kernel/rewards/ncu_gate.py` | **New.** `decide_enable_ncu(turn_idx, max_turns) -> Optional[bool]`. Pure function, no heavy imports → unit-testable. |
| `drkernel/kernel/rewards/reward_client.py` | Forwards `enable_ncu` into the HTTP payload only when explicitly `True/False`. `None` is omitted so the server falls back to its env var. |
| `kernelgym/toolkit/kernelbench/ncu_gate.py` | **New.** `should_run_ncu(enable_ncu_flag, env_enabled, is_correct) -> bool`. AND of the three server-side gates. Used by `pipeline._maybe_run_ncu_profile`. |

## Launchers (in the workspace, not in this repo)

- **`baseline/launch_drkernel_8b_rl_baseline_ncu.sh`** — clone of the
  baseline launcher with:
  - `BASELINE_DIR` → `KernelGYM-upstream-sync-ncu`
  - Checkpoint root → `checkpoints/drkernel_baseline_ncu`
  - Train log dir → `logs/drkernel_baseline_ncu_rl`
  - KernelGYM GPU layout shifted to **0,1,2,3** (vs `2,3` in baseline) for
    NCU headroom; `NODE_ID=baseline-gpu-0-1-2-3`
  - **`export KERNELGYM_ENABLE_NCU="1"`** before
    `start_all_with_monitor.sh`. Pydantic `Settings` rejects unknown keys
    in `.env`, so this must be a real env var — `_maybe_run_ncu_profile`
    reads `os.environ` directly.
  - PID files renamed to `*_ncu.pid`.
- **`baseline/run_8b_rl_from_8bsft_baseline_ncu.sh`** — identical to the
  baseline run script except `BASELINE_DIR` points at the sync-ncu tree.

## Data flow

```
vllm_async_engine (knows turn_idx + max_turns)
  → kernel_async reward manager (forwards both)
    → kernel_reward.compute_score
       → decide_enable_ncu(turn_idx, max_turns)  [trainer-side predicate]
         → enable_ncu_flag = (turn_idx < max_turns - 1)
           → reward_client (HTTP POST with enable_ncu field)
             → KernelGYM API models.py (accepts field)
               → schema/task.py (task object carries enable_ncu)
                 → toolkit.py / kernelbench_helpers.py (forwards)
                   → pipeline.eval_kernel_against_ref
                     → _maybe_run_ncu_profile
                       → should_run_ncu(flag, env, is_correct)  [server-side predicate]
                         → ncu_profile.profile_and_summarize_source
                           → metadata.ncu_summary in HTTP response
                             → next-turn prompt rendering picks it up
```

## GPU layout for the NCU run

| GPUs | Use |
|---|---|
| 0,1,2,3 | KernelGYM workers (4 workers, expanded from baseline's 2,3) |
| 4,5,6,7 | RL training (FSDP + async vLLM rollout) |

The expansion to 4 KGym GPUs absorbs the NCU wallclock overhead so it
doesn't bottleneck rollout.

## Where NCU output is stored on disk

Two log surfaces depending on what you want.

### Validation trajectories — clean, structured, full conversations

```
drkernel/wandb/
  offline-run-<timestamp>-<run_id>/files/media/table/val/generations_<step>_*.table.json
```

- Each row has `input_N` + `output_N` + `score_N` columns. `input_N` is
  the rendered turn-2 user prompt with the NCU summary embedded inside
  an `Env Result: {...}` JSON block; `output_N` is the model's reply.

### Training trajectories — every NCU call, in the giant RL log

```
logs/drkernel_baseline_ncu_rl/drkernel_8b_rl_<timestamp>.log
```

Effectively every NCU pass that ran during training. The reward manager
prints the entire `Env Result` dict for every rollout, so each entry
carries the NCU summary alongside `speedup`, `num_custom_kernel`,
`num_total_kernels`, per-aten-op CUDA timing breakdown, rollout
`task_id`, and timestamp. Each entry is one log line; newlines inside
the summary are escaped as `\n`.

Useful grep patterns:

```bash
# Every NCU summary (multi-KB lines)
grep "ncu_summary" logs/drkernel_baseline_ncu_rl/drkernel_8b_rl_<timestamp>.log

# Gating decisions
grep "NCU-GATE" logs/drkernel_baseline_ncu_rl/*.log

# Full Env Result lines (ncu_summary + reward + speedup + per-kernel times)
grep "Env Result:" logs/drkernel_baseline_ncu_rl/drkernel_8b_rl_<timestamp>.log
```

### Where NCU output is **not** stored

- `kernelgym/logs/kernelgym_gpu0123_ncu/*.log` — KGym worker/API logs.
  The `print(f"[Eval] Running NCU profile ...")` calls in `pipeline.py`
  are gated by `verbose=True`, not set in production.
- `drkernel/outputs/2026-*/main_kernel.log` — Hydra-managed Python logs;
  don't capture Ray worker stdout where `Env Result` prints land.

## Operational notes

- `KERNELGYM_ENABLE_NCU=1` is the global kill-switch. To turn NCU off
  without re-launching, unset it on the KernelGYM side and restart the
  workers — the trainer's `enable_ncu` flag is "no opinion / fall back to
  env var" semantics when the var is absent.
- `[NCU-GATE]` log lines in trainer stdout confirm gating decisions per
  batch.
- KGym log dir for this fork: `kernelgym/logs/kernelgym_gpu0123_ncu/`.
- Training log dir: `logs/drkernel_baseline_ncu_rl/` (workspace).
- The NCU pass is best-effort: timeout/crash → empty summary, training
  continues. Errors land in `metadata.ncu_error`.
