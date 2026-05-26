#!/usr/bin/env python3
"""Live monitor for the all-turn NCU training run.

Watches a `drkernel_8b_rl_*.log` (passed as argv[1] or auto-discovered),
parses training-step and eval-step events, and prints structured updates.
Also checks for process hang, training-driver death, and OOM markers.

Usage:
    python3 /tmp/training_monitor.py [/path/to/training.log]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional


# ----- log line regexes -----------------------------------------------------

ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

# Step boundary: the trainer prints multiple `step:N - ...` lines per
# step (one short header w/ over_sampling/, then the long metrics dump).
# We want the LONG metrics dump line — for training it starts with
# `batch/` after the step prefix; for eval it contains `val/test_score`.
# Match either; the dispatcher below picks eval vs train.
STEP_LINE = re.compile(
    r'\bstep:(\d+)\s*-\s+(?:batch/|val/test_score/|val/kernel/)'
)

# Best-of-3 metrics (printed in a wandb-pretty-print block on separate
# lines, one key per line). The plot script uses the `_in_all` variants
# (denominator = all 100 val problems, not just correct ones), so we
# prefer those.
BEST3_LINE = re.compile(
    r"val/kernel/best_by_turn_3/([A-Za-z0-9_@.]+)"
    r"['\"]?\s*:\s*([0-9.+\-eE]+)"
)

# Training-step scalar metrics — exact keys this verl/drkernel build emits
# in the `step:N - <kv> - <kv> ...` dump line.
TRAIN_KV = re.compile(
    r"['\"]?("
    r"critic/score/mean|"
    r"critic/rewards/mean|"
    r"critic/rewards_extra/correctness/mean|"
    r"timing_s/step|"
    r"actor/grad_norm|"
    r"response_length/mean|"
    r"train/kernel/best_by_turn_3/correctness_rate|"
    r"train/kernel/best_by_turn_3/mean_performance_in_all|"
    r"train/kernel/best_by_turn_3/fast@1_in_all"
    r")['\"]?\s*:\s*([0-9.+\-eE]+)"
)

# OOM / crash markers
OOM_PATTERNS = (
    re.compile(r'CUDA out of memory', re.IGNORECASE),
    re.compile(r'out of memory', re.IGNORECASE),
    re.compile(r'RuntimeError: CUDA', re.IGNORECASE),
    re.compile(r'\bkilled\b', re.IGNORECASE),
    re.compile(r'SIGKILL'),
    re.compile(r'Ray task error', re.IGNORECASE),
    re.compile(r'NCCL.*error', re.IGNORECASE),
)


# ----- utils ----------------------------------------------------------------


def now_str() -> str:
    return datetime.now(timezone.utc).strftime('%H:%M:%SZ')


def driver_alive() -> bool:
    """True iff a kernel.main_kernel process is currently running."""
    r = subprocess.run(['pgrep', '-f', 'kernel.main_kernel'],
                       capture_output=True)
    return r.returncode == 0


def find_latest_log(log_dir: str) -> Optional[str]:
    cands = sorted(glob.glob(os.path.join(log_dir, 'drkernel_8b_rl_*.log')))
    return cands[-1] if cands else None


def wait_for_log(log_dir: str, max_wait_sec: int = 600) -> Optional[str]:
    start = time.time()
    while time.time() - start < max_wait_sec:
        f = find_latest_log(log_dir)
        if f and os.path.getsize(f) > 0:
            return f
        time.sleep(2)
    return None


# ----- main parse loop ------------------------------------------------------


def tail(path: str, from_start: bool = False):
    """Generator yielding new lines as the file grows (poll-based).

    With from_start=True, reads the whole existing file first (useful for
    backfilling step events that fired before the monitor started), then
    continues tailing for new appends.
    """
    with open(path, 'r', errors='replace') as f:
        if not from_start:
            f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('log', nargs='?', default=None,
                    help='path to training log (auto-discover if omitted)')
    ap.add_argument('--log-dir',
                    default='/home/ubuntu/z84318463/logs/drkernel_baseline_ncu_rl',
                    help='where to auto-discover the latest *.log')
    ap.add_argument('--hang-sec', type=int, default=900,
                    help='alert if no new line for this many seconds (default 15 min)')
    ap.add_argument('--from-start', action='store_true',
                    help='read the whole existing log first (backfill step events) before tailing')
    args = ap.parse_args()

    log_path = args.log
    if log_path is None:
        print(f'[{now_str()}] [monitor] waiting for log under {args.log_dir} ...', flush=True)
        log_path = wait_for_log(args.log_dir, max_wait_sec=900)
        if log_path is None:
            print(f'[{now_str()}] [monitor] FATAL: no log appeared within 15 min', flush=True)
            return 2

    print(f'[{now_str()}] [monitor] tailing {log_path}', flush=True)

    # Per-step accumulators (best_by_turn_3 keys + training scalars are
    # printed line-by-line by wandb's pretty-printer; we buffer them and
    # flush on each `step:N - ...` boundary line, classifying that step
    # as eval (line contains `val/test_score`) or train (otherwise).
    pending_best3: dict[str, float] = {}
    pending_train: dict[str, float] = {}
    last_step_printed: tuple[int, str] = (-1, "")

    last_line_ts = time.time()
    last_alive_check = time.time()
    last_driver_alive = True

    for raw in tail(log_path, from_start=args.from_start):
        last_line_ts = time.time()
        line = ANSI.sub('', raw)

        # 1) collect best-by-turn-3 metrics — a single `step:N - ...` line
        # can contain many `key:value` pairs separated by ` - `, so we must
        # capture ALL matches per line (re.findall, not re.search).
        for key, val in BEST3_LINE.findall(line):
            try:
                pending_best3[key] = float(val)
            except ValueError:
                pass

        # 2) collect training scalar metrics, same rationale
        for key, val in TRAIN_KV.findall(line):
            try:
                pending_train[key] = float(val)
            except ValueError:
                pass

        # 3) step boundary line: `step:N - <key>:<val> - <key>:<val> - ...`
        # Classify as eval vs train by content, flush the matching buffer.
        m = STEP_LINE.search(line)
        if m:
            step = int(m.group(1))
            is_eval = 'val/test_score' in line or 'val/kernel/' in line
            kind = 'EVAL' if is_eval else 'TRAIN'

            # Dedupe — the trainer often prints multiple `step:N - ...` lines per
            # step (one per metric group). Only print on the first kind transition.
            if (step, kind) == last_step_printed:
                continue
            last_step_printed = (step, kind)

            def fmt(x, p=3):
                return f'{x:.{p}f}' if x is not None else '?'

            if is_eval:
                cor = pending_best3.get('correctness_rate')
                # Prefer _in_all variants (denominator = all 100 problems, like
                # plot_baseline_vs_ncu.py); fall back to plain if absent.
                f1 = pending_best3.get('fast@1_in_all') or pending_best3.get('fast@1')
                f12 = pending_best3.get('fast@1.2_in_all') or pending_best3.get('fast@1.2')
                f15 = pending_best3.get('fast@1.5_in_all') or pending_best3.get('fast@1.5')
                meanp = pending_best3.get('mean_performance_in_all') or pending_best3.get('mean_performance')
                maxp = pending_best3.get('max_performance')
                count = pending_best3.get('count')

                print(
                    f'\n[{now_str()}] [EVAL step={step}] best_by_turn_3: '
                    f'correctness={fmt(cor)}  fast@1={fmt(f1)}  '
                    f'fast@1.2={fmt(f12)}  fast@1.5={fmt(f15)}  '
                    f'mean_perf={fmt(meanp)}  max_perf={fmt(maxp,2)}  '
                    f'n={int(count) if count else "?"}',
                    flush=True,
                )
                pending_best3.clear()
            else:
                cor = pending_train.get('critic/rewards_extra/correctness/mean')
                cor_bt3 = pending_train.get('train/kernel/best_by_turn_3/correctness_rate')
                score = pending_train.get('critic/score/mean')
                stime = pending_train.get('timing_s/step')
                grad = pending_train.get('actor/grad_norm')
                rlen = pending_train.get('response_length/mean')
                f1_bt3 = pending_train.get('train/kernel/best_by_turn_3/fast@1_in_all')
                perf_bt3 = pending_train.get('train/kernel/best_by_turn_3/mean_performance_in_all')

                print(
                    f'[{now_str()}] [TRAIN step={step}] '
                    f'step_time={fmt(stime,1)}s  '
                    f'correctness_mean={fmt(cor)}  '
                    f'critic/score/mean={fmt(score)}  '
                    f'grad_norm={fmt(grad,2)}  '
                    f'resp_len={fmt(rlen,0)}  '
                    f'best_by_turn_3: correctness={fmt(cor_bt3)} '
                    f'fast@1={fmt(f1_bt3)} mean_perf={fmt(perf_bt3)}',
                    flush=True,
                )
                pending_train.clear()
            continue

        # 5) OOM / crash markers — but skip Env Result JSON blobs and
        # other normal-during-training error-string-containing lines.
        # We only want REAL crashes (driver / worker process death,
        # actual CUDA errors), not the model's own failed kernel
        # compilations that get echoed back in server feedback.
        if 'Env Result:' in line or 'runtime_error' in line or 'error_message' in line:
            pass
        else:
            for pat in OOM_PATTERNS:
                if pat.search(line):
                    snippet = line.strip()[:200]
                    print(
                        f'[{now_str()}] [ALERT] crash marker matched: {snippet}',
                        flush=True,
                    )
                    break

        # 6) periodic process-alive + hang check (every 30 s)
        if time.time() - last_alive_check > 30:
            last_alive_check = time.time()
            alive = driver_alive()
            if not alive and last_driver_alive:
                print(
                    f'[{now_str()}] [ALERT] training driver (kernel.main_kernel) is no longer running',
                    flush=True,
                )
            last_driver_alive = alive

            silence = time.time() - last_line_ts
            if silence > args.hang_sec:
                print(
                    f'[{now_str()}] [ALERT] no new log lines for {silence/60:.1f} min (hang threshold {args.hang_sec/60:.0f} min)',
                    flush=True,
                )
                # reset so we don't spam every 30s
                last_line_ts = time.time()


if __name__ == '__main__':
    sys.exit(main() or 0)
