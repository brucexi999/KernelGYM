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

# Eval-step boundary: line starting with "step:N - val/test_score/..."
EVAL_LINE = re.compile(r'\bstep:(\d+)\s*-\s*val/test_score/')

# Best-of-3 metrics (printed BEFORE the eval-step boundary line, one key per log line):
#   'val/kernel/best_by_turn_3/correctness_rate': 0.56,
#   'val/kernel/best_by_turn_3/fast@1': 0.7589,
#   ... etc
BEST3_LINE = re.compile(
    r"val/kernel/best_by_turn_3/([A-Za-z0-9_@.]+)"
    r"['\"]?\s*:\s*([0-9.+\-eE]+)"
)

# Training-step metric dump: same step:N prefix but with training metrics
# instead of val/. The trainer prints these every step.
# Examples that appear in logs:
#   'critic/score/mean': 0.123
#   'critic/score/max': 1.234
#   'response_length/mean': 8192
#   'training/global_step': 5
TRAIN_KV = re.compile(
    r"['\"](critic/score/mean|critic/score/max|critic/score/min|"
    r"response_length/mean|prompt_length/mean|"
    r"timing_s/step|"
    r"training/global_step|"
    r"actor/grad_norm|actor/pg_loss|"
    r"reward_extra/correctness/mean|reward_extra/correctness_tensor/mean|"
    r"reward_extra/performance/mean|reward_extra/is_speedup_positive/mean)"
    r"['\"]\s*:\s*([0-9.+\-eE]+)"
)

# Training-step boundary: a line like "step:N - training/...:Y - ..."
# (mirror of the eval boundary). Some verl versions emit a `Training step N done` line.
TRAIN_BOUNDARY = re.compile(r'\bstep:(\d+)\s*-\s*training/')

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


def tail(path: str):
    """Generator yielding new lines as the file grows (poll-based)."""
    with open(path, 'r', errors='replace') as f:
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
    args = ap.parse_args()

    log_path = args.log
    if log_path is None:
        print(f'[{now_str()}] [monitor] waiting for log under {args.log_dir} ...', flush=True)
        log_path = wait_for_log(args.log_dir, max_wait_sec=900)
        if log_path is None:
            print(f'[{now_str()}] [monitor] FATAL: no log appeared within 15 min', flush=True)
            return 2

    print(f'[{now_str()}] [monitor] tailing {log_path}', flush=True)

    # Per-eval-step accumulators (best_by_turn_3 keys appear BEFORE the
    # `step:N - val/test_score/` boundary line). We buffer them until we see
    # the boundary.
    pending_best3: dict[str, float] = {}
    pending_train: dict[str, float] = {}
    last_train_step_printed: int = -1
    last_eval_step_printed: int = -1

    last_line_ts = time.time()
    last_alive_check = time.time()
    last_driver_alive = True

    for raw in tail(log_path):
        last_line_ts = time.time()
        line = ANSI.sub('', raw)

        # 1) collect best-by-turn-3 metrics as they stream
        m = BEST3_LINE.search(line)
        if m:
            key, val = m.group(1), float(m.group(2))
            pending_best3[key] = val
            continue

        # 2) collect training scalar metrics as they stream
        m = TRAIN_KV.search(line)
        if m:
            key, val = m.group(1), float(m.group(2))
            pending_train[key] = val
            continue

        # 3) eval-step boundary line
        m = EVAL_LINE.search(line)
        if m:
            step = int(m.group(1))
            if step != last_eval_step_printed:
                last_eval_step_printed = step
                cor = pending_best3.get('correctness_rate')
                f1 = pending_best3.get('fast@1')
                f12 = pending_best3.get('fast@1.2')
                f15 = pending_best3.get('fast@1.5')
                meanp = pending_best3.get('mean_performance')
                maxp = pending_best3.get('max_performance')
                count = pending_best3.get('count')

                def fmt(x, p=4):
                    return f'{x:.{p}f}' if x is not None else '?'

                print(
                    f'\n[{now_str()}] [EVAL step={step}] best_by_turn_3: '
                    f'correctness={fmt(cor,3)}  fast@1={fmt(f1,3)}  '
                    f'fast@1.2={fmt(f12,3)}  fast@1.5={fmt(f15,3)}  '
                    f'mean_perf={fmt(meanp,3)}  max_perf={fmt(maxp,2)}  '
                    f'n={int(count) if count else "?"}',
                    flush=True,
                )
                pending_best3.clear()
            continue

        # 4) training-step boundary line
        m = TRAIN_BOUNDARY.search(line)
        if m:
            step = int(m.group(1))
            if step != last_train_step_printed:
                last_train_step_printed = step

                def fmt(x, p=4):
                    return f'{x:.{p}f}' if x is not None else '?'

                cor = pending_train.get('reward_extra/correctness/mean') or \
                      pending_train.get('reward_extra/correctness_tensor/mean')
                score = pending_train.get('critic/score/mean')
                stime = pending_train.get('timing_s/step')
                grad = pending_train.get('actor/grad_norm')
                perf = pending_train.get('reward_extra/performance/mean')
                pos = pending_train.get('reward_extra/is_speedup_positive/mean')

                print(
                    f'[{now_str()}] [TRAIN step={step}] '
                    f'step_time={fmt(stime,1)}s  '
                    f'correctness_mean={fmt(cor,3)}  '
                    f'critic/score/mean={fmt(score,3)}  '
                    f'perf_mean={fmt(perf,3)}  '
                    f'fast@1={fmt(pos,3)}  '
                    f'grad_norm={fmt(grad,2)}',
                    flush=True,
                )
                pending_train.clear()
            continue

        # 5) OOM / crash markers
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
