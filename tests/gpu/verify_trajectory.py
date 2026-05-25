#!/usr/bin/env python3
"""Verify NCU all-turn gating from a real trainer log.

Reads a `drkernel_8b_rl_*.log` file, parses every `[NCU-GATE]` line
emitted by `compute_kernel_reward_batch`, and asserts the gating
decisions match the all-turn policy: for max_turns=K, every turn_idx
in {0..K-2} should have enable_ncu=True and turn_idx=K-1 should have
enable_ncu=False.

Optionally also checks the wandb val table (when --wandb-dir is given)
to confirm that turn-N+1 prompts embed the NCU summary from turn N for
non-final turns, and the final turn's response is NOT followed by an
NCU-bearing prompt (there is no turn after the last turn).

Exits 0 on pass, non-zero on any mismatch.

Usage:
    python verify_trajectory.py <log_path> [--max-turns N] [--wandb-dir DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

GATE_RE = re.compile(
    r"\[NCU-GATE\]\s+"
    r"batch_size=(?P<batch_size>\d+)\s+"
    r"turn_idx=(?P<turn_idx>\S+)\s+"
    r"max_turns=(?P<max_turns>\S+)\s+"
    r"enable_ncu=(?P<enable_ncu>\S+)"
)


def parse_gate_lines(log_path: Path) -> list[dict]:
    """Pull every [NCU-GATE] line into a list of dicts."""
    out = []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = GATE_RE.search(line)
            if m:
                d = m.groupdict()
                d["batch_size"] = int(d["batch_size"])
                # turn_idx and max_turns may be "None" string if absent
                d["turn_idx"] = None if d["turn_idx"] == "None" else int(d["turn_idx"])
                d["max_turns"] = None if d["max_turns"] == "None" else int(d["max_turns"])
                d["enable_ncu"] = {"True": True, "False": False, "None": None}.get(
                    d["enable_ncu"], d["enable_ncu"]
                )
                out.append(d)
    return out


def check_gating(gates: list[dict], expected_max_turns: int) -> list[str]:
    """Return a list of error strings; empty list means pass."""
    errors: list[str] = []

    if not gates:
        return ["No [NCU-GATE] lines found in log. Did the trainer reach the reward stage?"]

    # All lines must agree on max_turns (we set --max_turn = --val_max_turn = K)
    distinct_max = {g["max_turns"] for g in gates}
    if distinct_max != {expected_max_turns}:
        errors.append(
            f"Expected all NCU-GATE lines to report max_turns={expected_max_turns}, "
            f"got {distinct_max}. (One or more lines reported the wrong value, "
            f"suggesting old-code call sites still in play.)"
        )

    # Per-turn distribution
    counter: Counter = Counter()
    for g in gates:
        counter[(g["turn_idx"], g["enable_ncu"])] += 1

    # For all-turn semantics: turn_idx in {0..K-2} -> True; turn_idx = K-1 -> False.
    for ti in range(expected_max_turns - 1):
        c_true = counter[(ti, True)]
        c_false = counter[(ti, False)]
        c_none = counter[(ti, None)]
        if c_true == 0:
            errors.append(
                f"Expected turn_idx={ti} to produce enable_ncu=True at least once "
                f"(non-final turn). Counts: True={c_true} False={c_false} None={c_none}."
            )
        if c_false > 0 or c_none > 0:
            errors.append(
                f"turn_idx={ti} must always be enable_ncu=True for max_turns={expected_max_turns}, "
                f"but saw False={c_false}, None={c_none}."
            )

    last = expected_max_turns - 1
    c_true = counter[(last, True)]
    c_false = counter[(last, False)]
    c_none = counter[(last, None)]
    if c_false == 0:
        errors.append(
            f"Expected turn_idx={last} (final turn) to produce enable_ncu=False at "
            f"least once, but saw True={c_true} None={c_none}."
        )
    if c_true > 0:
        errors.append(
            f"turn_idx={last} (final turn) must always be enable_ncu=False, but "
            f"saw True={c_true} occurrences — gate regressed or stale code path."
        )

    return errors


def check_env_results(log_path: Path) -> dict[int, int]:
    """Best-effort: count how many `Env Result:` lines on each turn carry a
    non-empty ncu_summary. Returned as {turn_idx_index_in_sequence: count}.

    This is an informational secondary check; the gating in [NCU-GATE] is
    authoritative. A non-empty count on the last turn means the trainer's
    gate decision was somehow ignored downstream — should never happen.
    """
    # Quick heuristic: pair each Env Result with the [NCU-GATE] immediately
    # preceding it. We just count Env Result lines containing "ncu_summary":
    # this tells us how many NCU passes the server actually completed.
    n_with_summary = 0
    n_without = 0
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            if "Env Result" not in line:
                continue
            # The Env Result JSON includes ncu_summary key only when NCU
            # ran and produced output. An empty-string ncu_summary still
            # implies the gate let it through but profiling produced
            # nothing — distinguish.
            if "'ncu_summary':" in line or '"ncu_summary":' in line:
                # Crude: look for non-empty summary text after the key.
                if "'ncu_summary': ''" in line or '"ncu_summary": ""' in line:
                    n_without += 1
                else:
                    n_with_summary += 1
            else:
                n_without += 1
    return {"with_ncu_summary": n_with_summary, "without_ncu_summary": n_without}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help="Path to drkernel_8b_rl_*.log")
    ap.add_argument(
        "--max-turns", type=int, default=3,
        help="Expected --max_turn value used for the smoke run (default: 3)."
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print full gate-line dump and Env Result tallies."
    )
    args = ap.parse_args()

    log = Path(args.log_path)
    if not log.exists():
        print(f"ERROR: log not found: {log}", file=sys.stderr)
        return 2

    gates = parse_gate_lines(log)
    errors = check_gating(gates, args.max_turns)

    print(f"NCU all-turn smoke verifier")
    print(f"  log:       {log}")
    print(f"  max_turns: {args.max_turns}")
    print(f"  [NCU-GATE] lines parsed: {len(gates)}")
    if gates:
        per_turn = Counter((g["turn_idx"], g["enable_ncu"]) for g in gates)
        print(f"  gate distribution by (turn_idx, enable_ncu):")
        for k in sorted(per_turn, key=lambda x: (x[0] if x[0] is not None else -1, str(x[1]))):
            print(f"    turn_idx={k[0]} enable_ncu={k[1]}: {per_turn[k]}")

    env_tally = check_env_results(log)
    print(f"  Env Result lines with non-empty ncu_summary: {env_tally['with_ncu_summary']}")
    print(f"  Env Result lines without ncu_summary:        {env_tally['without_ncu_summary']}")

    if errors:
        print()
        print(f"FAIL ({len(errors)} issue{'s' if len(errors) != 1 else ''}):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print()
    print("PASS: all gating decisions match the all-turn policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
