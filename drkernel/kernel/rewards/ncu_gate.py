"""NCU gating predicate (trainer side).

Decides whether NCU profiling should be requested for the kernel that
was just generated on a given rollout turn. Pure function with no heavy
imports so it can be unit-tested in isolation.
"""

from typing import Optional


def decide_enable_ncu(
    turn_idx: Optional[int],
    max_turns: Optional[int],
) -> Optional[bool]:
    """Enable NCU on every turn except the last configured turn.

    The NCU summary is only useful when there is a *next* turn whose
    prompt can consume it. Turn `max_turns - 1` is the last configured
    turn, so the summary would be wasted there.

    Returns:
      - True  if this is turn 0 .. max_turns-2 (a non-final turn).
      - False if this is turn max_turns-1 (the last turn) or beyond.
      - None  if either input is unknown or malformed; the server then
              falls back to the KERNELGYM_ENABLE_NCU env var.

    Note: early termination (model produces a final answer before
    max_turns) is not detectable here, so we may run NCU on a turn that
    happens to be the *effective* last turn. The wasted ~2-5s is the
    cost; the alternative (look-ahead detection) is not reliable.
    """
    if turn_idx is None or max_turns is None:
        return None
    try:
        return int(turn_idx) < int(max_turns) - 1
    except (TypeError, ValueError):
        return None
