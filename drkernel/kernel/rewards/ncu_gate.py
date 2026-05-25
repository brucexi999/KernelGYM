"""NCU gating predicate (trainer side).

Decides whether NCU profiling should be requested for the kernel that
was just generated on a given rollout turn. Pure function with no heavy
imports so it can be unit-tested in isolation.
"""

from typing import Optional


def decide_enable_ncu(turn_idx: Optional[int]) -> Optional[bool]:
    """First-turn-only gate.

    Returns:
      - True  if this is turn 0 (the model's first attempt).
      - False if this is turn 1+.
      - None  if turn_idx is unknown or malformed; the server then falls
              back to the KERNELGYM_ENABLE_NCU env var.
    """
    if turn_idx is None:
        return None
    try:
        return int(turn_idx) == 0
    except (TypeError, ValueError):
        return None
