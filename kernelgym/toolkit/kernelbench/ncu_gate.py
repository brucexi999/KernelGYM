"""NCU gating predicate (KernelGYM server side).

AND of three gates: explicit per-call `enable_ncu_flag`, global
`KERNELGYM_ENABLE_NCU` env var, and kernel correctness. Pure function
with no heavy imports so it can be unit-tested in isolation.
"""

from typing import Optional


def should_run_ncu(
    enable_ncu_flag: Optional[bool],
    env_enabled: bool,
    is_correct: bool,
) -> bool:
    """Return True iff all three gates allow NCU to run.

    Args:
      enable_ncu_flag: Per-call override from the trainer.
        - False -> always skip (turns 2+ in first-turn-only mode).
        - True or None -> fall through to env_enabled.
      env_enabled: Whether KERNELGYM_ENABLE_NCU=1 is set (global on/off).
      is_correct: Whether the kernel passed correctness; profiling a
        broken kernel is meaningless.
    """
    if not is_correct:
        return False
    if enable_ncu_flag is False:
        return False
    return bool(env_enabled)
