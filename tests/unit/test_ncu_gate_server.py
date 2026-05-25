"""Unit tests for should_run_ncu (KGym server-side predicate).

Three-gate AND: per-call enable_ncu_flag, global env_enabled, and
kernel correctness. Any False stops the run.
"""

import pytest

from tests.conftest import _load_module

should_run_ncu = _load_module(
    "_ncu_gate_server", "kernelgym/toolkit/kernelbench/ncu_gate.py"
).should_run_ncu


pytestmark = pytest.mark.unit


# Full 3 x 2 x 2 truth table. Only two rows should be True:
#   - (True, True, True)       trainer says yes, env on, kernel correct
#   - (None, True, True)       trainer is "no opinion", env on, kernel correct
# Everything else is False (one of the three gates blocks).
TRUTH_TABLE = [
    # (flag, env_enabled, is_correct, expected)
    (True,  True,  True,  True),
    (True,  True,  False, False),   # correctness gate
    (True,  False, True,  False),   # env gate
    (True,  False, False, False),
    (False, True,  True,  False),   # per-call override
    (False, True,  False, False),
    (False, False, True,  False),
    (False, False, False, False),
    (None,  True,  True,  True),    # fall through to env
    (None,  True,  False, False),
    (None,  False, True,  False),
    (None,  False, False, False),
]


@pytest.mark.parametrize(
    "flag, env_enabled, is_correct, expected", TRUTH_TABLE
)
def test_truth_table(flag, env_enabled, is_correct, expected):
    assert should_run_ncu(flag, env_enabled, is_correct) is expected


def test_correctness_gate_is_first():
    """A broken kernel is never profiled regardless of any other flag."""
    for flag in (True, False, None):
        for env in (True, False):
            assert should_run_ncu(flag, env, is_correct=False) is False


def test_env_gate_required_when_flag_is_none():
    """When the trainer has no opinion, server's env var is authoritative."""
    assert should_run_ncu(None, True, True) is True
    assert should_run_ncu(None, False, True) is False


def test_explicit_false_overrides_env():
    """enable_ncu=False from trainer disables even with env var on."""
    assert should_run_ncu(False, True, True) is False
