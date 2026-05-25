"""Unit tests for decide_enable_ncu (trainer-side predicate).

Policy under test: NCU runs on every non-final turn, i.e.
`turn_idx < max_turns - 1`. Missing or malformed inputs return None so
the server falls back to the KERNELGYM_ENABLE_NCU env var.
"""

import pytest

from tests.conftest import _load_module

decide_enable_ncu = _load_module(
    "_ncu_gate_trainer", "drkernel/kernel/rewards/ncu_gate.py"
).decide_enable_ncu


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "turn_idx, max_turns, expected",
    [
        # max_turns=3: turns 0 and 1 enable NCU; turn 2 is the last -> skip
        (0, 3, True),
        (1, 3, True),
        (2, 3, False),
        # max_turns=4 (one more non-final turn)
        (0, 4, True),
        (1, 4, True),
        (2, 4, True),
        (3, 4, False),
        # max_turns=1: the only turn IS the last turn -> always False
        (0, 1, False),
        # max_turns=2: only turn 0 is non-final
        (0, 2, True),
        (1, 2, False),
        # Out-of-range turn_idx (shouldn't happen, but predicate must be safe)
        (5, 3, False),
        (-1, 3, True),  # -1 < 3 - 1 = 2 -> True. Documented quirk; trainer
                        # never passes negative turn_idx, so this is purely
                        # a "predicate doesn't crash" assertion.
    ],
)
def test_predicate_matrix(turn_idx, max_turns, expected):
    assert decide_enable_ncu(turn_idx, max_turns) is expected


@pytest.mark.parametrize("turn_idx, max_turns", [
    (None, 3),
    (0, None),
    (None, None),
])
def test_none_inputs_return_none(turn_idx, max_turns):
    """Missing inputs -> None -> server falls back to env var."""
    assert decide_enable_ncu(turn_idx, max_turns) is None


@pytest.mark.parametrize("turn_idx, max_turns", [
    ("not-a-number", 3),
    (0, "not-a-number"),
    ([0], 3),
    (0, {"max": 3}),
])
def test_bad_types_return_none(turn_idx, max_turns):
    """Bad inputs -> None (no crash). Trainer shouldn't pass these, but
    the predicate is a soft boundary."""
    assert decide_enable_ncu(turn_idx, max_turns) is None


def test_string_digits_coerce():
    """int-coercible strings work, since some kwarg paths stringify ints."""
    assert decide_enable_ncu("0", "3") is True
    assert decide_enable_ncu("2", "3") is False
