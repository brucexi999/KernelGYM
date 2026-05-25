"""Shared test helpers.

We deliberately avoid importing the helper modules through their normal
package paths (`kernel.rewards.ncu_gate`, `kernelgym.toolkit.kernelbench.ncu_gate`)
because those packages eagerly import heavy deps (ray, torch, vllm, the
KGym backends) through their `__init__.py` chains. Instead we load each
helper directly from its file location via `importlib.util`.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(unique_name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(unique_name, REPO_ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
