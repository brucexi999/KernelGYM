"""Stage-1 Nsight Compute (NCU) integration for richer per-kernel feedback.

Runs `sudo -n ncu` against a small runner script that imports the kernel
source as a module, builds inputs, and invokes the forward pass once.
Parses the CSV output and builds a compact human-readable summary suitable
for embedding in the next-turn prompt fed back to the RL agent.

Why source-string input (not callable + pickle): pickling
dynamically-defined classes (ModelNew) fails on the receiving side; the
runner process can't import the original module. We sidestep by writing
the source to a file and importing it under a known name inside the
runner.

Why this is invoked AFTER the regular timing pass: NCU's clock-locking
changes apparent kernel runtime. Speedup measurement (which drives Fast@p)
must be kept separate from any NCU pass.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import subprocess
import tempfile
import textwrap
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("kernelgym.toolkit.kernelbench.ncu_profile")

# H100 peak hardware constants (context only — NCU reports % of peak directly).
H100_PEAK_HBM_GB_S = 3000.0
H100_PEAK_BF16_TFLOPS = 1979.0

# Compact metric set, single-section pass.
NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "gpu__time_duration.sum",
]

# Cap kernels included in rendered summary — keep prompt small.
MAX_KERNELS_IN_SUMMARY = 5

# Hard upper bound on NCU subprocess wall time (sec).
NCU_TIMEOUT_SEC = 120

NCU_BIN = "/usr/local/cuda/bin/ncu"
DEFAULT_PYTHON = "/home/ubuntu/z84318463/envs/drkernel310/bin/python"


def _build_runner_script(
    kernel_source: str,
    reference_source: Optional[str],
    entry_point: str,
    device_index: int,
) -> str:
    """Return a self-contained Python program for NCU to launch.

    The program:
      1. Writes both source strings to temp files inside the child process
         and imports them as modules (avoids re-execing source in __main__).
      2. Builds inputs using `get_inputs()` (from reference if provided,
         else from kernel module).
      3. Builds the ModelNew with `get_init_inputs()` if defined.
      4. Runs one forward pass, then forces a CUDA sync + a tiny readback
         (so NCU has fully flushed buffers before process exit).
    """
    # We embed the source code as Python triple-quoted strings.
    # Use repr-encoded bytes via base64 to avoid escaping nightmares.
    import base64
    kernel_b64 = base64.b64encode(kernel_source.encode()).decode()
    ref_b64 = (
        base64.b64encode(reference_source.encode()).decode()
        if reference_source else ""
    )

    return textwrap.dedent(f"""\
        import base64, importlib.util, os, sys, tempfile, torch

        device = torch.device("cuda:{device_index}")
        torch.cuda.set_device(device)

        def _import_from_b64(b64_src: str, modname: str):
            src = base64.b64decode(b64_src.encode()).decode()
            fd, path = tempfile.mkstemp(suffix=".py", prefix=modname + "_")
            with os.fdopen(fd, "w") as f:
                f.write(src)
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, path

        kernel_mod, kernel_path = _import_from_b64({kernel_b64!r}, "kernel_mod")
        ref_b64 = {ref_b64!r}
        if ref_b64:
            ref_mod, ref_path = _import_from_b64(ref_b64, "ref_mod")
            get_inputs = ref_mod.get_inputs
            get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
        else:
            get_inputs = kernel_mod.get_inputs
            get_init_inputs = getattr(kernel_mod, "get_init_inputs", lambda: [])

        init_args = get_init_inputs() or []
        inputs = get_inputs()

        def _to_cuda(x):
            return x.cuda(device=device) if isinstance(x, torch.Tensor) else x

        init_args = [_to_cuda(a) for a in init_args]
        inputs = [_to_cuda(x) for x in inputs]

        ModelNew = getattr(kernel_mod, {entry_point!r})
        model = ModelNew(*init_args).cuda(device=device)

        # Warmup once outside the profiled region. NCU profiles ALL kernel
        # launches in the process; we accept that the warmup launches will
        # also appear in the report but with their own metrics.
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = model(*inputs)
        torch.cuda.synchronize()

        # Profiled forward pass
        with torch.no_grad():
            out = model(*inputs)
        torch.cuda.synchronize()

        # Force buffer flush before process exit
        if isinstance(out, torch.Tensor):
            _ = out.float().sum().item()
        print("ok")
    """)


def run_ncu_profile_source(
    kernel_source: str,
    *,
    reference_source: Optional[str] = None,
    entry_point: str = "ModelNew",
    device: int = 0,
    python_bin: str = DEFAULT_PYTHON,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], float]:
    """Profile one forward pass of `ModelNew(*get_init_inputs())(*get_inputs())`.

    Returns (parsed_metrics_dict, wallclock_seconds). On failure → ({}, secs).
    """
    start = time.time()

    runner = _build_runner_script(kernel_source, reference_source, entry_point, device)
    runner_fd, runner_path = tempfile.mkstemp(suffix=".py", prefix="ncu_runner_")
    os.close(runner_fd)
    with open(runner_path, "w") as f:
        f.write(runner)

    try:
        ncu_cmd = [
            "sudo", "-n", NCU_BIN,
            "--target-processes", "all",
            "--csv",
            "--section", "SpeedOfLight",
            "--metrics", ",".join(NCU_METRICS),
            python_bin,
            runner_path,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        if env_overrides:
            env.update(env_overrides)

        try:
            proc = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                timeout=NCU_TIMEOUT_SEC,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"[ncu] timed out after {NCU_TIMEOUT_SEC}s")
            return {}, time.time() - start

        if proc.returncode != 0:
            logger.warning(
                f"[ncu] exit={proc.returncode}; stderr-tail: {proc.stderr[-400:]}"
            )
            return {}, time.time() - start

        return _parse_ncu_csv(proc.stdout), time.time() - start

    finally:
        try:
            os.unlink(runner_path)
        except OSError:
            pass


def _parse_ncu_csv(csv_text: str) -> Dict[str, Any]:
    """Parse NCU CSV output. Returns {ID: {name, metrics: {name: val}}}."""
    csv_start = csv_text.find('"ID"')
    if csv_start < 0:
        return {}
    rdr = csv.DictReader(io.StringIO(csv_text[csv_start:]))
    out: Dict[str, Any] = {}
    for row in rdr:
        kid = row.get("ID")
        if kid is None:
            continue
        entry = out.setdefault(kid, {"name": row.get("Kernel Name", ""), "metrics": {}})
        raw = (row.get("Metric Value") or "").replace(",", "")
        try:
            val: Any = float(raw)
        except (TypeError, ValueError):
            val = raw
        entry["metrics"][row.get("Metric Name", "")] = val
    return out


def _shorten(name: str, max_len: int = 90) -> str:
    if len(name) <= max_len:
        return name
    head = name.split("<", 1)[0]
    if 0 < len(head) <= max_len:
        return head + "<...>"
    return name[:max_len] + "..."


def _roofline_label(sm_pct: float, dram_pct: float) -> str:
    if sm_pct >= 60:
        return f"compute-bound ({sm_pct:.0f}% of SM peak)"
    if dram_pct >= 60:
        return f"memory-bound — well-utilized ({dram_pct:.0f}% of DRAM peak)"
    if dram_pct < 30 and sm_pct < 30:
        return f"latency-bound or under-utilized ({dram_pct:.0f}% DRAM, {sm_pct:.0f}% SM)"
    return f"memory-bound — under-utilized ({dram_pct:.0f}% of DRAM peak)"


def _hint(sm_pct: float, dram_pct: float, tc_pct: float,
          occ_pct: float, n_regs: int) -> str:
    h = []
    if tc_pct < 1.0 and sm_pct < 50.0:
        h.append("no tensor-core activity — consider tl.dot for matmul-shaped reductions")
    if occ_pct < 30.0 and n_regs >= 64:
        h.append("low occupancy likely register-bound — reduce per-thread registers or BLOCK size")
    if dram_pct < 30.0 and sm_pct < 30.0:
        h.append("kernel under-utilized — increase work per program or check coalescing")
    if dram_pct > 60.0 and sm_pct < 30.0:
        h.append("bandwidth-saturated — fuse adjacent ops or reduce redundant reads")
    return "; ".join(h) if h else "looks balanced; further gains likely need algorithmic changes"


def build_ncu_summary(ncu_data: Dict[str, Any]) -> str:
    if not ncu_data:
        return ""

    kernels = []
    for kid, entry in ncu_data.items():
        m = entry.get("metrics", {})
        kernels.append({
            "id": kid,
            "name": entry.get("name", ""),
            "dur_ns": float(m.get("gpu__time_duration.sum", 0) or 0),
            "m": m,
        })
    kernels.sort(key=lambda k: k["dur_ns"], reverse=True)
    kernels = kernels[:MAX_KERNELS_IN_SUMMARY]
    total = sum(k["dur_ns"] for k in kernels) or 1.0

    lines = [
        "=== Nsight Compute summary (top kernels by GPU time) ===",
        "Hardware: NVIDIA H100 80GB — peak HBM ~3000 GB/s, peak BF16 tensor ~1979 TFLOPS",
        "",
    ]
    for i, k in enumerate(kernels, 1):
        m = k["m"]
        sm_pct   = float(m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0) or 0)
        dram_pct = float(m.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0) or 0)
        tc_pct   = float(m.get("smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active", 0) or 0)
        occ_pct  = float(m.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0) or 0)
        n_regs   = int(m.get("launch__registers_per_thread", 0) or 0)
        smem     = int(m.get("launch__shared_mem_per_block_static", 0) or 0)
        l1_hit   = float(m.get("l1tex__t_sector_hit_rate.pct", 0) or 0)
        l2_hit   = float(m.get("lts__t_sector_hit_rate.pct", 0) or 0)
        dur_pct  = 100.0 * k["dur_ns"] / total

        lines.append(f"[{i}] {_shorten(k['name'])}")
        lines.append(f"    Time: {k['dur_ns']/1000:.2f} us ({dur_pct:.0f}% of profiled)")
        lines.append(f"    Roofline: {_roofline_label(sm_pct, dram_pct)}")
        lines.append(f"    SM {sm_pct:.0f}% | DRAM {dram_pct:.0f}% | TensorCores {tc_pct:.1f}%")
        lines.append(f"    Occupancy {occ_pct:.0f}% | regs/thread {n_regs} | smem {smem} B | L1 hit {l1_hit:.0f}% | L2 hit {l2_hit:.0f}%")
        lines.append(f"    Hint: {_hint(sm_pct, dram_pct, tc_pct, occ_pct, n_regs)}")
        lines.append("")

    return "\n".join(lines).rstrip()


def profile_and_summarize_source(
    kernel_source: str,
    *,
    reference_source: Optional[str] = None,
    entry_point: str = "ModelNew",
    device: int = 0,
) -> Dict[str, Any]:
    """Convenience: returns {raw_metrics, summary_text, wallclock_sec, num_kernels}."""
    raw, secs = run_ncu_profile_source(
        kernel_source,
        reference_source=reference_source,
        entry_point=entry_point,
        device=device,
    )
    return {
        "raw_metrics": raw,
        "summary_text": build_ncu_summary(raw),
        "ncu_wallclock_sec": round(secs, 2),
        "num_kernels": len(raw),
    }
