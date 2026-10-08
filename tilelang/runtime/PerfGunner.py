"""Runtime performance measurement for TileLang — sherlock-it-style tracing.

Usage::

    import os
    os.environ["TILELANG_PERF"] = "1"
    import tilelang as tl
    from tilelang.runtime import PerfGunner

    # Option A: context-managed region
    with PerfGunner.region("my_kernel_run"):
        out = tl.autotune(...)(...)  # your tilelang program

    # Option B: explicit trace collection
    PerfGunner.enable()
    PerfGunner.start("matmul_1024x1024")
    # ... run kernel ...
    PerfGunner.stop()

    snapshot = PerfGunner.snapshot()
    traces   = PerfGunner.drain_traces()
    PerfGunner.report()
"""
from __future__ import annotations

import json
import os

try:
    import tvm.ffi as ffi
    _HAS_FFI = True
except ImportError:
    _HAS_FFI = False

__all__ = ["PerfGunner", "perf_region"]


def _get(name: str, default=None):
    if not _HAS_FFI:
        return default
    return ffi.get_global_func(f"tl.Perf{name}", True)


class PerfGunner:
    """Singleton facade over the C++ PerfGunner in src/runtime/perfgunner.cc."""

    _enabled = False

    @classmethod
    def _ensure_enabled(cls):
        if not cls._enabled:
            f = _get("Enable")
            if f is not None:
                f(1)
            cls._enabled = True

    @classmethod
    def enable(cls):
        cls._ensure_enabled()

    @classmethod
    def disable(cls):
        cls._enabled = False
        f = _get("Enable")
        if f is not None:
            f(0)

    @classmethod
    def start(cls, label: str):
        f = _get("Start")
        if f is not None:
            cls._ensure_enabled()
            f(label)

    @classmethod
    def stop(cls):
        f = _get("Stop")
        if f is not None:
            f()

    @classmethod
    def snapshot(cls) -> dict:
        """Return current GPU VRAM / host RAM snapshot as a dict."""
        f = _get("Snapshot")
        if f is None:
            return {"gpu_alloc_bytes": 0, "gpu_free_bytes": 0,
                    "host_rss_bytes": 0, "host_vsize_bytes": 0}
        cls._ensure_enabled()
        return json.loads(f())

    @classmethod
    def drain_traces(cls) -> list[dict]:
        f = _get("DrainTraces")
        if f is None:
            return []
        cls._ensure_enabled()
        return json.loads(f())

    @classmethod
    def clear(cls):
        f = _get("Clear")
        if f is not None:
            f()

    @classmethod
    def region(cls, label: str):
        return _PerfRegion(label)

    @classmethod
    def report(cls) -> str:
        traces = cls.drain_traces()
        snap = cls.snapshot()
        lines = ["=" * 72,
                 "TileLang Performance Report",
                 "=" * 72]

        # Resource snapshot
        lines.append(f"\n  [Resource Snapshot]")
        lines.append(f"    GPU allocated: {snap['gpu_alloc_bytes'] / 1e6:.1f} MB")
        lines.append(f"    GPU free:      {snap['gpu_free_bytes'] / 1e6:.1f} MB")
        lines.append(f"    Host RSS:      {snap['host_rss_bytes'] / 1e6:.1f} MB")
        lines.append(f"    Host vsize:    {snap['host_vsize_bytes'] / 1e6:.1f} MB")

        # Trace table
        if traces:
            lines.append(f"\n  [Kernel Traces - {len(traces)} entries]")
            hdr = f"  {'name':<32} {'GPU us':>10} {'CPU us':>10} {'alloc MB':>10}"
            lines.append(hdr)
            lines.append("  " + "-" * (len(hdr) - 2))
            total_gpu = 0
            total_cpu = 0
            for t in traces:
                g = t.get("gpu_time_us", 0)
                c = t.get("cpu_wall_us", 0)
                alloc = t.get("gpu_alloc_bytes", 0) / 1e6
                lines.append(f"  {t['name'][:30]:<32} {g:>10.1f} {c:>10.1f} {alloc:>10.1f}")
                total_gpu += g
                total_cpu += c
            lines.append("  " + "-" * (len(hdr) - 2))
            lines.append(f"  {'TOTAL':<32} {total_gpu:>10.1f} {total_cpu:>10.1f}")
        else:
            lines.append("\n  [No traces recorded]")

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)


class _PerfRegion:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        PerfGunner.start(self.label)
        return self

    def __exit__(self, *exc):
        PerfGunner.stop()


perf_region = PerfGunner.region
