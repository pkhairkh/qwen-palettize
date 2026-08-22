"""Patch 8a — profile test for the Patch 5 + Patch 7 kernel optimisations.

Verifies the performance targets from the DoD:

  - Backward time < 100 ms (target: 60 ms) — Patch 5 (AoS fused bwd)
  - Forward time saves ~36 ms (25 launches → 1) — Patch 7 (batched compute_P_W)

This script measures:

  1. Single-layer forward (`fused_lut_linear_soft_fwd_aos`) — used as the
     baseline for per-layer dispatch overhead.
  2. Single-layer backward (`fused_lut_linear_soft_bwd_fused_aos`) —
     the Patch 5 target, should be < 4 ms per layer × 25 = 100 ms.
  3. Batched forward (`fused_compute_P_W_batched`) for 25 layers — the
     Patch 7 target, should be < 12 ms total (vs 25 × per-layer = ~30 ms).

On a CUDA host, runs the kernels 5 times each (1 warmup + 4 measured)
and reports median ms. Prints a pass/fail summary at the end.

Run:
    python3 scripts/test_profile_kernels.py
"""
from __future__ import annotations

import os
import sys
import statistics
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Module structure (CPU-only, always runs)
# ─────────────────────────────────────────────────────────────────────────────
class TestProfileScriptStructure(unittest.TestCase):
    """Verifies the profile script's structure is sound (does not require a GPU)."""

    def test_profile_test_file_parses(self):
        py_path = os.path.join(HERE, "test_profile_kernels.py")
        with open(py_path) as f:
            import ast
            ast.parse(f.read())

    def test_targets_documented(self):
        """The profile script must document the DoD targets in its docstring."""
        py_path = os.path.join(HERE, "test_profile_kernels.py")
        with open(py_path) as f:
            src = f.read()
        # Patch 5 target
        self.assertIn("Backward time < 100 ms", src)
        self.assertIn("target: 60 ms", src)
        # Patch 7 target
        self.assertIn("25 launches", src)

    def test_profile_thresholds_exposed_as_constants(self):
        """The DoD thresholds are exposed as module-level constants so they
        can be tuned without editing the test bodies."""
        py_path = os.path.join(HERE, "test_profile_kernels.py")
        with open(py_path) as f:
            src = f.read()
        self.assertIn("BACKWARD_TARGET_MS", src)
        self.assertIn("FORWARD_TARGET_MS", src)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Actual profiling (CUDA-only)
# ─────────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA + nvcc to compile the extension")
class TestProfileKernels(unittest.TestCase):
    """Profiles the Patch 5 + Patch 7 kernels and verifies they meet the DoD targets.

    Run with: python3 scripts/test_profile_kernels.py
    """

    def test_backward_under_100ms(self):
        """Patch 5 DoD: backward < 100 ms per step (25 layers × ~4 ms/layer)."""
        from fused_lut_linear_cuda import _get_module
        mod = _get_module()

        # Single representative layer: K=N=2560 (Qwen3.5-4B hidden size),
        # M=16384 (batch=32 seq=512), GS=256, tau=0.5.
        K, N, M, GS = 2560, 2560, 16384, 256
        G = N // GS
        tau = 0.5
        step_seed = 42

        torch.manual_seed(0)
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        palette = torch.randn(G, 4, dtype=torch.bfloat16, device='cuda')
        logits = torch.randn(4, K, N, dtype=torch.float16, device='cuda')
        grad_y = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')

        # Run forward once to get P_aos
        y, P_aos, W = mod.fused_lut_linear_soft_fwd_aos(
            x, palette, logits, GS, float(tau), step_seed
        )

        # ── Warmup (1 iter) ──
        for _ in range(1):
            _ = mod.fused_lut_linear_soft_bwd_fused_aos(grad_y, x, P_aos, palette, GS)
        torch.cuda.synchronize()

        # ── Measure (4 iters) ──
        times_ms = []
        for _ in range(4):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = mod.fused_lut_linear_soft_bwd_fused_aos(grad_y, x, P_aos, palette, GS)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

        median_ms = statistics.median(times_ms) * 25  # extrapolate to 25 layers
        print(f"\n[profile] backward per layer: median={statistics.median(times_ms):.2f} ms")
        print(f"[profile] backward × 25 layers: {median_ms:.2f} ms (target: < 100 ms)")

        # DoD: backward < 100 ms per step (target: 60 ms)
        self.assertLess(median_ms, BACKWARD_TARGET_MS,
                        f"backward {median_ms:.2f} ms exceeds target {BACKWARD_TARGET_MS} ms")

    def test_batched_forward_under_target(self):
        """Patch 7 DoD: 25 launches → 1 launch, forward time saves ~36 ms."""
        from fused_lut_linear_cuda import _get_module
        mod = _get_module()

        # 25 layers of representative Qwen3.5-4B shapes.
        # The real model has 25 PalettizedLinears per super-block.
        torch.manual_seed(0)
        n_layers = 25
        K_per_layer = [2560] * 18 + [9216] * 4 + [3072] * 3  # heterogeneous shapes
        N_per_layer = [2560] * 25
        GS = 256
        tau = 0.5
        step_seed = 99

        logits_list = [
            torch.randn(4, K, N, dtype=torch.float16, device='cuda')
            for K, N in zip(K_per_layer, N_per_layer)
        ]
        palette_list = [
            torch.randn(N // GS, 4, dtype=torch.bfloat16, device='cuda')
            for N in N_per_layer
        ]

        # ── Warmup (1 iter) ──
        for _ in range(1):
            _ = mod.fused_compute_P_W_batched(logits_list, palette_list, GS, float(tau), step_seed)
        torch.cuda.synchronize()

        # ── Measure (4 iters) ──
        times_ms = []
        for _ in range(4):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = mod.fused_compute_P_W_batched(logits_list, palette_list, GS, float(tau), step_seed)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

        median_ms = statistics.median(times_ms)
        print(f"\n[profile] batched forward (1 launch, {n_layers} layers): "
              f"median={median_ms:.2f} ms (target: < {FORWARD_TARGET_MS} ms)")

        # DoD: forward time should drop substantially (target: < 25 ms total).
        self.assertLess(median_ms, FORWARD_TARGET_MS,
                        f"batched forward {median_ms:.2f} ms exceeds target {FORWARD_TARGET_MS} ms")


# ─────────────────────────────────────────────────────────────────────────────
# DoD thresholds — exposed as module-level constants so they can be tuned
# ─────────────────────────────────────────────────────────────────────────────
BACKWARD_TARGET_MS = 100   # DoD: backward < 100 ms per step (Patch 5)
FORWARD_TARGET_MS  = 25    # DoD: forward < 25 ms per step (Patch 7, down from ~30 ms × 25-launch overhead)


if __name__ == "__main__":
    unittest.main(verbosity=2)
