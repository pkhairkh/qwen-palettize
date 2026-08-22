"""Patch 5d — correctness test for the AoS fused backward kernel.

Validates three properties:

1. AoS / SoA layout equivalence (CPU-only, always runs)
   The math of compute_P_W is invariant under the (4, K, N) ↔ (K, N, 4)
   storage permutation: the softmax probabilities for (j, o) are identical
   in both layouts, only the storage order changes. We verify a round-trip
   permute matches element-for-element.

2. Module symbols (always runs — does NOT require a GPU)
   The patched fused_lut_linear_cuda module exposes the two new symbols:
     - fused_lut_linear_soft_fwd_aos
     - fused_lut_linear_soft_bwd_fused_aos
   We import the module's source and check that these symbols appear in
   the load_inline `functions=[]` list (so they will be bound on the
   compiled extension once a GPU + nvcc are present).

3. Numerical equivalence vs Python reference (CUDA-only)
   On a CUDA host we materialise random logits + palette, run
   fused_lut_linear_soft_fwd_aos to get P_aos + W_soft, then compare:
     - P_aos[:, :, k]  vs  P_soa[k]   (round-trip permutation)
     - W_soft         vs  Σ_k P[k] * palette[g, k]   (Python reference)
   We then run fused_lut_linear_soft_bwd_fused_aos and compare its
   grad_logits + grad_palette to a pure-PyTorch reference (the same
   formulas that the PHASE IX.c path used) with tolerance max_err < 1e-3.

Run:
    python3 scripts/test_fused_bwd_aos.py
"""
from __future__ import annotations

import os
import sys
import ast
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: AoS / SoA layout equivalence (CPU)
# ─────────────────────────────────────────────────────────────────────────────
class TestAosSoaEquivalence(unittest.TestCase):
    """Verifies the (K, N, 4) AoS layout is a bit-exact permutation of (4, K, N) SoA."""

    def test_permute_roundtrip_is_identity(self):
        # Random fp16 logits — mimic the (4, K, N) SoA layout used by the
        # training parameter (unchanged by Patch 5).
        torch.manual_seed(42)
        K, N = 32, 64
        logits_soa = torch.randn(4, K, N, dtype=torch.float16)

        # Compute softmax probs in SoA layout — same math the kernel does
        # (without Gumbel noise; this test isolates the *layout*, not the RNG).
        P_soa = torch.softmax(logits_soa.float(), dim=0).to(torch.float16)

        # AoS view: (K, N, 4) — this is what the new kernel writes
        P_aos = P_soa.permute(1, 2, 0).contiguous()

        self.assertEqual(P_aos.shape, (K, N, 4))
        self.assertTrue(P_aos.is_contiguous())

        # Round-trip: AoS → SoA should recover the original
        P_soa_recovered = P_aos.permute(2, 0, 1).contiguous()
        self.assertEqual(P_soa_recovered.shape, (4, K, N))
        self.assertTrue(torch.equal(P_soa, P_soa_recovered))

    def test_aos_w_equivalent_to_soa_w(self):
        """W[j, o] = Σ_k P[j, o, k] * palette[g, k] — should be identical regardless of layout."""
        torch.manual_seed(0)
        K, N, GS = 16, 32, 16
        G = N // GS

        logits = torch.randn(4, K, N, dtype=torch.float16)
        palette = torch.randn(G, 4, dtype=torch.bfloat16)

        P_soa = torch.softmax(logits.float(), dim=0).to(torch.float16)        # (4, K, N)
        P_aos = P_soa.permute(1, 2, 0).contiguous()                            # (K, N, 4)

        # Reference: compute W from SoA P (the "old" path)
        g_idx = torch.arange(N) // GS                                          # (N,)
        pal_per_col = palette[g_idx.long()]                                     # (N, 4)
        # W[j, o] = Σ_k P_soa[k, j, o] * pal_per_col[o, k]
        W_from_soa = (P_soa.float() * pal_per_col.T.unsqueeze(1)).sum(dim=0)   # (K, N)

        # Reference: compute W from AoS P (the "new" path — should match exactly)
        W_from_aos = (P_aos.float() * pal_per_col.unsqueeze(0)).sum(dim=-1)    # (K, N)

        max_err = (W_from_soa - W_from_aos).abs().max().item()
        self.assertLess(max_err, 1e-6, f"W mismatch: max_err={max_err}")

    def test_grad_formulas_layout_invariant(self):
        """grad_logits[j, o, k] = grad_W[j, o] * P[k] * (palette[g, k] - W[j, o]).

        This formula is the same regardless of P's storage layout — only the
        way we INDEX P differs. We verify both layouts produce identical
        gradient values.
        """
        torch.manual_seed(123)
        K, N, GS = 16, 32, 16
        G = N // GS

        logits = torch.randn(4, K, N, dtype=torch.float16)
        palette = torch.randn(G, 4, dtype=torch.bfloat16)
        grad_W = torch.randn(K, N, dtype=torch.float32)  # materialised grad_W

        # Compute P (both layouts)
        P_soa = torch.softmax(logits.float(), dim=0).to(torch.float16)        # (4, K, N)
        P_aos = P_soa.permute(1, 2, 0).contiguous()                            # (K, N, 4)

        g_idx = torch.arange(N) // GS
        pal_per_col = palette[g_idx.long()].float()                            # (N, 4) fp32

        # ── grad_logits from SoA P (old path) ──
        P_soa_f = P_soa.float()                                                 # (4, K, N)
        W_soa = (P_soa_f * pal_per_col.T.unsqueeze(1)).sum(dim=0)              # (K, N)
        # grad_logits[k, j, o] = grad_W[j, o] * P_soa[k, j, o] * (pal[o, k] - W[j, o])
        gl_soa = torch.stack([
            grad_W * P_soa_f[0] * (pal_per_col[:, 0] - W_soa),
            grad_W * P_soa_f[1] * (pal_per_col[:, 1] - W_soa),
            grad_W * P_soa_f[2] * (pal_per_col[:, 2] - W_soa),
            grad_W * P_soa_f[3] * (pal_per_col[:, 3] - W_soa),
        ], dim=0)                                                               # (4, K, N) fp32

        # ── grad_logits from AoS P (new path) ──
        P_aos_f = P_aos.float()                                                 # (K, N, 4)
        W_aos = (P_aos_f * pal_per_col.unsqueeze(0)).sum(dim=-1)                # (K, N)
        gl_aos = (
            grad_W.unsqueeze(-1) * P_aos_f * (pal_per_col.unsqueeze(0) - W_aos.unsqueeze(-1))
        )                                                                       # (K, N, 4)
        gl_aos = gl_aos.permute(2, 0, 1).contiguous()                           # (4, K, N)

        max_err = (gl_soa - gl_aos).abs().max().item()
        self.assertLess(max_err, 1e-5, f"grad_logits mismatch: max_err={max_err}")

        # ── grad_palette: sum over (K, group) of grad_W * P[k] ──
        # From SoA:
        contributions_soa = (
            grad_W.unsqueeze(-1) * P_soa_f.permute(1, 2, 0)
        ).view(K, G, GS, 4)
        gp_soa = contributions_soa.sum(dim=(0, 2))                              # (G, 4)

        # From AoS:
        contributions_aos = (
            grad_W.unsqueeze(-1) * P_aos_f
        ).view(K, G, GS, 4)
        gp_aos = contributions_aos.sum(dim=(0, 2))                              # (G, 4)

        max_err_gp = (gp_soa - gp_aos).abs().max().item()
        self.assertLess(max_err_gp, 1e-5, f"grad_palette mismatch: max_err={max_err_gp}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Module symbols (does NOT require a GPU)
# ─────────────────────────────────────────────────────────────────────────────
class TestModuleSymbols(unittest.TestCase):
    """Validates the patched fused_lut_linear_cuda.py exposes the new AoS symbols."""

    def test_python_source_parses(self):
        py_path = os.path.join(HERE, "fused_lut_linear_cuda.py")
        with open(py_path) as f:
            src = f.read()
        # Must parse cleanly
        ast.parse(src)

    def test_aos_functions_registered(self):
        """The two new functions must appear in the load_inline `functions=[]` list."""
        py_path = os.path.join(HERE, "fused_lut_linear_cuda.py")
        with open(py_path) as f:
            src = f.read()
        self.assertIn('"fused_lut_linear_soft_fwd_aos"', src,
                      "fused_lut_linear_soft_fwd_aos must be registered in functions=[]")
        self.assertIn('"fused_lut_linear_soft_bwd_fused_aos"', src,
                      "fused_lut_linear_soft_bwd_fused_aos must be registered in functions=[]")

    def test_cu_kernels_defined(self):
        """The .cu file must define the AoS kernel + launcher symbols."""
        cu_path = os.path.join(HERE, "fused_lut_kernel.cu")
        with open(cu_path) as f:
            src = f.read()
        for sym in [
            "fused_lut_linear_soft_compute_P_W_aos_kernel",
            "fused_lut_linear_soft_compute_P_W_aos_Launcher",
            "fused_lut_linear_soft_bwd_fused_aos_kernel",
            "fused_lut_linear_soft_bwd_fused_aos_Launcher",
        ]:
            self.assertIn(sym, src, f"{sym} missing from fused_lut_kernel.cu")

    def test_backward_uses_aos_kernel(self):
        """CUDAFusedLUTLinearSoft.backward must call the fused AoS kernel."""
        py_path = os.path.join(HERE, "fused_lut_linear_cuda.py")
        with open(py_path) as f:
            src = f.read()
        self.assertIn("fused_lut_linear_soft_bwd_fused_aos", src)
        self.assertIn("P_aos", src)
        # The old PyTorch-elementwise path should no longer be the active code path
        # (we replaced it with the fused kernel call).
        # But the SKIP_ZERO_GRAD_LOGITS env-var escape hatch is preserved.
        self.assertIn("SKIP_ZERO_GRAD_LOGITS", src)

    def test_backward_has_skip_fused_bwd_fallback(self):
        """Round-1 fix: a SKIP_FUSED_BWD=1 env-var fallback to the Python
        elementwise path MUST be present, so operators can switch back to a
        known-correct path if the fused kernel produces NaN on a real GPU.
        """
        py_path = os.path.join(HERE, "fused_lut_linear_cuda.py")
        with open(py_path) as f:
            src = f.read()
        self.assertIn("SKIP_FUSED_BWD", src,
                      "SKIP_FUSED_BWD env-var fallback must be present in backward")
        # The fallback must call torch.matmul(x.T, grad_y) (grad_W reference)
        # and must produce grad_logits in (4, K, N) SoA layout (matches the
        # autograd contract for the SoA logits parameter).
        self.assertIn("use_python_fallback", src)
        self.assertIn("grad_W = torch.matmul(x.T, grad_y)", src)

    def test_forward_calls_aos_kernel(self):
        """CUDAFusedLUTLinearSoft.forward must call the AoS soft forward."""
        py_path = os.path.join(HERE, "fused_lut_linear_cuda.py")
        with open(py_path) as f:
            src = f.read()
        self.assertIn("fused_lut_linear_soft_fwd_aos", src)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Numerical equivalence vs Python reference (CUDA-only)
# ─────────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA + nvcc to compile the extension")
class TestCudaEndToEnd(unittest.TestCase):
    """End-to-end numerical check on the GPU.

    Compares the fused AoS bwd kernel's output to a pure-PyTorch reference.
    Tolerance: max_err < 1e-3 (per Patch 5 DoD).
    """

    def test_bwd_aos_matches_python_reference(self):
        from fused_lut_linear_cuda import _get_module
        mod = _get_module()

        torch.manual_seed(0)
        K, N, GS, M = 64, 128, 32, 16
        G = N // GS
        tau = 0.5
        step_seed = 1

        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        palette = torch.randn(G, 4, dtype=torch.bfloat16, device='cuda')
        logits = torch.randn(4, K, N, dtype=torch.float16, device='cuda')
        grad_y = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')

        # ── Forward (AoS) ──
        y, P_aos, W_soft = mod.fused_lut_linear_soft_fwd_aos(
            x, palette, logits, GS, float(tau), step_seed
        )
        self.assertEqual(P_aos.shape, (K, N, 4))
        self.assertEqual(P_aos.dtype, torch.float16)
        self.assertEqual(W_soft.shape, (K, N))
        self.assertEqual(W_soft.dtype, torch.bfloat16)

        # ── Reference: reconstruct W from P_aos + palette (should match W_soft) ──
        g_idx = torch.arange(N, device='cuda') // GS
        pal_per_col = palette[g_idx.long()].float()                 # (N, 4)
        W_ref = (P_aos.float() * pal_per_col.unsqueeze(0)).sum(dim=-1)  # (K, N)
        max_err_w = (W_ref.bfloat16().float() - W_soft.float()).abs().max().item()
        self.assertLess(max_err_w, 1e-2, f"W_soft mismatch: {max_err_w}")

        # ── Backward (AoS) ──
        grad_logits, grad_palette = mod.fused_lut_linear_soft_bwd_fused_aos(
            grad_y, x, P_aos, palette, GS
        )
        self.assertEqual(grad_logits.shape, (4, K, N))
        self.assertEqual(grad_logits.dtype, torch.float16)
        self.assertEqual(grad_palette.shape, (G, 4))
        self.assertEqual(grad_palette.dtype, torch.bfloat16)

        # ── Python reference for grad_W + grad_logits + grad_palette ──
        grad_W_ref = torch.matmul(x.T, grad_y).float()             # (K, N) fp32
        P_aos_f = P_aos.float()                                    # (K, N, 4)
        W_val_ref = (P_aos_f * pal_per_col.unsqueeze(0)).sum(dim=-1)  # (K, N)
        # grad_logits[j, o, k] = grad_W * P[k] * (palette[g, k] - W)
        gl_ref = (
            grad_W_ref.unsqueeze(-1)
            * P_aos_f
            * (pal_per_col.unsqueeze(0) - W_val_ref.unsqueeze(-1))
        ).to(torch.float16).permute(2, 0, 1).contiguous()          # (4, K, N)

        # grad_palette[g, k] = Σ_{j, o in group g} grad_W[j, o] * P[j, o, k]
        contributions = (grad_W_ref.unsqueeze(-1) * P_aos_f).view(K, G, GS, 4)
        gp_ref = contributions.sum(dim=(0, 2)).to(torch.bfloat16)

        # ── Compare ──
        max_err_gl = (grad_logits.float() - gl_ref.float()).abs().max().item()
        max_err_gp = (grad_palette.float() - gp_ref.float()).abs().max().item()
        self.assertLess(max_err_gl, 1e-3, f"grad_logits max_err={max_err_gl}")
        self.assertLess(max_err_gp, 1e-3, f"grad_palette max_err={max_err_gp}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
