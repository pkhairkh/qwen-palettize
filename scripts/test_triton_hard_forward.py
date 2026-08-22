"""Wave 3 test: verify TritonHardLinear.forward matches reference (eval mode).

Reference: pure PyTorch gather + matmul (matches existing CUDA path's math):
  1. W = palette[g, indices[j, o]]  where g = o // group_size
  2. y = x @ W + bias  (bf16 matmul with fp32 acc)

Pass criterion: max|y_triton - y_ref| within bf16 ULPs (threshold ~2 ULPs).
"""
from __future__ import annotations
import sys
import math
import torch

from triton_hard_forward import (
    compute_hard_W_triton,
    fused_hard_matmul_triton,
    TritonHardLinear,
)


def ref_forward(x_bf, palette_bf, indices_i8, bias_bf, group_size):
    """Pure-PyTorch reference: gather + matmul."""
    K, N = indices_i8.shape
    G, _ = palette_bf.shape
    # W[j, o] = palette[g_idx[o], indices[j, o]]
    g_idx = torch.arange(N, device=palette_bf.device) // group_size  # (N,)
    g_per_col = g_idx[None, :].expand(K, N)  # (K, N)
    W_ref = palette_bf[g_per_col.long(), indices_i8.long()].to(torch.bfloat16)
    # y = x @ W + bias (bf16 matmul with fp32 acc — matches torch.matmul)
    y = torch.matmul(x_bf, W_ref)
    if bias_bf is not None:
        y = y + bias_bf.to(torch.bfloat16)
    return y, W_ref


def main():
    torch.manual_seed(0)
    device = "cuda"

    # Mid-sized PalettizedLinear eval
    M = 32 * 16   # 512
    K = 1024
    N = 1024
    group_size = 256
    G = N // group_size

    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
    palette = torch.randn(G, 4, device=device, dtype=torch.bfloat16) * 0.3
    indices = torch.randint(0, 4, (K, N), device=device).to(torch.int8)
    bias = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.05

    print(f"Shapes: M={M}, K={K}, N={N}, G={G}")

    # ── Run Triton hard forward ──────────────────────────────────────────
    print("\nRunning Triton hard forward (compute_hard_W + matmul)...")
    W_triton = compute_hard_W_triton(palette, indices, group_size)
    y_triton = fused_hard_matmul_triton(x, W_triton, bias)
    print(f"  y: {y_triton.shape} {y_triton.dtype}")

    # ── Run reference ────────────────────────────────────────────────────
    print("\nRunning reference (pure PyTorch)...")
    y_ref, W_ref = ref_forward(x, palette, indices, bias, group_size)
    print(f"  y_ref: {y_ref.shape} {y_ref.dtype}")

    # ── Compare W ────────────────────────────────────────────────────────
    err_W = (W_triton.float() - W_ref.float()).abs().max().item()
    print(f"\nmax|W_triton - W_ref| = {err_W:.6e}")
    assert err_W < 1e-3, f"W mismatch (err={err_W})"

    # ── Compare y ────────────────────────────────────────────────────────
    err_y = (y_triton.float() - y_ref.float()).abs().max().item()
    y_abs_max = y_triton.float().abs().max().item()
    rel = err_y / max(y_abs_max, 1e-6)
    print(f"max|y_triton - y_ref| = {err_y:.6e}  (|y|_max={y_abs_max:.4f}, rel={rel:.4e})")
    ulp_bf16 = 2.0 ** (max(0, int(math.floor(math.log2(max(y_abs_max, 1.0)))) - 7))
    thresh = max(1e-3, 2 * ulp_bf16)
    print(f"threshold (2 ULPs bf16 at |y|_max): {thresh:.4e}")
    assert err_y < thresh, f"y mismatch (err={err_y}, threshold={thresh})"

    # ── Autograd Function smoke test ──────────────────────────────────────
    print("\nAutograd Function smoke test...")
    y_fn = TritonHardLinear.apply(x, palette, indices, bias, group_size)
    err_fn = (y_fn.float() - y_ref.float()).abs().max().item()
    print(f"  max|y_fn - y_ref| = {err_fn:.6e}")
    assert err_fn < thresh, f"Function.apply mismatch (err={err_fn}, threshold={thresh})"

    print("\n✅ Wave 3 hard forward test PASSED.")
    print(f"   max|W_triton - W_ref| = {err_W:.6e}  (exact match)")
    print(f"   max|y_triton - y_ref| = {err_y:.6e}  (rel={rel:.4e}, |y|_max={y_abs_max:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
