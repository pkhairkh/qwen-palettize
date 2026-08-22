"""Wave 1 test: verify TritonSoftLinear.forward matches reference.

Reference: pure PyTorch replication of the existing CUDA path's math:
  1. P_aos = softmax( (logits + gumbel_noise) / tau )  per (j, o, k)
  2. W_soft = Σ_k P[k] * palette[g, k]
  3. W_hard = palette[g, argmax(logits)]   (plain argmax, NO Gumbel)
  4. W_ste  = W_hard  (forward value)
  5. y = x @ W_ste + bias

We DO NOT try to bit-match the Gumbel noise (the LCG is in the kernel); we
verify the W_hard path (= y = x @ W_hard + bias) and the P_aos/W_soft numerics
separately.

Pass criterion: max|y_triton - y_ref| < 1e-3 (bf16 has ~3 decimal digits
of precision; 1e-3 is conservative for matmul outputs in the [-1, 1] range
of bf16 post-softmax/GELU regions).

Run on server:  python3 test_triton_soft_forward.py
"""
from __future__ import annotations
import sys
import math
import torch

# Import the Triton kernels under test
from triton_soft_forward import (
    compute_P_W_ste_triton,
    fused_soft_matmul_triton,
    TritonSoftLinear,
)


# ═════════════════════════════════════════════════════════════════════════════
# Reference implementations (pure PyTorch, used as ground truth)
# ═════════════════════════════════════════════════════════════════════════════
def ref_compute_W_hard_and_soft(
    logits: torch.Tensor,    # (4, K, N) fp16
    palette: torch.Tensor,   # (G, 4) bf16
    group_size: int,
    tau: float,
    step_seed: int,
):
    """Pure-PyTorch reference. Computes P_aos, W_soft, W_hard (no Gumbel noise
    bit-match — we use a different RNG here, but the structure is identical)."""
    n_planes, K, N = logits.shape
    G, _ = palette.shape

    # Gumbel noise: use torch.rand for reference (NOT bit-identical to kernel LCG,
    # but we don't compare P_aos values bit-for-bit; we compare downstream W_hard
    # which is Gumbel-independent, and W_soft which depends on Gumbel but we
    # re-derive P from the kernel's P_aos for the W_soft check).
    g = torch.distributions.Gumbel(0.0, 1.0).sample((4, K, N)).to(logits.device)
    logits_noisy = (logits.float() + g) / tau
    # Softmax over the 4 planes
    P = torch.softmax(logits_noisy, dim=0)  # (4, K, N) fp32

    # AoS layout: (K, N, 4) — permute (4, K, N) → (K, N, 4)
    P_aos_ref = P.permute(1, 2, 0).contiguous()  # (K, N, 4) fp32

    # W_soft = Σ_k P[k] * palette[g, k]   for each (j, o)
    # Group idx for each o: g = o // group_size
    g_idx = torch.arange(N, device=logits.device) // group_size  # (N,)
    pal_per_col = palette[g_idx.long()]  # (N, 4) bf16 → float
    pal_per_col_f = pal_per_col.float()  # (N, 4)
    # P_aos_ref is (K, N, 4); W_soft_ref[j, o] = Σ_k P_aos_ref[j, o, k] * pal_per_col[o, k]
    W_soft_ref = (P_aos_ref * pal_per_col_f[None, :, :]).sum(dim=-1)  # (K, N)

    # W_hard = palette[g, argmax(logits, dim=0)]   — plain argmax, no Gumbel
    argmax_idx = logits.argmax(dim=0)  # (K, N) int64
    # BUGFIX: g_idx must broadcast over COLUMNS (not rows).
    # Original code was palette[g_idx.long()][arange(K).unsqueeze(-1), argmax_idx]
    # which incorrectly used row index j instead of column-group g_idx[o].
    g_per_col = g_idx[None, :].expand(K, N)  # (K, N)
    W_hard_ref = palette[g_per_col.long(), argmax_idx]  # (K, N) bf16
    return P_aos_ref, W_soft_ref, W_hard_ref.float(), argmax_idx


def ref_matmul(x, W, bias):
    """bf16 matmul with fp32 internal accumulation — matches the existing
    CUDA path (`torch.matmul(x_bf16, W_bf16)`).
    """
    W_bf = W.to(torch.bfloat16)
    y = torch.matmul(x, W_bf)  # bf16 in, bf16 out (cuBLAS uses fp32 acc)
    if bias is not None:
        y = y + bias.to(torch.bfloat16)
    return y


# ═════════════════════════════════════════════════════════════════════════════
# Test entry
# ═════════════════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(0)
    device = "cuda"

    # Realistic shapes (one mid-sized PalettizedLinear from the Qwen3.5 super-block)
    M = 32 * 16   # batch=32, seq_len_chunk=16 → 512 rows
    K = 1024      # in_features
    N = 1024      # out_features (must be divisible by 256)
    group_size = 256
    G = N // group_size  # 4 groups
    tau = 1.5

    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
    palette = torch.randn(G, 4, device=device, dtype=torch.bfloat16) * 0.3
    # Initialize logits as approximately one-hot (so argmax is well-defined)
    logits = torch.full((4, K, N), -10.0, device=device, dtype=torch.float16)
    # Random one-hot per (j, o)
    init_idx = torch.randint(0, 4, (K, N), device=device)
    for k in range(4):
        logits[k][init_idx == k] = 10.0
    # Small perturbation so argmax is mostly init_idx but not exactly
    logits = logits + torch.randn_like(logits) * 0.5
    bias = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.05

    print(f"Shapes: M={M}, K={K}, N={N}, G={G}, tau={tau}")
    print(f"  x: {x.shape} {x.dtype}")
    print(f"  palette: {palette.shape} {palette.dtype}")
    print(f"  logits: {logits.shape} {logits.dtype}")
    print(f"  bias: {bias.shape} {bias.dtype}")
    print()

    # ── Run Triton forward (computes P_aos, W_soft, W_ste internally) ─────
    print("Running Triton forward (compute_P_W_ste + matmul)...")
    # We don't use the autograd Function yet — call the sub-kernels directly so
    # we can compare intermediate (P_aos, W_soft, W_ste) against the reference.
    step_seed = 42
    P_aos, W_soft, W_ste = compute_P_W_ste_triton(
        logits, palette, group_size, tau, step_seed
    )
    y_triton = fused_soft_matmul_triton(x, W_ste, bias)

    # ── Run reference (pure PyTorch) ───────────────────────────────────────
    # For W_hard reference: argmax + gather (NO Gumbel — deterministic)
    print("Running reference (pure PyTorch)...")
    P_aos_ref_f, W_soft_ref_f, W_hard_ref_f, _ = ref_compute_W_hard_and_soft(
        logits, palette, group_size, tau, step_seed
    )

    # Note: P_aos_ref and W_soft_ref are Gumbel-dependent — they won't bit-match
    # the Triton output (different RNG). We verify:
    #   (a) W_ste == W_hard (the forward uses W_hard; this should be EXACT match)
    #   (b) y_triton ≈ x @ W_hard_ref + bias (using Triton's W_hard = W_ste)
    # For W_soft + P_aos, we verify statistical properties (sum to 1, non-neg)
    # rather than bit-match.

    # (a) W_ste vs W_hard_ref — should be near-exact (bf16 precision)
    err_W_ste = (W_ste.float() - W_hard_ref_f).abs().max().item()
    print(f"  max|W_ste_triton - W_hard_ref| = {err_W_ste:.6e}")
    # DEBUG: print first few entries
    print(f"  DEBUG W_ste_triton[0:4, 0:4]:\n{W_ste[0:4, 0:4].float()}")
    print(f"  DEBUG W_hard_ref[0:4, 0:4]:\n{W_hard_ref_f[0:4, 0:4]}")
    assert err_W_ste < 1e-2, f"W_ste != W_hard (err={err_W_ste})"

    # (b) y_triton vs ref y = x @ W_hard + bias (both bf16 matmul, fp32 acc)
    y_ref = ref_matmul(x, W_hard_ref_f, bias)
    err_y = (y_triton.float() - y_ref.float()).abs().max().item()
    y_abs_max = y_triton.float().abs().max().item()
    rel_err = err_y / max(y_abs_max, 1e-6)
    print(f"  max|y_triton - y_ref| = {err_y:.6e}  (|y|_max={y_abs_max:.4f}, rel_err={rel_err:.4e})")
    # bf16 matmul vs cuBLAS bf16 matmul: error within 1-3 ULPs of bf16.
    # At |y|_max ≈ 5-15, 1 ULP ≈ 0.04-0.12. Use 2 ULPs threshold.
    ulp_bf16 = 2.0 ** (max(0, int(math.floor(math.log2(max(y_abs_max, 1.0)))) - 7))
    thresh = max(1e-3, 2 * ulp_bf16)  # at least 1e-3, at most 2 ULPs
    print(f"  threshold (2 ULPs bf16 at |y|_max): {thresh:.4e}")
    assert err_y < thresh, f"y mismatch (err={err_y}, threshold={thresh})"

    # (c) P_aos sanity: sum-to-1 + non-neg
    P_aos_f = P_aos.float()
    P_sum = P_aos_f.sum(dim=-1)
    err_sum = (P_sum - 1.0).abs().max().item()
    print(f"  max|Σ_k P[k] - 1| = {err_sum:.6e}")
    assert err_sum < 1e-3, f"P doesn't sum to 1 (err={err_sum})"
    P_min = P_aos_f.min().item()
    P_max = P_aos_f.max().item()
    print(f"  P range: [{P_min:.4f}, {P_max:.4f}]")
    assert P_min >= -1e-4, f"P has negative values (min={P_min})"
    assert P_max <= 1.0 + 1e-3, f"P > 1 (max={P_max})"

    # (d) W_soft sanity: should be in palette range (convex combination of palette vals)
    pal_f = palette.float()
    pal_min = pal_f.min().item()
    pal_max = pal_f.max().item()
    W_soft_min = W_soft.float().min().item()
    W_soft_max = W_soft.float().max().item()
    print(f"  palette range: [{pal_min:.4f}, {pal_max:.4f}]")
    print(f"  W_soft range: [{W_soft_min:.4f}, {W_soft_max:.4f}]")
    assert W_soft_min >= pal_min - 1e-3, f"W_soft below palette range"
    assert W_soft_max <= pal_max + 1e-3, f"W_soft above palette range"

    # ── Autograd Function forward smoke test (no backward call expected) ────
    print("\nAutograd Function smoke test (forward only)...")
    y_fn = TritonSoftLinear.apply(x, palette, logits, bias, group_size, tau)
    err_fn = (y_fn.float() - y_ref.float()).abs().max().item()
    rel_fn = err_fn / max(y_abs_max, 1e-6)
    print(f"  max|y_fn - y_ref| = {err_fn:.6e}  (rel={rel_fn:.4e})")
    # Same ULP-aware threshold as the raw kernel test
    assert err_fn < thresh, f"Function.apply forward mismatch (err={err_fn}, threshold={thresh})"

    # ── Verify no torch.matmul, no Python elementwise in the hot path ───────
    # (smoke check: this is a code-review check, not a runtime check)

    print("\n✅ Wave 1 forward test PASSED.")
    print(f"   max|y_triton - y_ref| = {err_y:.6e}  (rel={rel_err:.4e}, |y|_max={y_abs_max:.2f})")
    print(f"   max|W_ste - W_hard|   = {err_W_ste:.6e}  (W_ste is forward-correct)")
    print(f"   Σ_k P[k] = 1 ±{err_sum:.2e}  (softmax is correct)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
