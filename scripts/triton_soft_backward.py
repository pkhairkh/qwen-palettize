"""Triton fused kernels for PalettizedLinear soft (Gumbel-Softmax + STE) BACKWARD.

WAVE 2 — fused_soft_backward:
  - `fused_soft_bwd_grad_x_kernel`:   grad_x = grad_y @ W_ste.T (Triton TC matmul)
  - `fused_soft_bwd_grad_W_kernel`:   grad_W = x.T @ grad_y      (Triton TC matmul)
  - `fused_soft_bwd_elementwise_kernel`: per-(j, o) kernel that loads P_aos,
    reconstructs W_soft on-the-fly, and computes:
      grad_logits[k, j, o] = grad_W[j, o] * P[k, j, o] * (palette[g, k] - W_soft[j, o])
      grad_palette[g, k] += grad_W[j, o] * P[k, j, o]   (atomic_add, fp32)

Math (matches existing CUDA `fused_lut_linear_soft_bwd_fused_aos_kernel`):
  STE forward: y = x @ W_ste where W_ste = W_hard - W_soft.detach() + W_soft
               → forward value = W_hard; backward gradient flows through W_soft
  Backward:
    grad_x      = grad_y @ W_ste.T          (= grad_y @ W_hard.T numerically)
    grad_W_soft = x.T @ grad_y               (= grad_W_ste under STE)
    grad_logits[k,j,o] = grad_W_soft[j,o] * P[k,j,o] * (palette[g,k] - W_soft[j,o])
      derivation: d(W_soft[j,o])/d(logits[k,j,o]) = P[k,j,o] * (palette[g,k] - W_soft[j,o])
                  (softmax derivative → diagonal P[k]*(1-P[k]) minus off-diagonal P[k']*P[k],
                   which simplifies to P[k] * (palette[g,k] - Σ P[k']*palette[g,k']))
    grad_palette[g,k] = Σ_{j,o in group g} grad_W_soft[j,o] * P[k,j,o]   (atomicAdd)

Three kernels split by parallelism pattern (matmul TC vs. elementwise). The
existing CUDA bwd kernel fuses (b) + (c) using shared-memory tiling, but the
CUDA implementation's SOFT_BWD_BM_CHUNK=128 / BK=16 / BN=16 leaves the matmul
under-tuned. Splitting into 3 Triton kernels lets `tl.dot` autotune to the
optimal TC config (BM=128/256, BN=128/256, BK=32/64) for the matmuls while the
elementwise kernel handles the per-(j, o) work.

Layouts (must match Wave 1 forward + existing CUDA path):
  grad_y:    (M, N) bf16  — input
  x:         (M, K) bf16  — saved in forward ctx
  W_ste:     (K, N) bf16  — saved in forward ctx (= W_hard numerically)
  P_aos:     (K, N, 4) fp16 — saved in forward ctx
  palette:   (G, 4) bf16  — saved in forward ctx
  Outputs:
    grad_x:      (M, K) bf16
    grad_logits: (4, K, N) fp16 — SoA (plane stride = K*N), matches optimizer
    grad_palette:(G, 4) fp32     — atomic accumulation
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 2a — grad_x = grad_y @ W_ste.T  (Triton TC matmul, transposed W)
# ═════════════════════════════════════════════════════════════════════════════
# W_ste is (K, N) contiguous → W_ste.T is (N, K) with stride (1, N).
# grad_y is (M, N) contiguous.
# grad_x[m, k] = Σ_n grad_y[m, n] * W_ste[k, n]   (we load W_ste row k = col k of W_ste.T)
@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 256, "BK": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_soft_bwd_grad_x_kernel(
    grad_y_ptr, W_ste_ptr,
    grad_x_ptr,
    M, N, K,
    stride_gym, stride_gyn,
    stride_wk, stride_wn,
    stride_gxm, stride_gxk,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """grad_x[m, k] = Σ_n grad_y[m, n] * W_ste[k, n]"""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    # L2-friendly swizzle: pid_m, pid_k for grad_x output (M, K)
    # Inner reduction dim is N — load grad_y tile (BM, BN) and W_ste tile (BN, BK)
    # But W_ste is (K, N) → we want W_ste[k_chunk, n_chunk] transposed access.
    # Layout: W_ste[k, n] at offset k*stride_wk + n*stride_wn
    # For grad_x[m, k] = Σ_n grad_y[m, n] * W_ste[k, n],
    # we tile (m, k) and reduce over n in chunks of BN.
    pid_m = pid // tl.cdiv(K, BK)
    pid_k = pid % tl.cdiv(K, BK)

    offs_m = pid_m * BM + tl.arange(0, BM)  # M direction
    offs_k = pid_k * BK + tl.arange(0, BK)  # K direction
    offs_n = tl.arange(0, BN)               # N direction (reduction)

    # grad_y tile (BM, BN) at (m_chunk, n_offset)
    gy_ptrs = grad_y_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
    # W_ste tile (BK, BN) at (k_chunk, n_offset) — transposed access (load row k, n)
    w_ptrs = W_ste_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BM, BK), dtype=tl.float32)
    for n_iter in range(0, tl.cdiv(N, BN)):
        mask_gy = (offs_m[:, None] < M) & (offs_n[None, :] + n_iter * BN < N)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] + n_iter * BN < N)
        gy_tile = tl.load(gy_ptrs, mask=mask_gy, other=0.0)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        # grad_x[m, k] += Σ_n gy[m, n] * w[k, n]
        # = tl.dot(gy_tile (BM, BN), w_tile.T (BN, BK))
        # tl.dot(A, B) does A @ B. We want gy @ w.T → use w_tile of shape (BN, BK)
        # by transposing: actually we have w_tile (BK, BN), so transpose to (BN, BK).
        # Triton's tl.dot supports transposing via tl.trans(w_tile).
        w_tile_t = tl.trans(w_tile)  # (BN, BK)
        acc += tl.dot(gy_tile, w_tile_t)

        gy_ptrs += BN * stride_gyn
        w_ptrs += BN * stride_wn

    gx_ptrs = grad_x_ptr + offs_m[:, None] * stride_gxm + offs_k[None, :] * stride_gxk
    mask_gx = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(gx_ptrs, acc.to(tl.bfloat16), mask=mask_gx)


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 2b — grad_W = x.T @ grad_y  (Triton TC matmul)
# ═════════════════════════════════════════════════════════════════════════════
# x is (M, K) bf16, grad_y is (M, N) bf16 → grad_W (K, N) bf16
# grad_W[k, n] = Σ_m x[m, k] * grad_y[m, n]
@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 256, "BK": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_soft_bwd_grad_W_kernel(
    x_ptr, grad_y_ptr,
    grad_W_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_gym, stride_gyn,
    stride_wk, stride_wn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """grad_W[k, n] = Σ_m x[m, k] * grad_y[m, n]  → output (K, N) bf16."""
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BM)  # output K direction
    grid_n = tl.cdiv(N, BN)
    pid_k = pid // grid_n
    pid_n = pid % grid_n

    offs_k = pid_k * BM + tl.arange(0, BM)  # K direction (output row)
    offs_n = pid_n * BN + tl.arange(0, BN)  # N direction (output col)
    offs_m = tl.arange(0, BK)               # M direction (reduction)

    # x tile (BK, BM) at (m_offset, k_chunk) — x[m, k]
    # We want x.T tile shape (BK, BM) so we can dot with grad_y (BK, BN)
    # x is (M, K) so x[m, k] at offset m*stride_xm + k*stride_xk
    # We load x as (BK, BM) by transposing: load (BM, BK) then tl.trans
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # grad_y tile (BK, BN) at (m_offset, n_chunk) — grad_y[m, n]
    gy_ptrs = grad_y_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for m_iter in range(0, tl.cdiv(M, BK)):
        mask_x = (offs_m[:, None] + m_iter * BK < M) & (offs_k[None, :] < K)
        mask_gy = (offs_m[:, None] + m_iter * BK < M) & (offs_n[None, :] < N)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)  # (BK, BM)
        gy_tile = tl.load(gy_ptrs, mask=mask_gy, other=0.0)  # (BK, BN)
        # grad_W[k, n] += Σ_m x[m, k] * grad_y[m, n]
        # = tl.dot(x_tile.T (BM, BK), gy_tile (BK, BN))
        # = tl.dot(tl.trans(x_tile), gy_tile)
        x_tile_t = tl.trans(x_tile)  # (BM, BK)
        acc += tl.dot(x_tile_t, gy_tile)

        x_ptrs += BK * stride_xm
        gy_ptrs += BK * stride_gym

    gW_ptrs = grad_W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    mask_gW = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    # Store grad_W in fp32 (used as input to elementwise kernel which expects fp32)
    tl.store(gW_ptrs, acc, mask=mask_gW)


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 2c — elementwise bwd: grad_logits + atomic grad_palette
# ═════════════════════════════════════════════════════════════════════════════
# For each (j, o): load P_aos[j, o, 0..3] (4 fp16, AoS coalesced) + palette[g, 0..3]
# Reconstruct W_soft = Σ P[k] * palette[g, k]  (fp32)
# Compute grad_logits[k, j, o] = grad_W[j, o] * P[k, j, o] * (palette[g, k] - W_soft)
# Atomic-add grad_palette[g, k] += grad_W[j, o] * P[k, j, o]   (fp32)
#
# NOTE: We use a FIXED config (no @triton.autotune) because Triton's autotuner
# runs the kernel multiple times during benchmarking, and atomic_add side
# effects accumulate across runs. The autotuner's `pre_hook` option could
# reset grad_palette to zero between runs, but on Triton 3.5 the pre_hook
# signature is brittle and varies across versions. A fixed config sidesteps
# the issue cleanly. The elementwise kernel is also not the hot path (the
# matmul kernels 2a/2b dominate), so autotune wouldn't help much anyway.
@triton.jit
def fused_soft_bwd_elementwise_kernel(
    grad_W_ptr,    # (K, N) fp32 — from Kernel 2b
    P_aos_ptr,     # (K, N, 4) fp16 AoS — from forward ctx
    palette_ptr,   # (G, 4) bf16
    grad_logits_ptr,  # (4, K, N) fp16 — OUTPUT SoA (plane stride = K*N)
    grad_palette_ptr, # (G, 4) fp32 — OUTPUT (atomic)
    K, N, G,
    group_size: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)  # K direction (j)
    offs_n = pid_n * BN + tl.arange(0, BN)  # N direction (o)

    mask_m = offs_m < K
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    j_grid = offs_m[:, None]
    o_grid = offs_n[None, :]
    idx_grid = j_grid * N + o_grid  # (BM, BN) — flat (K, N) index
    g_grid = o_grid // group_size   # (1, BN)

    plane_size = K * N

    # ── Load grad_W[j, o] (fp32) ───────────────────────────────────────────
    grad_W_val = tl.load(grad_W_ptr + idx_grid, mask=mask, other=0.0)  # (BM, BN) f32

    # ── Load P_aos[j, o, 0..3] (4 fp16, AoS — coalesced) ──────────────────
    paos_base = idx_grid * 4  # (BM, BN)
    p0 = tl.load(P_aos_ptr + paos_base + 0, mask=mask, other=0.0).to(tl.float32)
    p1 = tl.load(P_aos_ptr + paos_base + 1, mask=mask, other=0.0).to(tl.float32)
    p2 = tl.load(P_aos_ptr + paos_base + 2, mask=mask, other=0.0).to(tl.float32)
    p3 = tl.load(P_aos_ptr + paos_base + 3, mask=mask, other=0.0).to(tl.float32)

    # ── Load palette[g, 0..3] (4 bf16) ─────────────────────────────────────
    pal_base = g_grid * 4  # (1, BN)
    c0 = tl.load(palette_ptr + pal_base + 0, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c1 = tl.load(palette_ptr + pal_base + 1, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c2 = tl.load(palette_ptr + pal_base + 2, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c3 = tl.load(palette_ptr + pal_base + 3, mask=mask_n[None, :], other=0.0).to(tl.float32)

    # ── Reconstruct W_soft = Σ P[k] * palette[g, k] ───────────────────────
    W_soft = p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3

    # ── grad_logits[k, j, o] = grad_W * P[k] * (palette[g, k] - W_soft) ─────
    # Output layout: (4, K, N) SoA — plane k at offset k*plane_size + j*N + o
    gl_base = idx_grid  # offset within a plane
    gl0 = (grad_W_val * p0 * (c0 - W_soft)).to(tl.float16)
    gl1 = (grad_W_val * p1 * (c1 - W_soft)).to(tl.float16)
    gl2 = (grad_W_val * p2 * (c2 - W_soft)).to(tl.float16)
    gl3 = (grad_W_val * p3 * (c3 - W_soft)).to(tl.float16)
    tl.store(grad_logits_ptr + 0 * plane_size + gl_base, gl0, mask=mask)
    tl.store(grad_logits_ptr + 1 * plane_size + gl_base, gl1, mask=mask)
    tl.store(grad_logits_ptr + 2 * plane_size + gl_base, gl2, mask=mask)
    tl.store(grad_logits_ptr + 3 * plane_size + gl_base, gl3, mask=mask)

    # ── grad_palette[g, k] += grad_W * P[k]  (atomic_add, fp32) ─────────────
    # pal_base is (1, BN) — same shape as g_grid; each (j, o) maps to one g.
    # We need to atomic_add at offset pal_base[o] + k for k = 0..3.
    # Since pal_base is per-column, broadcast over rows: pal_base[:, None] doesn't
    # work directly because we want different o → different g. Use the column
    # index directly.
    pal_off = g_grid * 4  # (1, BN) — group offset for each column
    # Broadcast pal_off to (BM, BN) by replicating across rows:
    # tl.atomic_add expects pointer + offset tensors of matching shape.
    pal_off_b = pal_off + tl.zeros_like(j_grid)  # (BM, BN) broadcast
    tl.atomic_add(grad_palette_ptr + pal_off_b + 0, (grad_W_val * p0), mask=mask)
    tl.atomic_add(grad_palette_ptr + pal_off_b + 1, (grad_W_val * p1), mask=mask)
    tl.atomic_add(grad_palette_ptr + pal_off_b + 2, (grad_W_val * p2), mask=mask)
    tl.atomic_add(grad_palette_ptr + pal_off_b + 3, (grad_W_val * p3), mask=mask)


# ═════════════════════════════════════════════════════════════════════════════
# Python launchers
# ═════════════════════════════════════════════════════════════════════════════
def fused_soft_bwd_grad_x_triton(
    grad_y: torch.Tensor,    # (M, N) bf16
    W_ste: torch.Tensor,     # (K, N) bf16
) -> torch.Tensor:
    """grad_x = grad_y @ W_ste.T  → (M, K) bf16."""
    assert grad_y.dtype == torch.bfloat16
    assert W_ste.dtype == torch.bfloat16
    M, N = grad_y.shape
    K, N2 = W_ste.shape
    assert N == N2
    grad_y = grad_y.contiguous()
    W_ste = W_ste.contiguous()
    grad_x = torch.empty((M, K), dtype=torch.bfloat16, device=grad_y.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(K, meta["BK"]),)
    fused_soft_bwd_grad_x_kernel[grid](
        grad_y, W_ste, grad_x,
        M, N, K,
        grad_y.stride(0), grad_y.stride(1),
        W_ste.stride(0), W_ste.stride(1),
        grad_x.stride(0), grad_x.stride(1),
    )
    return grad_x


def fused_soft_bwd_grad_W_triton(
    x: torch.Tensor,        # (M, K) bf16
    grad_y: torch.Tensor,    # (M, N) bf16
) -> torch.Tensor:
    """grad_W = x.T @ grad_y → (K, N) fp32 (kept in fp32 for elementwise kernel)."""
    assert x.dtype == torch.bfloat16
    assert grad_y.dtype == torch.bfloat16
    M, K = x.shape
    M2, N = grad_y.shape
    assert M == M2
    x = x.contiguous()
    grad_y = grad_y.contiguous()
    grad_W = torch.empty((K, N), dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(K, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    fused_soft_bwd_grad_W_kernel[grid](
        x, grad_y, grad_W,
        M, N, K,
        x.stride(0), x.stride(1),
        grad_y.stride(0), grad_y.stride(1),
        grad_W.stride(0), grad_W.stride(1),
    )
    return grad_W


def fused_soft_bwd_elementwise_triton(
    grad_W: torch.Tensor,    # (K, N) fp32
    P_aos: torch.Tensor,     # (K, N, 4) fp16 AoS
    palette: torch.Tensor,   # (G, 4) bf16
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (grad_logits (4,K,N) fp16 SoA, grad_palette (G,4) fp32)."""
    assert grad_W.dtype == torch.float32
    assert P_aos.dtype == torch.float16
    assert palette.dtype == torch.bfloat16
    K, N, _ = P_aos.shape
    G, _ = palette.shape
    assert N // group_size == G
    grad_W = grad_W.contiguous()
    P_aos = P_aos.contiguous()
    palette = palette.contiguous()
    grad_logits = torch.empty((4, K, N), dtype=torch.float16, device=P_aos.device)
    grad_palette = torch.zeros((G, 4), dtype=torch.float32, device=palette.device)
    # Fixed config (BM=32, BN=32, 8 warps) — see kernel note on autotune
    # side-effect issue with atomic_add.
    BM, BN = 32, 32
    grid = (triton.cdiv(K, BM), triton.cdiv(N, BN))
    fused_soft_bwd_elementwise_kernel[grid](
        grad_W, P_aos, palette,
        grad_logits, grad_palette,
        K, N, G,
        group_size=group_size,
        BM=BM, BN=BN,
        num_warps=8,
        num_stages=1,
    )
    return grad_logits, grad_palette
