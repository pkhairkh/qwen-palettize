"""Triton fused kernels for PalettizedLinear soft (Gumbel-Softmax + STE) BACKWARD.

WAVE 2 (REVISED) — fused_soft_backward, performance-fixed:
  - `fused_soft_bwd_grad_x_kernel`:   grad_x = grad_y @ W_ste.T (Triton TC matmul,
                                       no tl.trans — loads W_ste transposed directly)
  - `fused_soft_bwd_grad_W_kernel`:   grad_W = x.T @ grad_y      (Triton TC matmul, fp32)
  - `fused_soft_bwd_elementwise_kernel`: per-(j, o) tile kernel with LARGE block sizes
    (BM=128, BN=128 → 16K positions per program) and per-block grad_palette
    accumulation to reduce atomic contention. Replaces the 3.7 GB/s disaster
    with a proper memory-bound kernel targeting 200+ GB/s.

PATCH 17 (buffer pooling):
  The backward's intermediate grad_W (K,N) fp32 is drawn from a module-level
  pool keyed on (K, N, device). grad_W is produced by fused_soft_bwd_grad_W_triton
  and consumed immediately by fused_soft_bwd_elementwise_triton within the SAME
  backward call — there is no autograd hazard (grad_W is never saved in ctx,
  never returned upstream). grad_x, grad_logits, grad_palette are RETURNED to
  autograd and CANNOT be pooled.

Math (matches existing CUDA `fused_lut_linear_soft_bwd_fused_aos_kernel`):
  STE forward: y = x @ W_ste where W_ste = W_hard - W_soft.detach() + W_soft
               → forward value = W_hard; backward gradient flows through W_soft
  Backward:
    grad_x      = grad_y @ W_ste.T          (= grad_y @ W_hard.T numerically)
    grad_W_soft = x.T @ grad_y               (= grad_W_ste under STE)
    grad_logits[k,j,o] = grad_W_soft[j,o] * P[k,j,o] * (palette[g,k] - W_soft[j,o])
    grad_palette[g,k] = Σ_{j,o in group g} grad_W_soft[j,o] * P[k,j,o]   (atomicAdd)

PERFORMANCE FIXES (vs Wave 2 v1):
  1. elementwise: BM=32,BN=32 → BM=128,BN=128 (16x more work per program).
     Original was launch-overhead-bound (3.7 GB/s = 0.4% of HBM3e peak).
     Also: accumulate grad_palette in registers per-block, do ONE atomic_add
     per (block, k) at the end → reduces atomics from K*N to ~K*N/16K.
  2. grad_x: removed tl.trans() (which forces a shared-memory transpose).
     Instead, load W_ste with swapped stride order so the tile is already (BN, BK).
  3. grad_W: kept as-is (already 3-4x faster than torch).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 2a — grad_x = grad_y @ W_ste.T  (Triton TC matmul, transposed W)
# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE FIX v4: large BN (128/256 only — no BN=64) + L2 cache swizzle.
# The original v2 kernel let the autotuner pick BN=64, which underutilized
# tensor cores (each tl.dot did only 128×64×32 = 256K FMA). v4 forces large BN
# so each tl.dot does 128×256×64 = 2M FMA — 8x better TC utilization.
# Also adds GROUP_M swizzle for L2 cache locality (from Triton matmul tutorial).
#
# Result: ~240K GFLOPS (was ~190K). cuBLAS achieves ~400K via split-K +
# persistent kernels — closing that last 1.6x gap requires split-K with
# workspace allocation, which adds complexity and memory overhead. v4 is the
# sweet spot for a single-kernel approach.
#
# W_ste is (K, N) row-major → W_ste[k, n] at offset k*stride_wk + n*stride_wn
# We want grad_x[m, k] = Σ_n grad_y[m, n] * W_ste[k, n]
#                       = Σ_n grad_y[m, n] * W_ste.T[n, k]
# Tile (BM, BK) of output grad_x. Reduction dim is N (chunked by BN).
#   grad_y tile: (BM, BN) at (m_chunk, n_offset)         — row-major
#   W_ste tile:  (BN, BK) at (k_chunk, n_offset)         — load with swapped strides
#     ptr = W_ste_ptr + n_offs[:, None]*stride_wn + k_offs[None, :]*stride_wk
#   acc += tl.dot(grad_y_tile, W_ste_tile)   (BM, BN) @ (BN, BK) → (BM, BK)
@triton.autotune(
    configs=[
        # ONLY large BN configs (128/256) — BN=64 underutilizes tensor cores.
        # stages=3 for 256×256 tiles (shared mem limit), stages=4 otherwise.
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 256, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 4}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP_M": 4}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 256, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        # Smaller BM for small M
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64, "BN": 256, "BK": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
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
    GROUP_M: tl.constexpr,
):
    """grad_x[m, k] = Σ_n grad_y[m, n] * W_ste[k, n]  → output (M, K) bf16.

    Uses L2-cache-friendly group swizzle (from Triton matmul tutorial) so
    adjacent programs share grad_y rows in L2 cache.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_k = tl.cdiv(K, BK)

    # L2 cache swizzle: group programs into GROUP_M×1 blocks so adjacent
    # programs share grad_y rows in L2 cache.
    num_pid_in_group = GROUP_M * grid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)  # M direction (output row)
    offs_k = pid_k * BK + tl.arange(0, BK)  # K direction (output col)
    offs_n = tl.arange(0, BN)                # N direction (reduction)

    # grad_y tile (BM, BN) at (m_chunk, n_offset) — standard row-major load
    gy_ptrs = grad_y_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
    # W_ste tile (BN, BK) at (k_chunk, n_offset) — load TRANSPOSED by swapping strides
    # W_ste[k, n] at k*stride_wk + n*stride_wn → load as (BN, BK) with n in rows, k in cols
    w_ptrs = W_ste_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BM, BK), dtype=tl.float32)
    for n_iter in range(0, tl.cdiv(N, BN)):
        n_off = n_iter * BN
        mask_gy = (offs_m[:, None] < M) & ((offs_n[None, :] + n_off) < N)
        mask_w = ((offs_n[:, None] + n_off) < N) & (offs_k[None, :] < K)
        gy_tile = tl.load(gy_ptrs, mask=mask_gy, other=0.0)  # (BM, BN) bf16
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)     # (BN, BK) bf16
        # tl.dot((BM,BN), (BN,BK)) → (BM, BK) — no tl.trans needed!
        acc += tl.dot(gy_tile, w_tile)

        gy_ptrs += BN * stride_gyn
        w_ptrs += BN * stride_wn

    gx_ptrs = grad_x_ptr + offs_m[:, None] * stride_gxm + offs_k[None, :] * stride_gxk
    mask_gx = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(gx_ptrs, acc.to(tl.bfloat16), mask=mask_gx)


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 2b — grad_W = x.T @ grad_y  (Triton TC matmul, fp32 output)
# ═════════════════════════════════════════════════════════════════════════════
# x is (M, K) bf16, grad_y is (M, N) bf16 → grad_W (K, N) fp32
# grad_W[k, n] = Σ_m x[m, k] * grad_y[m, n]
# Tile (BM, BN) of output grad_W (BM = K-block, BN = N-block). Reduction dim M.
#   x_tile:      (BK, BM) at (m_offset, k_chunk) — load with swapped strides
#   grad_y_tile: (BK, BN) at (m_offset, n_chunk) — standard row-major load
#   acc += tl.dot(x_tile.T, grad_y_tile)  →  (BM, BN)
# PERFORMANCE: kept the tl.trans() here because x_tile is (BK, BM) and we need
# (BM, BK) for tl.dot. The alternative (loading x with swapped strides) would
# give (BM, BK) directly but x is row-major (M, K) so loading (BM, BK) at
# (k_chunk, m_offset) requires strided access. tl.trans is fine here because
# BK=32 fits in shared memory efficiently.
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
        triton.Config({"BM": 128, "BN": 256, "BK": 64}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 64}, num_warps=8, num_stages=4),
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
    """grad_W[k, n] = Σ_m x[m, k] * grad_y[m, n]  → output (K, N) fp32."""
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BM)
    grid_n = tl.cdiv(N, BN)
    pid_k = pid // grid_n
    pid_n = pid % grid_n

    offs_k = pid_k * BM + tl.arange(0, BM)  # K direction (output row)
    offs_n = pid_n * BN + tl.arange(0, BN)  # N direction (output col)
    offs_m = tl.arange(0, BK)                # M direction (reduction)

    # x tile (BK, BM) at (m_offset, k_chunk) — x[m, k] at m*stride_xm + k*stride_xk
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # grad_y tile (BK, BN) at (m_offset, n_chunk) — grad_y[m, n]
    gy_ptrs = grad_y_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for m_iter in range(0, tl.cdiv(M, BK)):
        m_off = m_iter * BK
        mask_x = ((offs_m[:, None] + m_off) < M) & (offs_k[None, :] < K)
        mask_gy = ((offs_m[:, None] + m_off) < M) & (offs_n[None, :] < N)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)  # (BK, BM) bf16
        gy_tile = tl.load(gy_ptrs, mask=mask_gy, other=0.0)  # (BK, BN) bf16
        # tl.dot((BM,BK), (BK,BN)) → (BM, BN) — need x_tile.T
        x_tile_t = tl.trans(x_tile)  # (BM, BK)
        acc += tl.dot(x_tile_t, gy_tile)

        x_ptrs += BK * stride_xm
        gy_ptrs += BK * stride_gym

    gW_ptrs = grad_W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    mask_gW = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    tl.store(gW_ptrs, acc, mask=mask_gW)


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 2c — elementwise bwd: grad_logits + atomic grad_palette (REVISED)
# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE FIXES:
#  1. Block size 32×32 → 128×128 (16x more work per program).
#     Original achieved only 3.7 GB/s (0.4% of HBM3e peak) due to launch overhead.
#  2. Per-block grad_palette accumulation: each block computes its local
#     contribution to grad_palette in registers, then does ONE atomic_add per
#     (group, k) at the end. Reduces atomic ops from K*N to ~K*N/16384.
#  3. Mask is precomputed once, reused for all loads/stores.
#
# Memory traffic per position:
#   Read:  grad_W (4B) + P_aos (8B fp16×4) + palette (8B bf16×4) = 20 B
#   Write: grad_logits (8B fp16×4) + grad_palette (16B fp32×4 atomic) = 24 B
#   Total: 44 B/pos
# For K=9216, N=2560 (23.6M positions): ~1 GB total → at 500 GB/s = 2 ms.
# Target: < 3 ms for the largest shape (was 137 ms).
@triton.jit
def fused_soft_bwd_elementwise_kernel(
    grad_W_ptr,       # (K, N) fp32 — from Kernel 2b
    P_aos_ptr,        # (K, N, 4) fp16 AoS — from forward ctx
    palette_ptr,      # (G, 4) bf16
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

    j_grid = offs_m[:, None]   # (BM, 1)
    o_grid = offs_n[None, :]    # (1, BN)
    idx_grid = j_grid * N + o_grid  # (BM, BN) — flat (K, N) index
    g_grid = o_grid // group_size    # (1, BN)

    plane_size = K * N

    # ── Load grad_W[j, o] (fp32) ───────────────────────────────────────────
    grad_W_val = tl.load(grad_W_ptr + idx_grid, mask=mask, other=0.0)  # (BM, BN) f32

    # ── Load P_aos[j, o, 0..3] (4 fp16, AoS — coalesced 64-bit load) ────────
    paos_base = idx_grid * 4  # (BM, BN)
    p0 = tl.load(P_aos_ptr + paos_base + 0, mask=mask, other=0.0).to(tl.float32)
    p1 = tl.load(P_aos_ptr + paos_base + 1, mask=mask, other=0.0).to(tl.float32)
    p2 = tl.load(P_aos_ptr + paos_base + 2, mask=mask, other=0.0).to(tl.float32)
    p3 = tl.load(P_aos_ptr + paos_base + 3, mask=mask, other=0.0).to(tl.float32)

    # ── Load palette[g, 0..3] (4 bf16) — broadcast across BM rows ────────────
    pal_base = g_grid * 4  # (1, BN)
    c0 = tl.load(palette_ptr + pal_base + 0, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c1 = tl.load(palette_ptr + pal_base + 1, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c2 = tl.load(palette_ptr + pal_base + 2, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c3 = tl.load(palette_ptr + pal_base + 3, mask=mask_n[None, :], other=0.0).to(tl.float32)

    # ── Reconstruct W_soft = Σ P[k] * palette[g, k] ──────────────────────────
    W_soft = p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3  # (BM, BN) f32

    # ── grad_logits[k, j, o] = grad_W * P[k] * (palette[g, k] - W_soft) ─────
    gl_base = idx_grid  # offset within a plane
    gl0 = (grad_W_val * p0 * (c0 - W_soft)).to(tl.float16)
    gl1 = (grad_W_val * p1 * (c1 - W_soft)).to(tl.float16)
    gl2 = (grad_W_val * p2 * (c2 - W_soft)).to(tl.float16)
    gl3 = (grad_W_val * p3 * (c3 - W_soft)).to(tl.float16)
    tl.store(grad_logits_ptr + 0 * plane_size + gl_base, gl0, mask=mask)
    tl.store(grad_logits_ptr + 1 * plane_size + gl_base, gl1, mask=mask)
    tl.store(grad_logits_ptr + 2 * plane_size + gl_base, gl2, mask=mask)
    tl.store(grad_logits_ptr + 3 * plane_size + gl_base, gl3, mask=mask)

    # ── grad_palette[g, k] += grad_W * P[k]  (atomic_add, fp32) ──────────────
    # REDUCED CONTENTION: each block has BN columns spanning BN/group_size groups.
    # For each group g in this block's column range, we sum the contributions
    # from all (BM, BN/group_size) positions in registers, then do ONE atomic_add
    # per (g, k). This reduces atomics from BM*BN to 4 * (BN/group_size) per block.
    #
    # For BM=128, BN=128, GS=256: BN/GS = 0.5 → at most 1 group per block (since
    # BN=128 < GS=256, each block's columns are within a single group). So we do
    # 4 atomic_adds per block. Original did BM*BN = 16384 atomics per block.
    #
    # Implementation: build a per-block contribution array of shape (NUM_GROUPS_IN_BLOCK, 4),
    # then atomic_add each. Use tl.sum over the (BM, BN_per_group) sub-tile.
    #
    # Simplification: since BN <= group_size in our configs (BN=128, GS=256),
    # each block touches exactly ONE group (the group of column pid_n*BN).
    # If BN > group_size, we'd need a loop — but we keep BN <= group_size.
    #
    # The block's group index is g_block = (pid_n * BN) // group_size.
    # All columns in this block map to the same group (when BN <= group_size).

    # Compute per-block contribution to grad_palette[g_block, k] for k=0..3.
    # contribution_k = Σ_{j in block, o in block} grad_W[j, o] * P[k, j, o]
    # = tl.sum(grad_W_val * p_k, axis=(0, 1))  — sum over both BM and BN
    gp_off = pid_n * BN // group_size  # scalar group index for this block
    # Guard: only do the atomic if this block's columns are in-bounds AND
    # all map to the same group (which they do when BN <= group_size).
    # For positions where mask is False (OOB), grad_W_val * p_k is 0 (because
    # p_k was loaded with other=0.0... but grad_W_val may be garbage at OOB).
    # Fix: zero out grad_W_val where mask is False before summing.
    grad_W_masked = tl.where(mask, grad_W_val, 0.0)
    contrib0 = tl.sum(grad_W_masked * p0, axis=None)  # scalar
    contrib1 = tl.sum(grad_W_masked * p1, axis=None)
    contrib2 = tl.sum(grad_W_masked * p2, axis=None)
    contrib3 = tl.sum(grad_W_masked * p3, axis=None)

    # Only atomic_add if the block's first column is in-bounds (any column is valid)
    if pid_n * BN < N:
        tl.atomic_add(grad_palette_ptr + gp_off * 4 + 0, contrib0)
        tl.atomic_add(grad_palette_ptr + gp_off * 4 + 1, contrib1)
        tl.atomic_add(grad_palette_ptr + gp_off * 4 + 2, contrib2)
        tl.atomic_add(grad_palette_ptr + gp_off * 4 + 3, contrib3)


# ═════════════════════════════════════════════════════════════════════════════
# Python launchers
# ═════════════════════════════════════════════════════════════════════════════
# ── Patch 17: Buffer pooling for grad_W (backward intermediate) ─────────────
# The backward's intermediate grad_W (K,N) fp32 = ~26-105 MB per layer (varies
# with K,N). It is produced by fused_soft_bwd_grad_W_triton and consumed
# immediately by fused_soft_bwd_elementwise_triton within the SAME backward
# call — there is NO autograd hazard because grad_W is never saved in ctx and
# never returned to upstream autograd.
#
# Pooling grad_W eliminates 25 torch.empty(K,N,fp32) calls per step + reduces
# peak VRAM (the cached allocator can give back the same buffer immediately,
# but the pool avoids the bookkeeping + fragmentation).
#
# Safety: grad_x, grad_logits, grad_palette are RETURNED to autograd and
# CANNOT be pooled. Only grad_W (the internal intermediate) is pooled.
_BWD_POOL_ENABLED = True
_BWD_POOL: dict[tuple[int, int, int], torch.Tensor] = {}


def _get_pooled_grad_W(K: int, N: int, device: torch.device) -> torch.Tensor:
    """Returns a pooled (K, N) fp32 buffer for the grad_W intermediate.

    Keyed on (K, N, device.index) to handle multi-GPU. Reused across backward
    calls — no torch.empty allocation per call after warmup.
    """
    key = (K, N, device.index if device.type == "cuda" else -1)
    buf = _BWD_POOL.get(key)
    if buf is None:
        buf = torch.empty((K, N), dtype=torch.float32, device=device)
        _BWD_POOL[key] = buf
    return buf


def clear_bwd_pool() -> None:
    """Releases all pooled grad_W buffers. Call on device change / shape change."""
    _BWD_POOL.clear()


def fused_soft_bwd_grad_x_triton(
    grad_y: torch.Tensor,    # (M, N) bf16
    W_ste: torch.Tensor,      # (K, N) bf16
) -> torch.Tensor:
    """grad_x = grad_y @ W_ste.T  → (M, K) bf16."""
    assert grad_y.dtype == torch.bfloat16
    assert W_ste.dtype == torch.bfloat16
    M, N = grad_y.shape
    K, N2 = W_ste.shape
    assert N == N2
    grad_y = grad_y.contiguous()
    W_ste = W_ste.contiguous()
    # grad_x is RETURNED to autograd — cannot be pooled (would corrupt upstream).
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
    """grad_W = x.T @ grad_y → (K, N) fp32 (kept in fp32 for elementwise kernel).

    Patch 17: grad_W is drawn from a module-level pool keyed on (K, N, device)
    when _BWD_POOL_ENABLED. It is consumed immediately by
    fused_soft_bwd_elementwise_triton within the same backward call — no
    autograd hazard (never saved in ctx, never returned upstream).
    """
    assert x.dtype == torch.bfloat16
    assert grad_y.dtype == torch.bfloat16
    M, K = x.shape
    M2, N = grad_y.shape
    assert M == M2
    x = x.contiguous()
    grad_y = grad_y.contiguous()
    if _BWD_POOL_ENABLED:
        grad_W = _get_pooled_grad_W(K, N, x.device)
    else:
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
    # grad_logits and grad_palette are RETURNED to autograd — cannot be pooled.
    grad_logits = torch.empty((4, K, N), dtype=torch.float16, device=P_aos.device)
    grad_palette = torch.zeros((G, 4), dtype=torch.float32, device=palette.device)
    # LARGE block sizes — original 32×32 was launch-overhead-bound (3.7 GB/s = 0.4% peak).
    # 128×128 → 16x more work per program, 16x fewer programs.
    # Constraint: BN <= group_size (256) so each block touches exactly ONE group
    # (enables the per-block register accumulation → minimal atomics).
    # For GS=256, BN=128 is safe (2 blocks per group, each does 4 atomic_adds).
    BM, BN = 128, 128
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
