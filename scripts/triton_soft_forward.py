"""Triton fused kernels for PalettizedLinear soft (Gumbel-Softmax + STE) path.

This module implements the COMPLETE forward + backward in Triton — no CUDA C,
no Python elementwise ops on the hot path, no torch.matmul. All matmuls go
through `tl.dot` (tensor cores). All elementwise ops are fused into Triton
kernels.

WAVE 1 — fused_soft_forward:
  - `compute_P_W_ste_kernel`: per-(j, o) kernel — samples Gumbel noise, softmax → P
    (4 values), writes P_aos (K,N,4) fp16, computes W_hard = palette[g, argmax(logits)]
    and writes W_ste = W_hard (forward value; gradients route through W_soft in
    the backward via on-the-fly reconstruction from P_aos + palette).
    [Patch 16: W_soft is NOT computed or stored here — see research-indices-training/
    01_gumbel_softmax_audit.md Finding 8. The forward only uses W_ste; the backward
    recomputes W_soft from P_aos + palette on-the-fly.]
  - `compute_P_W_ste_batched_kernel`: Patch 15 — single kernel launch processes
    all N layers via grid (cdiv(max_K, BM), cdiv(max_N, BN), N). blockIdx.z =
    layer_idx. Each program loads its layer's (K, N, palette_ptr, logits_ptr,
    P_aos_ptr, W_ste_ptr) from per-layer pointer/shape arrays. Replaces N separate
    compute_P_W_ste_kernel launches with one (eliminates N-1 dispatch overheads).
  - `fused_soft_matmul_kernel`: standard Triton matmul, y = x @ W_ste + bias.
  - `TritonSoftLinear` (torch.autograd.Function): forward wires the two kernels
    + saves ctx. Backward is filled in by Wave 2.

Layouts (must match existing CUDA path):
  x:        (M, K) bf16, contiguous
  palette:  (G, 4) bf16, contiguous
  logits:   (4, K, N) fp16, SoA (plane stride = K*N)
  bias:     (N,) bf16 or None
  y:        (M, N) bf16
  P_aos:    (K, N, 4) fp16 — last dim is the 4 planes (AoS for coalesced 64-bit
            load in the backward)
  W_ste:    (K, N) bf16 — the STE weight (= W_hard numerically) used in the
            forward matmul AND saved for grad_x = grad_y @ W_ste.T in the backward
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
# Gumbel LCG — matches the CUDA `gumbel_sample(seed, idx)` exactly
# ═════════════════════════════════════════════════════════════════════════════
# CUDA uses uint32_t arithmetic (wraps mod 2^32). Triton uses int32 — same
# wraparound semantics for *, +, ^, >>. The only catch is the literal
# 0x9E3779B9 (= 2654435769 unsigned) overflows int32's positive range; we use
# its signed-int32 representation -1640531527 (= 0x9E3779B9 - 2^32), which has
# the same bit pattern under two's complement, so all downstream ops match.
#
# Constants are inlined directly inside the @triton.jit body because Triton
# 3.5+ forbids accessing non-constexpr module globals from JIT kernels.


@triton.jit
def _gumbel_sample(seed, idx):
    """Returns a float32 Gumbel(0, 1) sample. Matches CUDA gumbel_sample(seed, idx).

    Constants are inlined (Triton 3.5+ forbids non-constexpr globals).
    """
    # 0x9E3779B9 as signed int32 = -1640531527 (same bit pattern, fits in int32)
    x = seed ^ (idx * -1640531527)
    x = x ^ (x >> 13)
    x = x * 1103515245 + 12345
    x = x ^ (x >> 17)
    x = x * 1103515245 + 12345
    # Low 24 bits → uniform in [0, 1)
    u24 = (x & 0xFFFFFF).to(tl.float32) * (1.0 / 16777216.0)
    u24 = tl.maximum(u24, 1e-7)
    return -tl.log(-tl.log(u24))


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 1a — compute P_aos + W_ste (one program per (j, o) tile)
# ═════════════════════════════════════════════════════════════════════════════
# Patch 16 (research-indices-training/01_gumbel_softmax_audit.md Finding 8):
# This kernel previously ALSO computed and stored W_soft = Σ_k P[k]*palette[g,k]
# to HBM (a wasted write — the forward only uses W_ste = W_hard for the matmul,
# and the backward reconstructs W_soft on-the-fly from P_aos + palette). The
# W_soft store + HBM write has been removed; the kernel now ONLY writes
# P_aos (K,N,4) fp16 + W_ste (K,N) bf16.
@triton.autotune(
    configs=[
        triton.Config({"BM": 16, "BN": 16}, num_warps=8, num_stages=1),
        triton.Config({"BM": 32, "BN": 16}, num_warps=8, num_stages=1),
        triton.Config({"BM": 16, "BN": 32}, num_warps=8, num_stages=1),
        triton.Config({"BM": 32, "BN": 32}, num_warps=8, num_stages=1),
        triton.Config({"BM": 64, "BN": 32}, num_warps=8, num_stages=1),
    ],
    key=["K", "N"],
)
@triton.jit
def compute_P_W_ste_kernel(
    logits_ptr, palette_ptr,
    P_aos_ptr, W_ste_ptr,
    K, N, G,
    group_size: tl.constexpr,
    tau,
    step_seed,
    BM: tl.constexpr, BN: tl.constexpr,
):
    """For each (j, o) in a BM×BN tile:
      * load 4 logits[k, j, o] from SoA layout
      * sample 4 Gumbel noises (deterministic LCG, matches CUDA)
      * (logits + gumbel) / tau → softmax → P[k]
      * store P_aos[j, o, 0..3]  (AoS — last dim is the 4 planes)
      * argmax over PLAIN logits (NO Gumbel) → W_hard = palette[g, argmax]
      * W_ste = W_hard  (forward value; gradients route through W_soft in bwd)
    NOTE (Patch 16): W_soft is NOT computed here — it is recomputed on-the-fly
    in the backward from P_aos + palette (see fused_soft_bwd_elementwise_kernel).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)  # K direction (j)
    offs_n = pid_n * BN + tl.arange(0, BN)  # N direction (o)

    mask_m = offs_m < K
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    j_grid = offs_m[:, None]   # (BM, 1)
    o_grid = offs_n[None, :]   # (1, BN)
    idx_grid = j_grid * N + o_grid  # (BM, BN)
    g_grid = o_grid // group_size  # group index for each column

    plane_size = K * N

    # ── Load 4 logits from SoA layout ──────────────────────────────────────
    l0 = tl.load(logits_ptr + 0 * plane_size + idx_grid, mask=mask, other=-1e4).to(tl.float32)
    l1 = tl.load(logits_ptr + 1 * plane_size + idx_grid, mask=mask, other=-1e4).to(tl.float32)
    l2 = tl.load(logits_ptr + 2 * plane_size + idx_grid, mask=mask, other=-1e4).to(tl.float32)
    l3 = tl.load(logits_ptr + 3 * plane_size + idx_grid, mask=mask, other=-1e4).to(tl.float32)

    # ── Sample 4 Gumbel noises — idx for noise k = idx * 4 + k (matches CUDA) ─
    inv_tau = 1.0 / tau
    n0 = (l0 + _gumbel_sample(step_seed, idx_grid * 4 + 0)) * inv_tau
    n1 = (l1 + _gumbel_sample(step_seed, idx_grid * 4 + 1)) * inv_tau
    n2 = (l2 + _gumbel_sample(step_seed, idx_grid * 4 + 2)) * inv_tau
    n3 = (l3 + _gumbel_sample(step_seed, idx_grid * 4 + 3)) * inv_tau

    # ── Softmax (numerically stable) ────────────────────────────────────────
    m = tl.maximum(tl.maximum(n0, n1), tl.maximum(n2, n3))
    e0 = tl.exp(n0 - m)
    e1 = tl.exp(n1 - m)
    e2 = tl.exp(n2 - m)
    e3 = tl.exp(n3 - m)
    s = e0 + e1 + e2 + e3
    p0 = e0 / s
    p1 = e1 / s
    p2 = e2 / s
    p3 = e3 / s

    # ── Write P_aos (K, N, 4) — last dim is the 4 planes (AoS) ─────────────
    paos_base = idx_grid * 4  # (BM, BN)
    tl.store(P_aos_ptr + paos_base + 0, p0.to(tl.float16), mask=mask)
    tl.store(P_aos_ptr + paos_base + 1, p1.to(tl.float16), mask=mask)
    tl.store(P_aos_ptr + paos_base + 2, p2.to(tl.float16), mask=mask)
    tl.store(P_aos_ptr + paos_base + 3, p3.to(tl.float16), mask=mask)

    # ── Load 4 palette values: palette[g, 0..3] ─────────────────────────────
    pal_base = g_grid * 4  # (1, BN)
    c0 = tl.load(palette_ptr + pal_base + 0, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c1 = tl.load(palette_ptr + pal_base + 1, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c2 = tl.load(palette_ptr + pal_base + 2, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c3 = tl.load(palette_ptr + pal_base + 3, mask=mask_n[None, :], other=0.0).to(tl.float32)

    # ── argmax over PLAIN logits (NO Gumbel — matches existing CUDA path) ────
    # Use the canonical "first-index wins" tie-break: k wins iff it's strictly
    # greater than all earlier k's and ≥ all later k's. This matches
    # torch.argmax which returns the FIRST occurrence of the max.
    # All comparisons are on the float32 l0..l3 (post-load cast).
    is0 = (l0 >= l1) & (l0 >= l2) & (l0 >= l3)
    is1 = (l1 >  l0) & (l1 >= l2) & (l1 >= l3)
    is2 = (l2 >  l0) & (l2 >  l1) & (l2 >= l3)
    is3 = (l3 >  l0) & (l3 >  l1) & (l3 >  l2)
    f0 = is0.to(tl.float32)
    f1 = is1.to(tl.float32)
    f2 = is2.to(tl.float32)
    f3 = is3.to(tl.float32)
    W_hard = f0 * c0 + f1 * c1 + f2 * c2 + f3 * c3

    # STE: forward uses W_hard (numerically), backward routes through W_soft
    # (recomputed on-the-fly from P_aos + palette in the elementwise kernel).
    W_ste = W_hard

    # ── Write W_ste (W_soft is NOT written — Patch 16) ──────────────────────
    idx_flat = j_grid * N + o_grid
    tl.store(W_ste_ptr + idx_flat, W_ste.to(tl.bfloat16), mask=mask)


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 1b — fused soft matmul: y = x @ W_ste + bias (Triton TC matmul)
# ═════════════════════════════════════════════════════════════════════════════
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
def fused_soft_matmul_kernel(
    x_ptr, W_ste_ptr, bias_ptr,
    y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ste_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BK)):
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] + k_iter * BK < K)
        mask_w = (offs_k[:, None] + k_iter * BK < K) & (offs_n[None, :] < N)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.bfloat16)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0).to(tl.bfloat16)
        acc += tl.dot(x_tile, w_tile)

        x_ptrs += BK * stride_xk
        w_ptrs += BK * stride_wk

    if bias_ptr is not None:
        bias_tile = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias_tile[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_y)


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 1c — BATCHED compute_P_W_ste (Patch 15): 25 layers → 1 kernel launch
# ═════════════════════════════════════════════════════════════════════════════
# Replaces 25 separate compute_P_W_ste_kernel launches with a single kernel
# whose grid is (cdiv(max_K, BM), cdiv(max_N, BN), n_layers). blockIdx.z is
# the layer index. Each program loads its layer's (K, N, palette_ptr,
# logits_ptr, P_aos_ptr, W_ste_ptr) from per-layer pointer/shape arrays,
# then runs the same per-(j, o) tile compute as the single-layer kernel.
#
# Per-layer Gumbel seed decorrelation: step_seed = base_seed + layer_idx
# (so each layer samples independent Gumbel noise — matches the research doc
# 03_batched_compute_pw.md §2.4).
#
# Early-exit: positions outside the layer's (K_l, N_l) shape are masked out
# (tl.load returns 'other', tl.store is a no-op). This wastes ~30% of compute
# on the smallest layer (which is [1024, 2560] vs max [9216, 9216]) but the
# wasted threads exit immediately and do not consume SM resources.
#
# Pointer passing: each layer's input/output tensors are passed as int64
# pointer arrays (logits_ptrs, palette_ptrs, P_aos_ptrs, W_ste_ptrs). The
# kernel loads the int64 value via tl.load, then casts to a typed pointer
# via .to(tl.pointer_type(dtype)).
#
# Autotune key: max_K, max_N (not per-layer K, N — those vary by layer).
# The autotuner picks the best config for the largest layer; smaller layers
# reuse the same config (with masking).
@triton.autotune(
    configs=[
        triton.Config({"BM": 16, "BN": 16}, num_warps=8, num_stages=1),
        triton.Config({"BM": 32, "BN": 16}, num_warps=8, num_stages=1),
        triton.Config({"BM": 16, "BN": 32}, num_warps=8, num_stages=1),
        triton.Config({"BM": 32, "BN": 32}, num_warps=8, num_stages=1),
        triton.Config({"BM": 64, "BN": 32}, num_warps=8, num_stages=1),
    ],
    key=["max_K", "max_N"],
)
@triton.jit
def compute_P_W_ste_batched_kernel(
    # Per-layer pointer arrays — (n_layers,) int64, each element is a typed ptr
    logits_ptrs,     # *int64 — array of pointers to (4, K_l, N_l) fp16 logits
    palette_ptrs,    # *int64 — array of pointers to (G_l, 4) bf16 palettes
    P_aos_ptrs,      # *int64 — array of pointers to (K_l, N_l, 4) fp16 OUTPUT
    W_ste_ptrs,      # *int64 — array of pointers to (K_l, N_l) bf16 OUTPUT
    # Per-layer shape arrays — (n_layers,) int32
    Ks,              # *int32 — K per layer
    Ns,              # *int32 — N per layer
    # Scalar args
    max_K, max_N,
    group_size: tl.constexpr,
    tau,
    base_seed,
    BM: tl.constexpr, BN: tl.constexpr,
):
    """Batched compute_P_W_ste — one program per (j_tile, o_tile, layer).

    For layer blockIdx.z, loads (K, N, palette_ptr, logits_ptr, P_aos_ptr,
    W_ste_ptr) from the per-layer arrays, then runs the same compute as
    compute_P_W_ste_kernel (Patch 16: P_aos + W_ste only, no W_soft).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    layer_idx = tl.program_id(2)

    # ── Load per-layer K, N (int32 scalars) ─────────────────────────────────
    K = tl.load(Ks + layer_idx)
    N = tl.load(Ns + layer_idx)

    # ── Load per-layer pointers (int64) and cast to typed pointers ──────────
    logits_ptr = tl.load(logits_ptrs + layer_idx).to(tl.pointer_type(tl.float16))
    palette_ptr = tl.load(palette_ptrs + layer_idx).to(tl.pointer_type(tl.bfloat16))
    P_aos_ptr = tl.load(P_aos_ptrs + layer_idx).to(tl.pointer_type(tl.float16))
    W_ste_ptr = tl.load(W_ste_ptrs + layer_idx).to(tl.pointer_type(tl.bfloat16))

    # Per-layer seed for Gumbel decorrelation
    step_seed = base_seed + layer_idx

    # ── Same per-(j, o) tile compute as compute_P_W_ste_kernel ──────────────
    offs_m = pid_m * BM + tl.arange(0, BM)  # K direction (j)
    offs_n = pid_n * BN + tl.arange(0, BN)  # N direction (o)

    mask_m = offs_m < K
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    j_grid = offs_m[:, None]   # (BM, 1)
    o_grid = offs_n[None, :]   # (1, BN)
    idx_grid = j_grid * N + o_grid  # (BM, BN) — NOTE: N is per-layer here
    g_grid = o_grid // group_size  # group index for each column

    plane_size = K * N  # per-layer plane stride for SoA logits

    # ── Load 4 logits from SoA layout ──────────────────────────────────────
    l0 = tl.load(logits_ptr + 0 * plane_size + idx_grid, mask=mask, other=-1e4).to(tl.float32)
    l1 = tl.load(logits_ptr + 1 * plane_size + idx_grid, mask=mask, other=-1e4).to(tl.float32)
    l2 = tl.load(logits_ptr + 2 * plane_size + idx_grid, mask=mask, other=-1e4).to(tl.float32)
    l3 = tl.load(logits_ptr + 3 * plane_size + idx_grid, mask=mask, other=-1e4).to(tl.float32)

    # ── Sample 4 Gumbel noises — per-layer seed for decorrelation ───────────
    inv_tau = 1.0 / tau
    n0 = (l0 + _gumbel_sample(step_seed, idx_grid * 4 + 0)) * inv_tau
    n1 = (l1 + _gumbel_sample(step_seed, idx_grid * 4 + 1)) * inv_tau
    n2 = (l2 + _gumbel_sample(step_seed, idx_grid * 4 + 2)) * inv_tau
    n3 = (l3 + _gumbel_sample(step_seed, idx_grid * 4 + 3)) * inv_tau

    # ── Softmax (numerically stable) ────────────────────────────────────────
    m = tl.maximum(tl.maximum(n0, n1), tl.maximum(n2, n3))
    e0 = tl.exp(n0 - m)
    e1 = tl.exp(n1 - m)
    e2 = tl.exp(n2 - m)
    e3 = tl.exp(n3 - m)
    s = e0 + e1 + e2 + e3
    p0 = e0 / s
    p1 = e1 / s
    p2 = e2 / s
    p3 = e3 / s

    # ── Write P_aos (K, N, 4) — last dim is the 4 planes (AoS) ─────────────
    paos_base = idx_grid * 4  # (BM, BN)
    tl.store(P_aos_ptr + paos_base + 0, p0.to(tl.float16), mask=mask)
    tl.store(P_aos_ptr + paos_base + 1, p1.to(tl.float16), mask=mask)
    tl.store(P_aos_ptr + paos_base + 2, p2.to(tl.float16), mask=mask)
    tl.store(P_aos_ptr + paos_base + 3, p3.to(tl.float16), mask=mask)

    # ── Load 4 palette values: palette[g, 0..3] ─────────────────────────────
    pal_base = g_grid * 4  # (1, BN)
    c0 = tl.load(palette_ptr + pal_base + 0, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c1 = tl.load(palette_ptr + pal_base + 1, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c2 = tl.load(palette_ptr + pal_base + 2, mask=mask_n[None, :], other=0.0).to(tl.float32)
    c3 = tl.load(palette_ptr + pal_base + 3, mask=mask_n[None, :], other=0.0).to(tl.float32)

    # ── argmax over PLAIN logits (NO Gumbel) → W_hard = palette[g, argmax] ──
    is0 = (l0 >= l1) & (l0 >= l2) & (l0 >= l3)
    is1 = (l1 >  l0) & (l1 >= l2) & (l1 >= l3)
    is2 = (l2 >  l0) & (l2 >  l1) & (l2 >= l3)
    is3 = (l3 >  l0) & (l3 >  l1) & (l3 >  l2)
    f0 = is0.to(tl.float32)
    f1 = is1.to(tl.float32)
    f2 = is2.to(tl.float32)
    f3 = is3.to(tl.float32)
    W_hard = f0 * c0 + f1 * c1 + f2 * c2 + f3 * c3

    # STE: forward uses W_hard (numerically), backward routes through W_soft
    # (recomputed on-the-fly from P_aos + palette in the elementwise kernel).
    W_ste = W_hard

    # ── Write W_ste ──────────────────────────────────────────────────────────
    idx_flat = j_grid * N + o_grid
    tl.store(W_ste_ptr + idx_flat, W_ste.to(tl.bfloat16), mask=mask)


# ═════════════════════════════════════════════════════════════════════════════
# Python launchers
# ═════════════════════════════════════════════════════════════════════════════
# ── Patch 17: Buffer pooling for P_aos + W_ste ──────────────────────────────
# A module-level pool keyed on (K, N, device.index) that reuses the same
# P_aos (K,N,4) fp16 + W_ste (K,N) bf16 tensors across forward calls. This
# eliminates ~50 torch.empty calls per step (2 per layer × 25 layers) +
# associated CUDA caching-allocator bookkeeping.
#
# SAFETY INVARIANT (must hold for correctness):
#   Pooling is only safe under the standard training pattern:
#       forward → loss → backward → (graph freed) → next forward
#   The pooled P_aos + W_ste are saved via ctx.save_for_backward in
#   TritonSoftLinear.forward. They must NOT be modified in-place after being
#   saved, until the corresponding backward has consumed them. Because the
#   backward of step N completes (and frees the graph) before step N+1's
#   forward starts, the next compute_P_W_ste_kernel write always lands AFTER
#   the previous backward's read. Triton tl.store bypasses PyTorch's
#   in-place modification detection (version counter), so corruption would
#   be SILENT if this invariant were violated (e.g. retain_graph=True across
#   steps). The training code (train_qwen.py) does NOT use retain_graph.
#
#   grad_x, grad_logits, grad_palette are RETURNED to autograd and CANNOT
#   be pooled — they flow into upstream layers / optimizer state. Only the
#   intermediate grad_W in the backward is pooled (see triton_soft_backward.py).
#
# To disable pooling (e.g. for debugging), set _POOLING_ENABLED = False.
_POOLING_ENABLED = True
_P_POOL: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}


def _get_pooled_buffers(K: int, N: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Returns pooled P_aos (K,N,4) fp16 + W_ste (K,N) bf16 buffers.

    Allocates on first call for a given (K, N, device); reuses thereafter.
    Keyed on (K, N, device.index) to handle multi-GPU.
    """
    key = (K, N, device.index if device.type == "cuda" else -1)
    bufs = _P_POOL.get(key)
    if bufs is None:
        bufs = {
            "P_aos": torch.empty((K, N, 4), dtype=torch.float16, device=device),
            "W_ste": torch.empty((K, N), dtype=torch.bfloat16, device=device),
        }
        _P_POOL[key] = bufs
    return bufs


def clear_P_pool() -> None:
    """Releases all pooled P_aos + W_ste buffers.

    Call this when:
      - The model is moved to a different device
      - Training resumes from a checkpoint with different shapes
      - Memory pressure requires reclaiming the pool (~1.3 GB for 25 layers)
    """
    _P_POOL.clear()


def compute_P_W_ste_triton(
    logits: torch.Tensor,
    palette: torch.Tensor,
    group_size: int,
    tau: float,
    step_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (P_aos, W_ste). All on the same device as `logits`.

    Patch 16: W_soft is NOT returned — the backward recomputes it on-the-fly
    from P_aos + palette (see fused_soft_bwd_elementwise_kernel).

    Patch 17: P_aos + W_ste are drawn from a module-level pool keyed on
    (K, N, device) when _POOLING_ENABLED. The kernel writes into the pooled
    buffers in-place — no torch.empty allocation per call after warmup.
    See _P_POOL docstring for the safety invariant.
    """
    assert logits.dtype == torch.float16, f"logits must be fp16, got {logits.dtype}"
    assert palette.dtype == torch.bfloat16, f"palette must be bf16, got {palette.dtype}"
    assert logits.shape[0] == 4
    _, K, N = logits.shape
    G, P4 = palette.shape
    assert P4 == 4
    assert N // group_size == G, f"G mismatch: N//GS={N//group_size} vs G={G}"

    logits = logits.contiguous()
    palette = palette.contiguous()
    if _POOLING_ENABLED:
        bufs = _get_pooled_buffers(K, N, logits.device)
        P_aos = bufs["P_aos"]
        W_ste = bufs["W_ste"]
    else:
        P_aos = torch.empty((K, N, 4), dtype=torch.float16, device=logits.device)
        W_ste = torch.empty((K, N), dtype=torch.bfloat16, device=logits.device)

    grid = lambda meta: (triton.cdiv(K, meta["BM"]), triton.cdiv(N, meta["BN"]))
    compute_P_W_ste_kernel[grid](
        logits, palette, P_aos, W_ste,
        K, N, G,
        group_size=group_size,
        tau=float(tau),
        step_seed=int(step_seed),
    )
    return P_aos, W_ste


def compute_P_W_ste_batched_triton(
    layers: list[dict],
    tau: float,
    base_seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Batched compute_P_W_ste — processes all layers in ONE kernel launch.

    Patch 15: replaces N separate compute_P_W_ste_triton calls with a single
    kernel launch whose grid is (cdiv(max_K, BM), cdiv(max_N, BN), n_layers).
    blockIdx.z = layer_idx. Each program loads its layer's (K, N, palette_ptr,
    logits_ptr, P_aos_ptr, W_ste_ptr) from per-layer pointer/shape arrays.

    Per-layer Gumbel seed decorrelation: step_seed = base_seed + layer_idx.

    Args:
        layers: list of dicts, each with keys:
            - 'logits':  (4, K_l, N_l) fp16 SoA
            - 'palette': (G_l, 4) bf16
            - 'group_size': int (typically 256 — must be SAME across all layers)
        tau: temperature (same for all layers)
        base_seed: base seed for Gumbel noise (per-layer seed = base_seed + idx)

    Returns:
        list of (P_aos, W_ste) tuples, one per layer — same as if
        compute_P_W_ste_triton were called per layer.

    Uses the buffer pool (Patch 17) for P_aos + W_ste per layer.
    """
    n_layers = len(layers)
    assert n_layers > 0, "compute_P_W_ste_batched_triton: empty layers list"
    group_size = layers[0]["group_size"]
    device = layers[0]["logits"].device
    for i, L in enumerate(layers):
        assert L["logits"].dtype == torch.float16, f"layer {i}: logits must be fp16"
        assert L["palette"].dtype == torch.bfloat16, f"layer {i}: palette must be bf16"
        assert L["logits"].shape[0] == 4, f"layer {i}: logits must have 4 planes"
        assert L["group_size"] == group_size, (
            f"layer {i}: group_size mismatch ({L['group_size']} vs {group_size}) — "
            "batched kernel requires same group_size across all layers"
        )
        assert L["logits"].device == device, f"layer {i}: device mismatch"

    # Compute max_K, max_N for the grid
    Ks_list = [L["logits"].shape[1] for L in layers]
    Ns_list = [L["logits"].shape[2] for L in layers]
    max_K = max(Ks_list)
    max_N = max(Ns_list)

    # Build per-layer shape + pointer arrays on the device
    Ks_t = torch.tensor(Ks_list, dtype=torch.int32, device=device)
    Ns_t = torch.tensor(Ns_list, dtype=torch.int32, device=device)
    logits_ptrs = torch.tensor(
        [L["logits"].contiguous().data_ptr() for L in layers],
        dtype=torch.int64, device=device,
    )
    palette_ptrs = torch.tensor(
        [L["palette"].contiguous().data_ptr() for L in layers],
        dtype=torch.int64, device=device,
    )

    # Allocate (or reuse from pool) per-layer P_aos + W_ste buffers
    P_aos_list: list[torch.Tensor] = []
    W_ste_list: list[torch.Tensor] = []
    P_aos_ptrs_list: list[int] = []
    W_ste_ptrs_list: list[int] = []
    for i, L in enumerate(layers):
        K_l, N_l = Ks_list[i], Ns_list[i]
        if _POOLING_ENABLED:
            bufs = _get_pooled_buffers(K_l, N_l, device)
            P_aos = bufs["P_aos"]
            W_ste = bufs["W_ste"]
        else:
            P_aos = torch.empty((K_l, N_l, 4), dtype=torch.float16, device=device)
            W_ste = torch.empty((K_l, N_l), dtype=torch.bfloat16, device=device)
        P_aos_list.append(P_aos)
        W_ste_list.append(W_ste)
        P_aos_ptrs_list.append(P_aos.data_ptr())
        W_ste_ptrs_list.append(W_ste.data_ptr())

    P_aos_ptrs = torch.tensor(P_aos_ptrs_list, dtype=torch.int64, device=device)
    W_ste_ptrs = torch.tensor(W_ste_ptrs_list, dtype=torch.int64, device=device)

    # Launch the batched kernel
    grid = lambda meta: (
        triton.cdiv(max_K, meta["BM"]),
        triton.cdiv(max_N, meta["BN"]),
        n_layers,
    )
    compute_P_W_ste_batched_kernel[grid](
        logits_ptrs, palette_ptrs, P_aos_ptrs, W_ste_ptrs,
        Ks_t, Ns_t,
        max_K, max_N,
        group_size=group_size,
        tau=float(tau),
        base_seed=int(base_seed),
    )
    return list(zip(P_aos_list, W_ste_list))


def fused_soft_matmul_triton(
    x: torch.Tensor,
    W_ste: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
    assert W_ste.dtype == torch.bfloat16, f"W_ste must be bf16, got {W_ste.dtype}"
    M, K = x.shape
    K2, N = W_ste.shape
    assert K == K2, f"K mismatch: x.K={K} vs W_ste.K={K2}"
    x = x.contiguous()
    W_ste = W_ste.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    fused_soft_matmul_kernel[grid](
        x, W_ste, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        W_ste.stride(0), W_ste.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


# ═════════════════════════════════════════════════════════════════════════════
# Autograd Function — wires forward kernels + saves ctx for Wave 2 backward
# ═════════════════════════════════════════════════════════════════════════════
_SOFT_STEP_SEED = 0


def _next_soft_step_seed() -> int:
    global _SOFT_STEP_SEED
    _SOFT_STEP_SEED = (_SOFT_STEP_SEED + 1) & 0xFFFFFFFF
    return _SOFT_STEP_SEED


class TritonSoftLinear(torch.autograd.Function):
    """Triton-only fused soft forward.

    forward(ctx, x, palette, logits, bias, group_size, tau) -> y
      Steps:
        1. compute_P_W_ste_triton(logits, palette, group_size, tau, step_seed)
           → (P_aos, W_ste)   [Patch 16: no W_soft — recomputed in backward]
        2. fused_soft_matmul_triton(x, W_ste, bias) → y
        3. ctx.save_for_backward(x, palette, logits, P_aos, W_ste)

    backward: filled in by Wave 2 (raises NotImplementedError for now).
    """

    @staticmethod
    def forward(ctx, x, palette, logits, bias, group_size, tau):
        x = x.contiguous()
        palette = palette.contiguous()
        logits = logits.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        M, K = x.shape
        G, P_size = palette.shape
        n_planes, K_, N = logits.shape
        assert P_size == 4
        assert n_planes == 4
        assert K == K_, f"x K={K} != logits K={K_}"
        assert N % group_size == 0
        assert N // group_size == G

        step_seed = _next_soft_step_seed()
        P_aos, W_ste = compute_P_W_ste_triton(
            logits, palette, group_size, float(tau), step_seed
        )

        y = fused_soft_matmul_triton(x, W_ste, bias)

        ctx.save_for_backward(x, palette, logits, P_aos, W_ste)
        ctx.group_size = group_size
        ctx.tau = tau
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # Wave 2 — fused Triton backward.
        #   1. grad_x = grad_y @ W_ste.T            (Triton TC matmul)
        #   2. grad_logits + grad_palette via Patch 18 chunked kernel:
        #      - Computes grad_W = x.T @ grad_y via tl.dot (tensor cores, fp32 acc)
        #        — the grad_W tile lives in REGISTERS, never written to HBM.
        #      - Immediately consumes grad_W for grad_logits + grad_palette.
        #      Replaces the two-kernel split (fused_soft_bwd_grad_W_triton +
        #      fused_soft_bwd_elementwise_triton) — eliminates 52 MB write +
        #      52 MB read of grad_W per layer × 25 = 2.6 GB/step HBM traffic.
        # Math matches existing CUDA `fused_lut_linear_soft_bwd_fused_aos_kernel`.
        from triton_soft_backward import (
            fused_soft_bwd_grad_x_triton,
            fused_soft_bwd_chunked_triton,
        )
        x, palette, logits, P_aos, W_ste = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        M, K = x.shape
        _, _, N = logits.shape
        G = palette.shape[0]
        GS = ctx.group_size

        needs_grad_x = ctx.needs_input_grad[0]
        needs_grad_palette = ctx.needs_input_grad[1]
        needs_grad_logits = ctx.needs_input_grad[2]
        needs_grad_bias = ctx.has_bias and ctx.needs_input_grad[3]

        # ── 1. grad_x = grad_y @ W_ste.T ────────────────────────────────────
        grad_x = None
        if needs_grad_x:
            grad_x = fused_soft_bwd_grad_x_triton(grad_y, W_ste)

        # ── 2. grad_logits + grad_palette via Patch 18 chunked kernel ────────
        # Fused grad_W (matmul) + elementwise — no HBM grad_W intermediate.
        grad_logits = None
        grad_palette = None
        if needs_grad_logits or needs_grad_palette:
            grad_logits, grad_palette = fused_soft_bwd_chunked_triton(
                x, grad_y, P_aos, palette, GS
            )
            if not needs_grad_logits:
                grad_logits = None
            if not needs_grad_palette:
                grad_palette = None

        # grad_bias = grad_y.sum(dim=0) — kept in PyTorch (one reduction per step,
        # not the hot path; fusing into a Triton kernel is in Wave 4 / Kernel 4)
        grad_bias = None
        if needs_grad_bias:
            grad_bias = grad_y.sum(dim=0)

        # Return tuple matches forward input order: (x, palette, logits, bias, group_size, tau)
        return grad_x, grad_palette, grad_logits, grad_bias, None, None


def triton_soft_linear(
    x: torch.Tensor,
    palette: torch.Tensor,
    logits: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 256,
    tau: float = 1.0,
) -> torch.Tensor:
    """Functional interface — matches `fused_lut_linear_soft` signature."""
    return TritonSoftLinear.apply(x, palette, logits, bias, group_size, tau)
