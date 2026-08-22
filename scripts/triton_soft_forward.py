"""Triton fused kernels for PalettizedLinear soft (Gumbel-Softmax + STE) path.

This module implements the COMPLETE forward + backward in Triton — no CUDA C,
no Python elementwise ops on the hot path, no torch.matmul. All matmuls go
through `tl.dot` (tensor cores). All elementwise ops are fused into Triton
kernels.

WAVE 1 — fused_soft_forward:
  - `compute_P_W_ste_kernel`: per-(j, o) kernel — samples Gumbel noise, softmax → P
    (4 values), writes P_aos (K,N,4) fp16, computes W_soft = Σ P[k]*palette[g,k]
    and W_hard = palette[g, argmax(logits)], and writes W_ste = W_hard (forward
    value; gradients route through W_soft in the backward via on-the-fly
    reconstruction from P_aos + palette).
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
  W_soft:   (K, N) bf16 — materialised for backward reference path (Triton bwd
            recomputes W_soft from P_aos + palette on-the-fly, so this is
            optional — saved only as a debug aid / fallback path)
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
# KERNEL 1a — compute P_aos + W_soft + W_ste (one program per (j, o) tile)
# ═════════════════════════════════════════════════════════════════════════════
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
    P_aos_ptr, W_soft_ptr, W_ste_ptr,
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
      * W_soft = Σ_k P[k] * palette[g, k]  → store W_soft[j, o]
      * argmax over PLAIN logits (NO Gumbel) → W_hard = palette[g, argmax]
      * W_ste = W_hard  (forward value; gradients route through W_soft in bwd)
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

    # ── W_soft = Σ_k P[k] * palette[g, k] ───────────────────────────────────
    W_soft = p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3  # (BM, BN) f32

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
    W_ste = W_hard

    # ── Write W_soft (debug) and W_ste ──
    idx_flat = j_grid * N + o_grid
    tl.store(W_soft_ptr + idx_flat, W_soft.to(tl.bfloat16), mask=mask)
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
# Python launchers
# ═════════════════════════════════════════════════════════════════════════════
def compute_P_W_ste_triton(
    logits: torch.Tensor,
    palette: torch.Tensor,
    group_size: int,
    tau: float,
    step_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (P_aos, W_soft, W_ste). All on the same device as `logits`."""
    assert logits.dtype == torch.float16, f"logits must be fp16, got {logits.dtype}"
    assert palette.dtype == torch.bfloat16, f"palette must be bf16, got {palette.dtype}"
    assert logits.shape[0] == 4
    _, K, N = logits.shape
    G, P4 = palette.shape
    assert P4 == 4
    assert N // group_size == G, f"G mismatch: N//GS={N//group_size} vs G={G}"

    logits = logits.contiguous()
    palette = palette.contiguous()
    P_aos = torch.empty((K, N, 4), dtype=torch.float16, device=logits.device)
    W_soft = torch.empty((K, N), dtype=torch.bfloat16, device=logits.device)
    W_ste = torch.empty((K, N), dtype=torch.bfloat16, device=logits.device)

    grid = lambda meta: (triton.cdiv(K, meta["BM"]), triton.cdiv(N, meta["BN"]))
    compute_P_W_ste_kernel[grid](
        logits, palette, P_aos, W_soft, W_ste,
        K, N, G,
        group_size=group_size,
        tau=float(tau),
        step_seed=int(step_seed),
    )
    return P_aos, W_soft, W_ste


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
           → (P_aos, W_soft, W_ste)
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
        P_aos, W_soft, W_ste = compute_P_W_ste_triton(
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
        # Wave 2 will implement this.
        raise NotImplementedError(
            "TritonSoftLinear.backward is implemented in Wave 2. "
            "Run the Wave 1 forward-only test until then."
        )


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
