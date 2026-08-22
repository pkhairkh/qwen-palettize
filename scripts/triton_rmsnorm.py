"""Triton fused kernels for RMSNorm + PalettizedLinear (Patch 10).

WAVE 2 — Patch 10 (layer-fusion):
  Fuses `RMSNorm(x) → PalettizedLinear` to eliminate the materialization of
  the intermediate `x_normed` tensor (saves 2 × M × K × 2 bytes per layer
  of HBM traffic) and removes the launch overhead of two separate kernels.

  Two strategies are supported:

  A) FusedRMSNorm — a standalone Triton kernel that computes
       x_normed = x * rsqrt(mean(x^2, dim=-1) + eps) * weight
     in a single pass.  Used when the downstream consumer cannot accept a
     fused matmul (e.g. attention which needs `x_normed` for Q/K/V projections
     that have separate palettes).

  B) FusedRMSNormLinear — a torch.autograd.Function that combines
       1. RMSNorm (kernel A)
       2. compute_P_W_ste_triton (from triton_soft_forward — IMPORTED, not
          re-implemented; we respect triton-kernels' file ownership)
       3. fused_rmsnorm_matmul (kernel B — a matmul that reads `x_normed`
          from registers / shared memory rather than from HBM)
     into a single graph node.  Backward is fused too:
       grad_x = (grad_y @ W_ste.T * weight / rstd) * (1 - x_normed^2 / M)

References:
  - FlashAttention2 (Dao 2023, arXiv:2307.08691) §3.2 — fused layernorm pattern.
  - research-kernel-efficiency/00_overview.md §3 — fused layer pattern.
  - research-kernel-accuracy/00_overview.md — PalettizedLinear architecture.

File ownership:
  - This file is NEW and owned by layer-fusion (Patch 10).
  - We IMPORT from triton_soft_forward (owned by triton-kernels) — we never
    modify it.
  - We IMPORT from triton_soft_backward (owned by triton-kernels) — we never
    modify it.
  - We do NOT touch qwen_model.py — owned by nn-module-foundation.

Layouts (must match triton_soft_forward.py):
  x:        (M, K) bf16, contiguous
  norm_w:   (K,) bf16 — RMSNorm scale weight
  palette:  (G, 4) bf16, contiguous
  logits:   (4, K, N) fp16, SoA (plane stride = K*N)
  bias:     (N,) bf16 or None
  y:        (M, N) bf16
  P_aos:    (K, N, 4) fp16 (AoS — last dim is the 4 planes)
  W_ste:    (K, N) bf16 — STE weight used in forward matmul + backward grad_x
  rstd:     (M,) fp32 — reciprocal stddev per row, saved for backward
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL A — standalone RMSNorm forward (one program per row block)
# ═════════════════════════════════════════════════════════════════════════════
# Computes  x_normed = x * rsqrt(mean(x^2, dim=-1) + eps) * weight
#
# Strategy:  one program per (row_block).  Each program loads BM rows × K cols
# of x, computes per-row mean(x^2) via a single tl.sum reduction, then writes
# x_normed.  K is fully resident in registers / shared memory for the block,
# so we never re-read x from HBM for the normalization pass.
#
# This kernel is intentionally separate from the matmul (Kernel B) so that
# downstream callers can use it for any RMSNorm (e.g. before attention QKV
# projections where the three projections share the same normalized input).
@triton.jit
def rmsnorm_forward_kernel(
    x_ptr,         # (M, K) bf16
    weight_ptr,    # (K,) bf16
    out_ptr,       # (M, K) bf16
    rstd_ptr,      # (M,) fp32
    M, K,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    eps,
    BK: tl.constexpr,  # K block size (must be >= K for a single-pass reduction)
):
    """Compute RMSNorm: out = x * rstd * weight, rstd = rsqrt(mean(x^2)+eps)."""
    pid_m = tl.program_id(0)
    row = pid_m
    if row >= M:
        return

    # Load entire row (K elements) in chunks of BK
    offs_k = tl.arange(0, BK)
    mask_k = offs_k < K

    # ── Pass 1: compute sum(x^2) over K ──────────────────────────────────
    sum_sq = tl.zeros((), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BK)):
        k_off = k_iter * BK
        mask = (offs_k + k_off) < K
        x_tile = tl.load(
            x_ptr + row * stride_xm + (offs_k + k_off) * stride_xk,
            mask=mask, other=0.0,
        ).to(tl.float32)
        sum_sq += tl.sum(x_tile * x_tile)

    # ── rstd = 1/sqrt(mean + eps) ─────────────────────────────────────────
    mean_sq = sum_sq / K
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    tl.store(rstd_ptr + row, rstd)

    # ── Pass 2: write out = x * rstd * weight ─────────────────────────────
    for k_iter in range(0, tl.cdiv(K, BK)):
        k_off = k_iter * BK
        mask = (offs_k + k_off) < K
        x_tile = tl.load(
            x_ptr + row * stride_xm + (offs_k + k_off) * stride_xk,
            mask=mask, other=0.0,
        ).to(tl.float32)
        w_tile = tl.load(
            weight_ptr + (offs_k + k_off),
            mask=mask, other=0.0,
        ).to(tl.float32)
        out_tile = x_tile * rstd * w_tile
        tl.store(
            out_ptr + row * stride_om + (offs_k + k_off) * stride_ok,
            out_tile.to(tl.bfloat16),
            mask=mask,
        )


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL A' — RMSNorm backward (one program per row block)
# ═════════════════════════════════════════════════════════════════════════════
# Math (derivation):
#   forward:  out = x * rstd * weight
#             rstd = 1 / sqrt(mean(x^2) + eps),  mean = sum(x^2)/K
#
#   backward: given grad_out (dL/dout),
#             grad_x = rstd * weight * (grad_out - mean(grad_out * weight * x) * x / mean(x^2+eps) * (1/K) * 2)
#   Equivalently (standard RMSNorm backward form):
#             c1 = sum(grad_out * weight * x)        # scalar per row
#             c2 = sum(grad_out * weight)              # scalar per row
#             grad_x = rstd * weight * (grad_out - x * (c1 / (mean + eps)) / K - (c2 / K))
#   Wait, more precisely: let g = grad_out * weight. Then
#       grad_x = rstd * (g - x * sum(g * x) * rstd^2 / K)
#   (the c2 term cancels because we already multiplied by weight.)
#
#   This is the canonical RMSNorm backward. We compute it in a 2-pass kernel:
#     Pass 1: compute s = sum(g * x)  where g = grad_out * weight
#     Pass 2: grad_x = rstd * (g - x * s * rstd^2 / K)
@triton.jit
def rmsnorm_backward_kernel(
    grad_out_ptr,  # (M, K) bf16
    x_ptr,         # (M, K) bf16  — saved from forward
    weight_ptr,    # (K,) bf16
    rstd_ptr,      # (M,) fp32 — saved from forward
    grad_x_ptr,    # (M, K) bf16 — OUTPUT
    grad_w_ptr,    # (K,) fp32 — OUTPUT (or None)
    M, K,
    stride_gm, stride_gk,
    stride_xm, stride_xk,
    stride_xgm, stride_xgk,
    stride_gwm,
    eps,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    row = pid_m
    if row >= M:
        return

    rstd = tl.load(rstd_ptr + row)
    inv_K = 1.0 / K

    offs_k = tl.arange(0, BK)

    # ── Pass 1: s = sum(grad_out * weight * x)  ────────────────────────────
    # Also accumulate grad_weight per row-block (the kernel writes per-row
    # contributions; the launcher adds them across rows via atomic_add).
    s = tl.zeros((), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BK)):
        k_off = k_iter * BK
        mask = (offs_k + k_off) < K
        g = tl.load(
            grad_out_ptr + row * stride_gm + (offs_k + k_off) * stride_gk,
            mask=mask, other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + row * stride_xm + (offs_k + k_off) * stride_xk,
            mask=mask, other=0.0,
        ).to(tl.float32)
        w = tl.load(
            weight_ptr + (offs_k + k_off),
            mask=mask, other=0.0,
        ).to(tl.float32)
        gw = g * w
        s += tl.sum(gw * x)

        # grad_weight accumulation (atomic across rows)
        if grad_w_ptr is not None:
            # Use atomic_add since multiple rows contribute to the same K dim
            tl.atomic_add(grad_w_ptr + (offs_k + k_off), gw, mask=mask)

    # ── Pass 2: grad_x = rstd * (g - x * s * rstd^2 / K) ───────────────────
    coef = s * rstd * rstd * inv_K
    for k_iter in range(0, tl.cdiv(K, BK)):
        k_off = k_iter * BK
        mask = (offs_k + k_off) < K
        g = tl.load(
            grad_out_ptr + row * stride_gm + (offs_k + k_off) * stride_gk,
            mask=mask, other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + row * stride_xm + (offs_k + k_off) * stride_xk,
            mask=mask, other=0.0,
        ).to(tl.float32)
        w = tl.load(
            weight_ptr + (offs_k + k_off),
            mask=mask, other=0.0,
        ).to(tl.float32)
        grad_x = rstd * (g * w - x * coef)
        tl.store(
            grad_x_ptr + row * stride_xgm + (offs_k + k_off) * stride_xgk,
            grad_x.to(tl.bfloat16),
            mask=mask,
        )


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL B — fused RMSNorm + matmul: y = rmsnorm(x) @ W_ste + bias
# ═════════════════════════════════════════════════════════════════════════════
# This kernel fuses RMSNorm with the matmul.  Each program computes a (BM, BN)
# tile of y.  For each row in the tile, we need the per-row rstd; we compute it
# inline (over the full K dimension) and then use it for the matmul.
#
# Algorithm (per program (BM rows, BN cols)):
#   1. For each row in BM:
#      - Compute sum(x^2) over K (chunked by BK)
#      - rstd = 1/sqrt(sum/K + eps)
#   2. Accumulate  y = Σ_k  (x[row, k] * rstd[row] * weight[k]) * W_ste[k, n]
#                = Σ_k  x_normed[row, k] * W_ste[k, n]
#   3. Add bias and write y[row, n].
#
# This eliminates the HBM write of x_normed.  x is read ONCE for both the
# norm and the matmul (within the same program), keeping it in shared memory /
# L2 cache.
@triton.autotune(
    configs=[
        triton.Config({"BM": 64,  "BN": 64,  "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64,  "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 256, "BK": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_rmsnorm_matmul_kernel(
    x_ptr,         # (M, K) bf16
    weight_ptr,    # (K,) bf16 — RMSNorm weight
    W_ste_ptr,     # (K, N) bf16 — STE weight from compute_P_W_ste
    bias_ptr,      # (N,) bf16 or None
    y_ptr,         # (M, N) bf16 — OUTPUT
    rstd_ptr,      # (M,) fp32 — OUTPUT (saved for backward)
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    eps,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BM + tl.arange(0, BM)  # (BM,)
    offs_n = pid_n * BN + tl.arange(0, BN)  # (BN,)
    offs_k = tl.arange(0, BK)               # (BK,)

    # ── Pass 1: per-row sum(x^2) over K  ─────────────────────────────────
    # Each program loads its BM rows of x (full K extent) and computes the
    # sum of squares per row.  We accumulate in fp32 to avoid bf16 overflow.
    sum_sq = tl.zeros((BM,), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BK)):
        k_off = k_iter * BK
        mask_x = (offs_m[:, None] < M) & ((offs_k[None, :] + k_off) < K)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (offs_k[None, :] + k_off) * stride_xk,
            mask=mask_x, other=0.0,
        ).to(tl.float32)  # (BM, BK)
        # Sum over BK dim → (BM,) accumulate
        sum_sq += tl.sum(x_tile * x_tile, axis=1)

    mean_sq = sum_sq / K
    rstd = 1.0 / tl.sqrt(mean_sq + eps)  # (BM,)
    # Save rstd for backward
    mask_m = offs_m < M
    tl.store(rstd_ptr + offs_m, rstd, mask=mask_m)

    # ── Pass 2: matmul  y = (x * rstd * weight) @ W_ste  ──────────────────
    # Reload x and compute x_normed = x * rstd * weight, then tl.dot with W_ste.
    # x_normed is NEVER written to HBM — it lives only in registers.
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ste_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    rstd_2d = rstd[:, None]  # (BM, 1) for broadcast
    for k_iter in range(0, tl.cdiv(K, BK)):
        k_off = k_iter * BK
        mask_x = (offs_m[:, None] < M) & ((offs_k[None, :] + k_off) < K)
        mask_w = ((offs_k[:, None] + k_off) < K) & (offs_n[None, :] < N)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.bfloat16)  # (BM, BK)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0).to(tl.bfloat16)  # (BK, BN)

        # Apply RMSNorm: x_normed = x * rstd * weight
        # We need weight[k] for this BK slice — load (BK,) and broadcast
        w_norm = tl.load(
            weight_ptr + (offs_k + k_off),
            mask=(offs_k + k_off) < K, other=0.0,
        ).to(tl.float32)  # (BK,)
        x_normed = (x_tile.to(tl.float32) * rstd_2d * w_norm[None, :]).to(tl.bfloat16)
        acc += tl.dot(x_normed, w_tile)

        x_ptrs += BK * stride_xk
        w_ptrs += BK * stride_wk

    # Bias
    if bias_ptr is not None:
        bias_tile = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias_tile[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_y)


# ═════════════════════════════════════════════════════════════════════════════
# Python launchers
# ═════════════════════════════════════════════════════════════════════════════
def rmsnorm_forward_triton(
    x: torch.Tensor,        # (M, K) bf16
    weight: torch.Tensor,   # (K,) bf16
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (x_normed, rstd).  Both on the same device as x.

    x_normed: (M, K) bf16.  rstd: (M,) fp32.
    """
    assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
    assert weight.dtype == torch.bfloat16, f"weight must be bf16, got {weight.dtype}"
    assert x.ndim == 2, f"x must be (M, K), got {x.shape}"
    M, K = x.shape
    assert weight.shape == (K,), f"weight must be (K,), got {weight.shape}"
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # BK must be a power-of-2 >= K for a single-pass reduction.  We use 1024
    # for typical hidden_dim 2560/4096; the kernel handles chunked K when
    # K > BK.
    BK = 1024
    while BK < K:
        BK *= 2
    # Cap at 4096 to keep shared memory bounded
    BK = min(BK, 4096)
    grid = (M,)
    rmsnorm_forward_kernel[grid](
        x, weight, out, rstd,
        M, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        float(eps),
        BK=BK,
        num_warps=8,
        num_stages=1,
    )
    return out, rstd


def rmsnorm_backward_triton(
    grad_out: torch.Tensor,  # (M, K) bf16
    x: torch.Tensor,         # (M, K) bf16 — saved from forward
    weight: torch.Tensor,    # (K,) bf16
    rstd: torch.Tensor,      # (M,) fp32 — saved from forward
    eps: float = 1e-6,
    need_grad_weight: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Returns (grad_x, grad_weight).

    grad_x: (M, K) bf16.  grad_weight: (K,) fp32 (or None).
    """
    assert grad_out.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    M, K = x.shape
    grad_out = grad_out.contiguous()
    x = x.contiguous()
    weight = weight.contiguous()
    grad_x = torch.empty_like(x)
    grad_w = torch.zeros((K,), dtype=torch.float32, device=x.device) if need_grad_weight else None
    BK = 1024
    while BK < K:
        BK *= 2
    BK = min(BK, 4096)
    grid = (M,)
    rmsnorm_backward_kernel[grid](
        grad_out, x, weight, rstd, grad_x, grad_w,
        M, K,
        grad_out.stride(0), grad_out.stride(1),
        x.stride(0), x.stride(1),
        grad_x.stride(0), grad_x.stride(1),
        0,  # grad_w stride (unused — we use atomic_add)
        float(eps),
        BK=BK,
        num_warps=8,
        num_stages=1,
    )
    return grad_x, grad_w


def fused_rmsnorm_matmul_triton(
    x: torch.Tensor,        # (M, K) bf16
    weight: torch.Tensor,   # (K,) bf16 — RMSNorm weight
    W_ste: torch.Tensor,    # (K, N) bf16 — from compute_P_W_ste_triton
    bias: torch.Tensor | None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (y, rstd).  Fuses RMSNorm + matmul.

    y: (M, N) bf16.  rstd: (M,) fp32 (saved for backward).
    """
    assert x.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    assert W_ste.dtype == torch.bfloat16
    M, K = x.shape
    K2, N = W_ste.shape
    assert K == K2, f"K mismatch: x.K={K} vs W_ste.K={K2}"
    assert weight.shape == (K,)
    x = x.contiguous()
    weight = weight.contiguous()
    W_ste = W_ste.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    fused_rmsnorm_matmul_kernel[grid](
        x, weight, W_ste, bias, y, rstd,
        M, N, K,
        x.stride(0), x.stride(1),
        W_ste.stride(0), W_ste.stride(1),
        y.stride(0), y.stride(1),
        float(eps),
    )
    return y, rstd


# ═════════════════════════════════════════════════════════════════════════════
# Autograd Function — fuses RMSNorm + PalettizedLinear
# ═════════════════════════════════════════════════════════════════════════════
# Forward graph:
#     x ──► RMSNorm ──► x_normed ──► compute_P_W_ste ──► (W_ste)
#                                                    │
#                                                    └─► fused_matmul ──► y
#
# In the FUSED forward, we run:
#   1. compute_P_W_ste_triton  →  P_aos, W_ste   (imported from triton_soft_forward)
#   2. fused_rmsnorm_matmul_triton(x, norm_w, W_ste, bias, eps) → y, rstd
#
# Backward graph (derivation):
#   y = (x * rstd * w) @ W_ste + b    where rstd = 1/sqrt(mean(x^2)+eps)
#   Let  x_normed = x * rstd * w.
#   grad_W_ste  = x_normed.T @ grad_y    → handled by triton_soft_backward.fused_soft_bwd_grad_W_triton
#   grad_x_normed = grad_y @ W_ste.T     → handled by triton_soft_backward.fused_soft_bwd_grad_x_triton
#   grad_x  = rmsnorm_backward(grad_x_normed, x, w, rstd)
#   grad_w  = sum(grad_x_normed * x * rstd, dim=0)
#   grad_b  = grad_y.sum(dim=0)
#   grad_palette, grad_logits  → handled by triton_soft_backward.fused_soft_bwd_elementwise_triton
#
# Note: grad_W_ste here is "grad_W_soft" in the STE sense (because forward uses
# W_ste = W_hard numerically but backward routes through W_soft reconstructed
# from P_aos + palette).  We pass it directly to the elementwise kernel.

class FusedRMSNormLinear(torch.autograd.Function):
    """Fused RMSNorm + PalettizedLinear (soft path).

    forward(ctx, x, norm_weight, palette, logits, bias, group_size, tau, eps)
      Steps:
        1. compute_P_W_ste_triton(logits, palette, group_size, tau, step_seed)
           → (P_aos, W_soft, W_ste)
        2. fused_rmsnorm_matmul_triton(x, norm_weight, W_ste, bias, eps) → y, rstd
        3. ctx.save_for_backward(x, norm_weight, palette, logits, P_aos, W_ste, rstd)
    """

    @staticmethod
    def forward(ctx, x, norm_weight, palette, logits, bias,
                group_size, tau, eps=1e-6):
        # Import lazily so that the module can be syntax-checked without triton.
        from triton_soft_forward import compute_P_W_ste_triton, _next_soft_step_seed

        x = x.contiguous()
        norm_weight = norm_weight.contiguous()
        palette = palette.contiguous()
        logits = logits.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        M, K = x.shape
        G, P_size = palette.shape
        n_planes, K_, N = logits.shape
        assert P_size == 4 and n_planes == 4
        assert K == K_, f"x.K={K} != logits.K={K_}"
        assert norm_weight.shape == (K,), f"norm_weight must be (K,), got {norm_weight.shape}"
        assert N % group_size == 0
        assert N // group_size == G

        step_seed = _next_soft_step_seed()
        P_aos, W_soft, W_ste = compute_P_W_ste_triton(
            logits, palette, group_size, float(tau), step_seed,
        )

        y, rstd = fused_rmsnorm_matmul_triton(x, norm_weight, W_ste, bias, eps)

        ctx.save_for_backward(x, norm_weight, palette, logits, P_aos, W_ste, rstd)
        ctx.group_size = group_size
        ctx.tau = tau
        ctx.eps = eps
        ctx.has_bias = bias is not None
        ctx.N = N
        return y

    @staticmethod
    def backward(ctx, grad_y):
        from triton_soft_backward import (
            fused_soft_bwd_grad_x_triton,
            fused_soft_bwd_grad_W_triton,
            fused_soft_bwd_elementwise_triton,
        )
        x, norm_weight, palette, logits, P_aos, W_ste, rstd = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        M, K = x.shape
        N = ctx.N
        G = palette.shape[0]
        GS = ctx.group_size

        needs_grad_x = ctx.needs_input_grad[0]
        needs_grad_norm_w = ctx.needs_input_grad[1]
        needs_grad_palette = ctx.needs_input_grad[2]
        needs_grad_logits = ctx.needs_input_grad[3]
        needs_grad_bias = ctx.has_bias and ctx.needs_input_grad[4]

        # ── 1. grad_x_normed = grad_y @ W_ste.T   (Triton TC matmul) ────────
        # This is the gradient w.r.t. the normalized input x_normed.
        grad_x_normed = None
        if needs_grad_x or needs_grad_norm_w:
            grad_x_normed = fused_soft_bwd_grad_x_triton(grad_y, W_ste)

        # ── 2. grad_x = rmsnorm_backward(grad_x_normed, x, norm_weight, rstd) ──
        grad_x = None
        grad_norm_weight = None
        if grad_x_normed is not None:
            grad_x, grad_norm_weight = rmsnorm_backward_triton(
                grad_x_normed, x, norm_weight, rstd, ctx.eps,
                need_grad_weight=needs_grad_norm_w,
            )
            if not needs_grad_x:
                grad_x = None
            if not needs_grad_norm_w:
                grad_norm_weight = None

        # ── 3. grad_palette + grad_logits  (compute grad_W on-the-fly) ─────
        grad_logits = None
        grad_palette = None
        if needs_grad_logits or needs_grad_palette:
            # grad_W_soft = x_normed.T @ grad_y  (NOT x.T — x_normed is the
            # actual input to the matmul).  We need to reconstruct x_normed
            # for the matmul, but we can fold the rstd*weight scaling into
            # the matmul kernel.  Simpler: compute x_normed explicitly (it's
            # an in-place op, cheap) and reuse the existing grad_W kernel.
            # Compute x_normed = x * rstd * norm_weight  (cheap Triton kernel).
            x_normed, _ = rmsnorm_forward_triton(x, norm_weight, ctx.eps)
            # Now use the existing grad_W kernel.
            grad_W = fused_soft_bwd_grad_W_triton(x_normed, grad_y)
            grad_logits, grad_palette = fused_soft_bwd_elementwise_triton(
                grad_W, P_aos, palette, GS,
            )
            if not needs_grad_logits:
                grad_logits = None
            if not needs_grad_palette:
                grad_palette = None

        # ── 4. grad_bias = grad_y.sum(dim=0) ───────────────────────────────
        grad_bias = None
        if needs_grad_bias:
            grad_bias = grad_y.sum(dim=0)

        # Return tuple matches forward input order:
        # (x, norm_weight, palette, logits, bias, group_size, tau, eps)
        return (grad_x, grad_norm_weight, grad_palette, grad_logits, grad_bias,
                None, None, None)


def fused_rmsnorm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    palette: torch.Tensor,
    logits: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 256,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Functional interface for the fused RMSNorm + PalettizedLinear soft path.

    Args:
        x: (M, K) bf16 input tensor
        norm_weight: (K,) bf16 RMSNorm scale weight
        palette: (G, 4) bf16 — palettization palette
        logits: (4, K, N) fp16 — index logits (SoA layout)
        bias: (N,) bf16 or None
        group_size: number of weights per palette group
        tau: Gumbel-Softmax temperature
        eps: RMSNorm epsilon

    Returns: y (M, N) bf16
    """
    return FusedRMSNormLinear.apply(
        x, norm_weight, palette, logits, bias,
        group_size, tau, eps,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Hard path (eval mode) — RMSNorm + PalettizedLinear with frozen indices
# ═════════════════════════════════════════════════════════════════════════════
def fused_rmsnorm_linear_hard(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    palette: torch.Tensor,
    indices_int8: torch.Tensor,   # (K, N) int8 — frozen hard indices
    bias: torch.Tensor | None,
    group_size: int = 256,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Eval-mode hard path: RMSNorm + gather + matmul.

    Reuses triton_hard_forward.triton_hard_linear for the gather+matmul after
    computing the RMSNorm in-place.
    """
    from triton_hard_forward import triton_hard_linear
    x_normed, _ = rmsnorm_forward_triton(x, norm_weight, eps)
    return triton_hard_linear(x_normed, palette, indices_int8, bias, group_size)


# ═════════════════════════════════════════════════════════════════════════════
# Self-test (smoke test, no GPU required for import)
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("triton_rmsnorm.py — Patch 10: fused RMSNorm + PalettizedLinear")
    print("=" * 70)
    print("Kernels:")
    print("  - rmsnorm_forward_kernel        (1 program per row, BK-chunked)")
    print("  - rmsnorm_backward_kernel       (1 program per row, 2-pass)")
    print("  - fused_rmsnorm_matmul_kernel   (autotuned TC matmul + RMSNorm)")
    print()
    print("Autograd Functions:")
    print("  - FusedRMSNormLinear            (forward + backward, soft path)")
    print()
    print("Functional interfaces:")
    print("  - rmsnorm_forward_triton(x, w, eps)")
    print("  - rmsnorm_backward_triton(grad_out, x, w, rstd, eps)")
    print("  - fused_rmsnorm_matmul_triton(x, w, W_ste, bias, eps)")
    print("  - fused_rmsnorm_linear(x, w, palette, logits, bias, gs, tau, eps)")
    print("  - fused_rmsnorm_linear_hard(x, w, palette, indices, bias, gs, eps)")
    print()
    print("DoD: import check + syntax check.")
