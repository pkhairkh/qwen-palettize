"""Triton reference implementation of the fused 2-bit LUT-quantized linear layer.

This is the CORRECTNESS ORACLE for the CUDA production kernel. It must:
  1. Match the reference PyTorch implementation (reference_pytorch.py) to
     within atol=1e-3, rtol=1e-3 on forward and backward.
  2. Be reasonably performant (1.5-3× slower than the CUDA production kernel
     on sm_89 is fine — the goal is correctness, not peak perf).

Layout assumptions:
  - x:         (M, K) bf16, contiguous row-major
  - palette:   (G, 4) bf16, contiguous
  - indices:   (K, N) int8 (values 0-3), contiguous row-major
  - bias:      (N,) bf16 or None
  - group_size = 256 (always, per spec)
  - G = N // group_size

Forward:  W[j, o] = palette[o // group_size, indices[j, o]]
          y = x @ W + bias
Backward: grad_x[i, j]  = Σ_o grad_y[i, o] * W[j, o]
          grad_palette[g, k] = Σ_(j, o where o//gs=g, indices[j,o]=k) Σ_i x[i, j] * grad_y[i, o]
          grad_bias[o]      = Σ_i grad_y[i, o]
"""
from __future__ import annotations
import torch
from torch import Tensor
import triton
import triton.language as tl


GROUP_SIZE = 256


# ──────────────────────────────────────────────────────────────────────────────
# Forward kernel
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fwd_kernel(
    x_ptr, palette_ptr, indices_ptr, bias_ptr, y_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_yo, stride_yn,
    GS: tl.constexpr,                # group_size = 256
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)            # (BM,)
    offs_n = pid_n * BN + tl.arange(0, BN)            # (BN,)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # ── Iterate over K in chunks of BK ─────────────────────────────────────
    for k_start in range(0, K, BK):
        offs_k = k_start + tl.arange(0, BK)           # (BK,)
        mask_k = offs_k < K

        # Load x tile: (BM, BK) bf16
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Load indices tile: (BK, BN) int8
        idx_ptrs = indices_ptr + offs_k[:, None] * N + offs_n[None, :]
        idx_tile = tl.load(
            idx_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.int32)

        # Compute palette offsets: palette has shape (G, 4), flat (G*4,)
        # group = offs_n // GS, then offset = group * 4 + idx
        g = offs_n // GS                                # (BN,) int32
        p_off = g[None, :] * 4 + idx_tile               # (BK, BN) int32

        # Load W tile via gather from palette: (BK, BN) bf16
        w_tile = tl.load(palette_ptr + p_off)

        # Accumulate: acc += x_tile @ w_tile
        acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    # ── Add bias ───────────────────────────────────────────────────────────
    if USE_BIAS:
        b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += b[None, :]

    # ── Store y ─────────────────────────────────────────────────────────────
    y_ptrs = y_ptr + offs_m[:, None] * stride_yo + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


# ──────────────────────────────────────────────────────────────────────────────
# Backward: grad_x = grad_y @ W.T
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _bwd_grad_x_kernel(
    grad_y_ptr, palette_ptr, indices_ptr, grad_x_ptr,
    M, K, N,
    stride_gym, stride_gyn,
    stride_gxm, stride_gxk,
    GS: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
):
    """grad_x[i, j] = Σ_o grad_y[i, o] * W[j, o]

    Same shape as forward but transposed: tile is (BM rows of M, BK cols of K) and
    we reduce over N in chunks of BN.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)            # (BM,)
    offs_k = pid_k * BK + tl.arange(0, BK)           # (BK,)
    mask_m = offs_m < M
    mask_k = offs_k < K

    acc = tl.zeros((BM, BK), dtype=tl.float32)

    for n_start in range(0, N, BN):
        offs_n = n_start + tl.arange(0, BN)
        mask_n = offs_n < N

        # Load grad_y tile: (BM, BN) bf16
        gy_ptrs = grad_y_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
        gy_tile = tl.load(
            gy_ptrs,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )

        # Load indices tile: (BK, BN) int8
        idx_ptrs = indices_ptr + offs_k[:, None] * N + offs_n[None, :]
        idx_tile = tl.load(
            idx_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.int32)

        # Compute palette offsets and gather W tile: (BK, BN) bf16
        g = offs_n // GS                                # (BN,) int32
        p_off = g[None, :] * 4 + idx_tile               # (BK, BN)
        w_tile = tl.load(palette_ptr + p_off)

        # grad_x += grad_y @ w_tile.T
        # tl.trans(w_tile) is (BN, BK); tl.dot(gy, w_T) is (BM, BK)
        acc += tl.dot(gy_tile, tl.trans(w_tile), out_dtype=tl.float32)

    # Store grad_x: (BM, BK) bf16
    gx_ptrs = grad_x_ptr + offs_m[:, None] * stride_gxm + offs_k[None, :] * stride_gxk
    tl.store(gx_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_k[None, :])


# ──────────────────────────────────────────────────────────────────────────────
# Backward: grad_palette = scatter_add(x.T @ grad_y, _flat_idx)
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _bwd_grad_palette_kernel(
    x_ptr, grad_y_ptr, indices_ptr, grad_palette_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_gym, stride_gyn,
    GS: tl.constexpr,
    BM_K: tl.constexpr, BN_N: tl.constexpr, MM_M: tl.constexpr,
):
    """Fused dW computation + scatter_add into grad_palette.

    For each (BM_K, BN_N) tile of (K, N) space:
      1. Compute dW[j, o] = Σ_i x[i, j] * grad_y[i, o]  by chunking over M
      2. Look up (group, idx) per (j, o) and atomic_add to grad_palette

    grad_palette is fp32 (4-byte aligned, native atomicAdd support).
    """
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k = pid_k * BM_K + tl.arange(0, BM_K)        # (BM_K,) rows of K (j dimension)
    offs_n = pid_n * BN_N + tl.arange(0, BN_N)        # (BN_N,) cols of N (o dimension)
    mask_k = offs_k < K
    mask_n = offs_n < N

    # Load indices tile ONCE: (BM_K, BN_N) int8 → int32
    idx_ptrs = indices_ptr + offs_k[:, None] * N + offs_n[None, :]
    idx_tile = tl.load(
        idx_ptrs,
        mask=mask_k[:, None] & mask_n[None, :],
        other=0,
    ).to(tl.int32)

    # dW accumulator: (BM_K, BN_N) fp32
    dW = tl.zeros((BM_K, BN_N), dtype=tl.float32)

    # Chunk over M
    for m_start in range(0, M, MM_M):
        offs_m = m_start + tl.arange(0, MM_M)
        mask_m = offs_m < M

        # Load x_chunk: (MM_M, BM_K) bf16 — pick columns [offs_k] for each row
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_chunk = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Load grad_y_chunk: (MM_M, BN_N) bf16
        gy_ptrs = grad_y_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
        gy_chunk = tl.load(
            gy_ptrs,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )

        # dW += x_chunk.T @ gy_chunk  →  (BM_K, BN_N)
        dW += tl.dot(tl.trans(x_chunk), gy_chunk, out_dtype=tl.float32)

    # ── Scatter_add dW into grad_palette (fp32) ────────────────────────────
    # group = offs_n // GS, offset = group * 4 + idx
    g = offs_n // GS                                    # (BN_N,) int32
    p_off = g[None, :] * 4 + idx_tile                   # (BM_K, BN_N) int32

    # Atomic add — supported on fp32 in Triton (>=2.0)
    full_mask = mask_k[:, None] & mask_n[None, :]
    tl.atomic_add(grad_palette_ptr + p_off, dW, mask=full_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Backward: grad_bias = grad_y.sum(dim=0)
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _bwd_grad_bias_kernel(
    grad_y_ptr, grad_bias_ptr,
    M, N,
    stride_gym, stride_gyn,
    BN: tl.constexpr,
    BM_CHUNK: tl.constexpr,
):
    """grad_bias[o] = Σ_i grad_y[i, o]

    One program per BN output columns; one warp reduces over M in chunks.
    """
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    mask_n = offs_n < N

    acc = tl.zeros((BN,), dtype=tl.float32)

    for m_start in range(0, M, BM_CHUNK):
        offs_m = m_start + tl.arange(0, BM_CHUNK)
        mask_m = offs_m < M
        gy_ptrs = grad_y_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
        gy_chunk = tl.load(gy_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        acc += tl.sum(gy_chunk.to(tl.float32), axis=0)

    tl.store(grad_bias_ptr + offs_n, acc.to(tl.bfloat16), mask=mask_n)


# ──────────────────────────────────────────────────────────────────────────────
# Public API + autograd Function
# ──────────────────────────────────────────────────────────────────────────────

def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def triton_lut_linear_forward(
    x: Tensor,
    palette: Tensor,
    indices: Tensor,
    bias: Tensor | None,
    group_size: int = GROUP_SIZE,
) -> Tensor:
    M, K = x.shape
    _, N = indices.shape
    G, P = palette.shape
    assert P == 4 and N % group_size == 0 and N // group_size == G

    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    # Tile sizes — tuned for typical K∈[2560, 9216], N∈[1024, 9216], M≈1024
    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fwd_kernel[grid](
        x, palette, indices, bias if bias is not None else x,  # dummy for bias ptr when None
        y,
        M, K, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        GS=group_size,
        BM=BM, BN=BN, BK=BK,
        USE_BIAS=(bias is not None),
    )
    return y


def triton_lut_linear_backward(
    grad_y: Tensor,
    x: Tensor,
    palette: Tensor,
    indices: Tensor,
    bias: Tensor | None,
    group_size: int = GROUP_SIZE,
    needs_grad_x: bool = True,
    needs_grad_palette: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None]:
    M, K = x.shape
    _, N = indices.shape
    G, P = palette.shape

    grad_x = torch.empty_like(x) if needs_grad_x else None
    grad_palette_fp32 = (
        torch.zeros((G, P), dtype=torch.float32, device=x.device)
        if needs_grad_palette else None
    )
    grad_bias = torch.empty(N, dtype=torch.bfloat16, device=x.device) if bias is not None else None

    # ── grad_x kernel ──────────────────────────────────────────────────────
    if needs_grad_x:
        BM, BK, BN = 64, 32, 64
        grid_gx = (triton.cdiv(M, BM), triton.cdiv(K, BK))
        _bwd_grad_x_kernel[grid_gx](
            grad_y, palette, indices, grad_x,
            M, K, N,
            grad_y.stride(0), grad_y.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            GS=group_size,
            BM=BM, BK=BK, BN=BN,
        )

    # ── grad_palette kernel ─────────────────────────────────────────────────
    if needs_grad_palette:
        # For grad_palette, tile (BM_K, BN_N) of (K, N), chunk over M
        BM_K, BN_N, MM_M = 64, 64, 64
        grid_gp = (triton.cdiv(K, BM_K), triton.cdiv(N, BN_N))
        _bwd_grad_palette_kernel[grid_gp](
            x, grad_y, indices, grad_palette_fp32,
            M, K, N,
            x.stride(0), x.stride(1),
            grad_y.stride(0), grad_y.stride(1),
            GS=group_size,
            BM_K=BM_K, BN_N=BN_N, MM_M=MM_M,
        )
        grad_palette = grad_palette_fp32.to(torch.bfloat16)
    else:
        grad_palette = None

    # ── grad_bias kernel ────────────────────────────────────────────────────
    if bias is not None:
        BN, BM_CHUNK = 64, 128
        grid_gb = (triton.cdiv(N, BN),)
        _bwd_grad_bias_kernel[grid_gb](
            grad_y, grad_bias,
            M, N,
            grad_y.stride(0), grad_y.stride(1),
            BN=BN, BM_CHUNK=BM_CHUNK,
        )

    return grad_x, grad_palette, grad_bias


class TritonFusedLUTLinear(torch.autograd.Function):
    """Autograd Function wrapping the Triton kernels."""

    @staticmethod
    def forward(ctx, x, palette, indices, bias, group_size):
        M, K = x.shape
        _, N = indices.shape
        y = triton_lut_linear_forward(x, palette, indices, bias, group_size)
        ctx.save_for_backward(x, palette, indices)
        ctx.group_size = group_size
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, palette, indices = ctx.saved_tensors
        grad_x, grad_palette, grad_bias = triton_lut_linear_backward(
            grad_y, x, palette, indices,
            bias=None,  # we recover grad_bias via ctx.has_bias
            group_size=ctx.group_size,
            needs_grad_x=ctx.needs_input_grad[0],
            needs_grad_palette=ctx.needs_input_grad[1],
        )
        if ctx.has_bias:
            # Re-run grad_bias (it was skipped above because we passed bias=None)
            M, K = x.shape
            _, N = indices.shape
            grad_bias = torch.empty(N, dtype=torch.bfloat16, device=x.device)
            BN, BM_CHUNK = 64, 128
            grid = (triton.cdiv(N, BN),)
            _bwd_grad_bias_kernel[grid](
                grad_y, grad_bias,
                M, N,
                grad_y.stride(0), grad_y.stride(1),
                BN=BN, BM_CHUNK=BM_CHUNK,
            )
        else:
            grad_bias = None
        # indices has no grad; group_size is a python int (no grad)
        return grad_x, grad_palette, None, grad_bias, None


def triton_lut_linear(
    x: Tensor,
    palette: Tensor,
    indices: Tensor,
    bias: Tensor | None = None,
    group_size: int = GROUP_SIZE,
) -> Tensor:
    """Functional interface — supports autograd."""
    return TritonFusedLUTLinear.apply(x, palette, indices, bias, group_size)
