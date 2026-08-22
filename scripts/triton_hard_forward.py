"""Triton fused kernel for PalettizedLinear HARD forward (eval mode).

WAVE 3 — fused_hard_forward:
  - `compute_hard_W_kernel`: per-(j, o) tile kernel that gathers
    W[j, o] = palette[g, indices[j, o]]  where g = o // group_size
    Stores the gathered W (K, N) bf16 — used by the matmul kernel.
  - `fused_hard_matmul_kernel`: standard Triton TC matmul, y = x @ W + bias.
    (Shares the autotuned configs with the soft matmul.)
  - `TritonHardLinear` (torch.autograd.Function): wires the two kernels.
    Eval mode = no autograd needed (just torch.no_grad in eval call site),
    but the Function is here so we can .apply() with proper context management.

Layouts (must match existing CUDA path):
  x:        (M, K) bf16
  palette:  (G, 4) bf16
  indices:  (K, N) int8 — frozen hard indices (from argmax of trained logits)
  bias:     (N,) bf16 or None
  y:        (M, N) bf16
  W:        (K, N) bf16 — intermediate (gathered from palette + indices)
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 3a — compute_hard_W: gather W = palette[g, indices[j, o]]
# ═════════════════════════════════════════════════════════════════════════════
# For each (j, o):
#   g = o // group_size
#   W[j, o] = palette[g, indices[j, o]]   (indices is int8 in [0, 3])
# We use Triton's tl.load with the gather pattern: build the flat offset
# g * 4 + indices[j, o] into the (G, 4) palette tensor, then load.
@triton.autotune(
    configs=[
        triton.Config({"BM": 16, "BN": 16}, num_warps=4, num_stages=1),
        triton.Config({"BM": 32, "BN": 16}, num_warps=4, num_stages=1),
        triton.Config({"BM": 16, "BN": 32}, num_warps=4, num_stages=1),
        triton.Config({"BM": 32, "BN": 32}, num_warps=4, num_stages=1),
        triton.Config({"BM": 64, "BN": 32}, num_warps=8, num_stages=1),
        triton.Config({"BM": 32, "BN": 64}, num_warps=8, num_stages=1),
        triton.Config({"BM": 64, "BN": 64}, num_warps=8, num_stages=1),
    ],
    key=["K", "N"],
)
@triton.jit
def compute_hard_W_kernel(
    palette_ptr,    # (G, 4) bf16
    indices_ptr,    # (K, N) int8
    W_ptr,          # (K, N) bf16 — OUTPUT
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
    idx_flat = j_grid * N + o_grid  # (BM, BN) — flat (K, N) index

    # ── Load indices[j, o] (int8) and convert to int32 ─────────────────────
    idx_val = tl.load(indices_ptr + idx_flat, mask=mask, other=0).to(tl.int32)  # (BM, BN)

    # ── Compute group index g = o // group_size ───────────────────────────
    g_grid = o_grid // group_size  # (1, BN)

    # ── Gather palette[g, idx] where palette is (G, 4) bf16 ──────────────
    # Flat offset into palette: g * 4 + idx_val (since palette[g, k] at g*4+k)
    pal_off = g_grid * 4 + idx_val  # (BM, BN) — broadcast g across rows
    # Load W[j, o] = palette[g, idx_val[j, o]]
    W_val = tl.load(palette_ptr + pal_off, mask=mask, other=0.0)  # bf16
    tl.store(W_ptr + idx_flat, W_val, mask=mask)


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 3b — fused hard matmul: y = x @ W + bias (Triton TC matmul)
# ═════════════════════════════════════════════════════════════════════════════
# Same structure as fused_soft_matmul_kernel — repeated here for self-contained
# module (no shared kernel across soft/hard paths).
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
def fused_hard_matmul_kernel(
    x_ptr, W_ptr, bias_ptr,
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
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BK)):
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] + k_iter * BK < K)
        mask_w = (offs_k[:, None] + k_iter * BK < K) & (offs_n[None, :] < N)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
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
def compute_hard_W_triton(
    palette: torch.Tensor,    # (G, 4) bf16
    indices: torch.Tensor,    # (K, N) int8
    group_size: int,
) -> torch.Tensor:
    """Returns W (K, N) bf16 — gathered from palette using indices."""
    assert palette.dtype == torch.bfloat16
    assert indices.dtype == torch.int8
    K, N = indices.shape
    G, P4 = palette.shape
    assert P4 == 4
    assert N // group_size == G
    palette = palette.contiguous()
    indices = indices.contiguous()
    W = torch.empty((K, N), dtype=torch.bfloat16, device=palette.device)
    grid = lambda meta: (triton.cdiv(K, meta["BM"]), triton.cdiv(N, meta["BN"]))
    compute_hard_W_kernel[grid](
        palette, indices, W,
        K, N, G,
        group_size=group_size,
    )
    return W


def fused_hard_matmul_triton(
    x: torch.Tensor,    # (M, K) bf16
    W: torch.Tensor,    # (K, N) bf16
    bias: torch.Tensor | None,
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    assert W.dtype == torch.bfloat16
    M, K = x.shape
    K2, N = W.shape
    assert K == K2
    x = x.contiguous()
    W = W.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    fused_hard_matmul_kernel[grid](
        x, W, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


# ═════════════════════════════════════════════════════════════════════════════
# Autograd Function — eval mode, no backward needed
# ═════════════════════════════════════════════════════════════════════════════
class TritonHardLinear(torch.autograd.Function):
    """Triton-only fused hard forward (eval mode).

    forward(ctx, x, palette, indices, bias, group_size) -> y
      Steps:
        1. compute_hard_W_triton(palette, indices, group_size) → W (K, N) bf16
        2. fused_hard_matmul_triton(x, W, bias) → y (M, N) bf16
      No saved ctx — eval mode does not need backward.

    backward: returns Nones (hard path is for inference; gradients flow only
    through the soft path during training).
    """

    @staticmethod
    def forward(ctx, x, palette, indices, bias, group_size):
        x = x.contiguous()
        palette = palette.contiguous()
        indices = indices.to(torch.int8).contiguous()
        if bias is not None:
            bias = bias.contiguous()

        M, K = x.shape
        G, P_size = palette.shape
        K_, N = indices.shape
        assert P_size == 4
        assert K == K_, f"x K={K} != indices K={K_}"
        assert N % group_size == 0
        assert N // group_size == G

        W = compute_hard_W_triton(palette, indices, group_size)
        y = fused_hard_matmul_triton(x, W, bias)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # Eval mode: no gradients flow through the hard path. The soft path
        # (TritonSoftLinear) handles training. Return Nones.
        return None, None, None, None, None


def triton_hard_linear(
    x: torch.Tensor,
    palette: torch.Tensor,
    indices: torch.Tensor,
    bias: torch.Tensor | None = None,
    group_size: int = 256,
) -> torch.Tensor:
    """Functional interface — matches `fused_lut_linear` (hard) signature."""
    return TritonHardLinear.apply(x, palette, indices, bias, group_size)
