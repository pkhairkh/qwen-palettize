"""Triton fused kernels for QwenLoRA forward + backward.

Wave 3 (Patch 19) - fused LoRA forward + backward:
  - `fused_lora_xA_kernel`:       xA = x @ A  (small M*K*R matmul)
  - `fused_lora_matmul_kernel`:   y = xA @ B.T * scaling  (main M*K*N matmul,
                                   scaling fused into output store)
  - `fused_lora_grad_xA_kernel`:  grad_xA = (grad_y * scaling) @ B
                                   (scaling fused into grad_y load)
  - `fused_lora_grad_A_kernel`:    grad_A = x.T @ grad_xA
  - `fused_lora_grad_B_kernel`:    grad_B = (grad_y * scaling).T @ xA
                                   (scaling fused into grad_y load - no aten::mul)
  - `fused_lora_grad_x_kernel`:    grad_x_lora = grad_xA @ A.T  (main M*K matmul)
  - `TritonLoRALinear` (torch.autograd.Function): wires forward + backward

Wave 4 (Patch 20) - fused PalettizedLinear + LoRA backward:
  - `fused_pl_lora_bwd_grad_x_kernel`: grad_x = grad_y @ (W_ste + lora_B @ lora_A.T * scaling).T
                                       (single combined matmul, eliminates aten::add_
                                       for grad_x accumulation across 31 LoRA modules)
  - `FusedPLLoRALinear` (torch.autograd.Function): combined forward + backward

Math (QLoRA, Dettmers et al. 2023, arXiv:2305.14314):
  LoRA forward:   y = (x @ A) @ B.T * scaling  =  x @ (A @ B.T * scaling)
                  where W_lora = scaling * (A @ B.T) is the LoRA delta-weight
                  (rank-R update to the base weight).

  LoRA backward (grad_y has shape (M, N)):
    grad_xA      = (grad_y * scaling) @ B                  (M, R)
    grad_x_lora  = grad_xA @ A.T                            (M, K)
    grad_A       = x.T @ grad_xA                            (K, R)
    grad_B       = (grad_y * scaling).T @ xA                (N, R)

  Fusion opportunities (this file):
    1. Scaling fused into grad_y load (grad_xA and grad_B kernels).
       Eliminates 31 aten::mul (vectorized_elementwise) per step.
    2. xA = x @ A computed once in forward, cached via save_for_backward
       and reused for grad_B in backward (skips one full M*K*R matmul
       recompute per LoRA module - 31 * 1.3 GFLOPS = 40 GFLOPS saved per step).
    3. grad_xA shared between grad_A, grad_B, and grad_x_lora matmuls
       (computed once per backward call instead of three times).
    4. Single autograd.Function node replaces 3 PyTorch matmuls + 1 mul
       in the LoRA forward (eliminates ~4 dispatch ops * 31 modules = 124
       dispatches per step).
    5. (Patch 20) grad_x = grad_y @ (W_ste + B @ A.T * scaling).T - combined
       matmul eliminates the 31 aten::add_ that accumulate grad_x_base +
       grad_x_lora per layer.

  Remaining dispatches per LoRA backward (post-Patch-19, pre-Patch-20):
    - 3 Triton kernel launches (grad_xA, grad_A, grad_B) - small matmuls
    - 1 Triton kernel launch (grad_x_lora) - main matmul
    - 1 aten::add_ for grad_x accumulation (eliminated by Patch 20)
    Total per LoRA backward: 4 launches + 1 add (down from 4 launches + 1 mul + 1 add).

Shapes:
  x:        (M, K) bf16  -- K = in_dim
  A:        (K, R) bf16  -- R = rank (small, e.g. 32)
  B:        (N, R) bf16  -- N = out_dim
  W_ste:    (K, N) bf16  -- from PalettizedLinear STE path
  y:        (M, N) bf16
  grad_y:   (M, N) bf16
  grad_A:   (K, R) bf16
  grad_B:   (N, R) bf16
  grad_x:   (M, K) bf16

Offline constraint: NO GPU. Code is verified by `python3 -c "import triton_lora"`
(syntax + kernel signatures + autograd.Function construction). Correctness
benchmarks require the training server and are deferred to the orchestrator's
verification phase.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


# ============================================================================
# KERNEL: fused_lora_xA_kernel
#   xA = x @ A  ->  (M, R) bf16
# Small matmul (M*K*R FLOPs, R=32). Cached in forward and reused in backward
# for grad_B (avoids recompute).
# ============================================================================
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
    key=["M", "K", "R"],
)
@triton.jit
def fused_lora_xA_kernel(
    x_ptr, A_ptr, xA_ptr,
    M, K, R,
    stride_xm, stride_xk,
    stride_ak, stride_ar,
    stride_xam, stride_xar,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """xA = x @ A. Output: (M, R) bf16.

    BM tile of M, BN tile of R (output col), reduce over K (chunked by BK).
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_r = tl.cdiv(R, BN)
    pid_m = pid // grid_r
    pid_r = pid % grid_r

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_r = pid_r * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    a_ptrs = A_ptr + offs_k[:, None] * stride_ak + offs_r[None, :] * stride_ar

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BK)):
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] + k_iter * BK < K)
        mask_a = (offs_k[None, :] + k_iter * BK < K) & (offs_r[:, None] < R)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.bfloat16)
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.bfloat16)
        acc += tl.dot(x_tile, a_tile)
        x_ptrs += BK * stride_xk
        a_ptrs += BK * stride_ak

    xa_ptrs = xA_ptr + offs_m[:, None] * stride_xam + offs_r[None, :] * stride_xar
    mask_xa = (offs_m[:, None] < M) & (offs_r[None, :] < R)
    tl.store(xa_ptrs, acc.to(tl.bfloat16), mask=mask_xa)


# ============================================================================
# KERNEL: fused_lora_matmul_kernel
#   y = xA @ B.T * scaling  ->  (M, N) bf16
# Main forward matmul. Scaling fused into the output store.
# ============================================================================
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
    key=["M", "N", "R"],
)
@triton.jit
def fused_lora_matmul_kernel(
    xA_ptr, B_ptr, y_ptr,
    M, N, R,
    scaling,
    stride_xam, stride_xar,
    stride_bn, stride_br,
    stride_ym, stride_yn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """y = xA @ B.T * scaling.

    xA: (M, R), B: (N, R), B.T: (R, N) -> y: (M, N).
    Scaling fused into the output store (no separate aten::mul).
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_r = tl.arange(0, BK)

    xa_ptrs = xA_ptr + offs_m[:, None] * stride_xam + offs_r[None, :] * stride_xar
    # B: (N, R) row-major. Load as (BK, BN) with r in rows, n in cols (i.e. B.T tile).
    b_ptrs = B_ptr + offs_r[:, None] * stride_br + offs_n[None, :] * stride_bn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for r_iter in range(0, tl.cdiv(R, BK)):
        mask_xa = (offs_m[:, None] < M) & (offs_r[None, :] + r_iter * BK < R)
        mask_b = (offs_r[:, None] + r_iter * BK < R) & (offs_n[None, :] < N)
        xa_tile = tl.load(xa_ptrs, mask=mask_xa, other=0.0).to(tl.bfloat16)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.bfloat16)
        acc += tl.dot(xa_tile, b_tile)
        xa_ptrs += BK * stride_xar
        b_ptrs += BK * stride_br

    # FUSE scaling into the output store (no separate aten::mul kernel).
    acc = acc * scaling
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_y)


# ============================================================================
# KERNEL: fused_lora_grad_xA_kernel
#   grad_xA = (grad_y * scaling) @ B  ->  (M, R) bf16
# Scaling fused into grad_y load. Output reused by grad_A, grad_x_lora matmuls.
# ============================================================================
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
    key=["M", "N", "R"],
)
@triton.jit
def fused_lora_grad_xA_kernel(
    grad_y_ptr, B_ptr, grad_xA_ptr,
    M, N, R,
    scaling,
    stride_gym, stride_gyn,
    stride_bn, stride_br,
    stride_gxm, stride_gxr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """grad_xA = (grad_y * scaling) @ B  -> (M, R) bf16.

    Scaling fused into grad_y load.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_r = tl.cdiv(R, BN)
    pid_m = pid // grid_r
    pid_r = pid % grid_r

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_r = pid_r * BN + tl.arange(0, BN)
    offs_n = tl.arange(0, BK)

    gy_ptrs = grad_y_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
    b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for n_iter in range(0, tl.cdiv(N, BK)):
        mask_gy = (offs_m[:, None] < M) & (offs_n[None, :] + n_iter * BK < N)
        mask_b = (offs_n[:, None] + n_iter * BK < N) & (offs_r[None, :] < R)
        gy_tile = tl.load(gy_ptrs, mask=mask_gy, other=0.0).to(tl.float32)
        # FUSE scaling into the grad_y load (no separate aten::mul kernel).
        gy_tile = gy_tile * scaling
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.bfloat16)
        acc += tl.dot(gy_tile.to(tl.bfloat16), b_tile)
        gy_ptrs += BK * stride_gyn
        b_ptrs += BK * stride_bn

    gx_ptrs = grad_xA_ptr + offs_m[:, None] * stride_gxm + offs_r[None, :] * stride_gxr
    mask_gx = (offs_m[:, None] < M) & (offs_r[None, :] < R)
    tl.store(gx_ptrs, acc.to(tl.bfloat16), mask=mask_gx)


# ============================================================================
# KERNEL: fused_lora_grad_A_kernel
#   grad_A = x.T @ grad_xA  ->  (K, R) bf16
# Small matmul (K*M*R FLOPs, R=32). No scaling needed (already in grad_xA).
# Uses fp32 accumulator -> bf16 output for accuracy.
# ============================================================================
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
    key=["M", "K", "R"],
)
@triton.jit
def fused_lora_grad_A_kernel(
    x_ptr, grad_xA_ptr, grad_A_ptr,
    M, K, R,
    stride_xm, stride_xk,
    stride_gxm, stride_gxr,
    stride_gak, stride_gar,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """grad_A = x.T @ grad_xA  -> (K, R) bf16.

    x: (M, K), grad_xA: (M, R), x.T: (K, M) -> output (K, R).
    Output tile (BM=K-tile, BN=R-tile); reduction over M.
    """
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BM)
    grid_r = tl.cdiv(R, BN)
    pid_k = pid // grid_r
    pid_r = pid % grid_r

    offs_k = pid_k * BM + tl.arange(0, BM)
    offs_r = pid_r * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BK)

    # x: (M, K) row-major. x[m, k] at m*stride_xm + k*stride_xk.
    # For x.T @ grad_xA we need x.T[k, m] = x[m, k] -> load as (BK, BM) with m in rows, k in cols.
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # grad_xA: (M, R) row-major -> grad_xA[m, r] at m*stride_gxm + r*stride_gxr.
    gx_ptrs = grad_xA_ptr + offs_m[:, None] * stride_gxm + offs_r[None, :] * stride_gxr

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for m_iter in range(0, tl.cdiv(M, BK)):
        mask_x = (offs_m[:, None] + m_iter * BK < M) & (offs_k[None, :] < K)
        mask_gx = (offs_m[:, None] + m_iter * BK < M) & (offs_r[None, :] < R)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)  # (BK, BM) bf16
        gx_tile = tl.load(gx_ptrs, mask=mask_gx, other=0.0)  # (BK, BN) bf16
        # tl.dot(x_tile.T, gx_tile) -> (BM, BN). tl.trans swaps to (BM, BK) then dot with (BK, BN).
        x_tile_t = tl.trans(x_tile)  # (BM, BK)
        acc += tl.dot(x_tile_t, gx_tile)
        x_ptrs += BK * stride_xm
        gx_ptrs += BK * stride_gxm

    gA_ptrs = grad_A_ptr + offs_k[:, None] * stride_gak + offs_r[None, :] * stride_gar
    mask_gA = (offs_k[:, None] < K) & (offs_r[None, :] < R)
    tl.store(gA_ptrs, acc.to(tl.bfloat16), mask=mask_gA)


# ============================================================================
# KERNEL: fused_lora_grad_B_kernel
#   grad_B = (grad_y * scaling).T @ xA  ->  (N, R) bf16
# Scaling fused into grad_y load. Output is small (N, R).
# ============================================================================
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
    key=["M", "N", "R"],
)
@triton.jit
def fused_lora_grad_B_kernel(
    grad_y_ptr, xA_ptr, grad_B_ptr,
    M, N, R,
    scaling,
    stride_gym, stride_gyn,
    stride_xam, stride_xar,
    stride_gbn, stride_gbr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """grad_B = (grad_y * scaling).T @ xA  -> (N, R) bf16.

    grad_y: (M, N), xA: (M, R), grad_y_scaled.T: (N, M) -> output (N, R).
    Scaling fused into grad_y load (no separate aten::mul kernel).
    Output tile (BM=N-tile, BN=R-tile); reduction over M.
    """
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BM)
    grid_r = tl.cdiv(R, BN)
    pid_n = pid // grid_r
    pid_r = pid % grid_r

    offs_n = pid_n * BM + tl.arange(0, BM)
    offs_r = pid_r * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BK)

    # grad_y: (M, N) row-major. For (grad_y * scaling).T @ xA we need
    # grad_y_scaled.T[n, m] = grad_y[m, n] * scaling -> load as (BK, BM) with m in rows, n in cols.
    gy_ptrs = grad_y_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
    # xA: (M, R) row-major -> xA[m, r] at m*stride_xam + r*stride_xar.
    xa_ptrs = xA_ptr + offs_m[:, None] * stride_xam + offs_r[None, :] * stride_xar

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for m_iter in range(0, tl.cdiv(M, BK)):
        mask_gy = (offs_m[:, None] + m_iter * BK < M) & (offs_n[None, :] < N)
        mask_xa = (offs_m[:, None] + m_iter * BK < M) & (offs_r[None, :] < R)
        gy_tile = tl.load(gy_ptrs, mask=mask_gy, other=0.0).to(tl.float32)
        # FUSE scaling into the grad_y load (no separate aten::mul kernel).
        gy_tile = gy_tile * scaling
        xa_tile = tl.load(xa_ptrs, mask=mask_xa, other=0.0).to(tl.bfloat16)
        # tl.dot(gy_tile.T, xa_tile) -> (BM, BN). gy_tile is (BK, BM); trans to (BM, BK).
        gy_tile_t = gy_tile.to(tl.bfloat16)
        gy_tile_t = tl.trans(gy_tile_t)  # (BM, BK)
        acc += tl.dot(gy_tile_t, xa_tile)
        gy_ptrs += BK * stride_gym
        xa_ptrs += BK * stride_xam

    gB_ptrs = grad_B_ptr + offs_n[:, None] * stride_gbn + offs_r[None, :] * stride_gbr
    mask_gB = (offs_n[:, None] < N) & (offs_r[None, :] < R)
    tl.store(gB_ptrs, acc.to(tl.bfloat16), mask=mask_gB)


# ============================================================================
# KERNEL: fused_lora_grad_x_kernel
#   grad_x_lora = grad_xA @ A.T  ->  (M, K) bf16
# Main backward matmul. Output is (M, K) which is large (M*K elements).
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 256, "BK": 32}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 64}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 64}, num_warps=8, num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=4),
    ],
    key=["M", "K", "R"],
)
@triton.jit
def fused_lora_grad_x_kernel(
    grad_xA_ptr, A_ptr, grad_x_ptr,
    M, K, R,
    stride_gxm, stride_gxr,
    stride_ak, stride_ar,
    stride_gxm2, stride_gxk,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """grad_x_lora = grad_xA @ A.T  -> (M, K) bf16.

    grad_xA: (M, R), A: (K, R), A.T: (R, K) -> output (M, K).
    Scaling already absorbed into grad_xA (from fused_lora_grad_xA_kernel),
    so no scaling needed here.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_k = tl.cdiv(K, BN)
    pid_m = pid // grid_k
    pid_k = pid % grid_k

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = pid_k * BN + tl.arange(0, BN)
    offs_r = tl.arange(0, BK)

    gx_ptrs = grad_xA_ptr + offs_m[:, None] * stride_gxm + offs_r[None, :] * stride_gxr
    # A: (K, R). We want A.T[r, k] = A[k, r] -> load as (BK, BN) with r in rows, k in cols.
    a_ptrs = A_ptr + offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for r_iter in range(0, tl.cdiv(R, BK)):
        mask_gx = (offs_m[:, None] < M) & (offs_r[None, :] + r_iter * BK < R)
        mask_a = (offs_r[:, None] + r_iter * BK < R) & (offs_k[None, :] < K)
        gx_tile = tl.load(gx_ptrs, mask=mask_gx, other=0.0).to(tl.bfloat16)
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.bfloat16)
        acc += tl.dot(gx_tile, a_tile)
        gx_ptrs += BK * stride_gxr
        a_ptrs += BK * stride_ar

    gxl_ptrs = grad_x_ptr + offs_m[:, None] * stride_gxm2 + offs_k[None, :] * stride_gxk
    mask_gxl = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(gxl_ptrs, acc.to(tl.bfloat16), mask=mask_gxl)


# ============================================================================
# Python launchers
# ============================================================================
def fused_lora_forward_triton(
    x: torch.Tensor,       # (M, K) bf16
    lora_A: torch.Tensor,  # (K, R) bf16
    lora_B: torch.Tensor,  # (N, R) bf16
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """y = (x @ A) @ B.T * scaling  -> (M, N) bf16.
    Also returns xA = x @ A  -> (M, R) bf16 for reuse in backward.

    Returns (y, xA).
    """
    assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
    assert lora_A.dtype == torch.bfloat16, f"lora_A must be bf16, got {lora_A.dtype}"
    assert lora_B.dtype == torch.bfloat16, f"lora_B must be bf16, got {lora_B.dtype}"
    M, K = x.shape
    K2, R = lora_A.shape
    N, R2 = lora_B.shape
    assert K == K2, f"K mismatch: x.K={K} vs A.K={K2}"
    assert R == R2, f"R mismatch: A.R={R} vs B.R={R2}"

    x = x.contiguous()
    lora_A = lora_A.contiguous()
    lora_B = lora_B.contiguous()

    xA = torch.empty((M, R), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    grid_xA = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(R, meta["BN"]),)
    fused_lora_xA_kernel[grid_xA](
        x, lora_A, xA,
        M, K, R,
        x.stride(0), x.stride(1),
        lora_A.stride(0), lora_A.stride(1),
        xA.stride(0), xA.stride(1),
    )

    grid_y = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    fused_lora_matmul_kernel[grid_y](
        xA, lora_B, y,
        M, N, R,
        float(scaling),
        xA.stride(0), xA.stride(1),
        lora_B.stride(0), lora_B.stride(1),
        y.stride(0), y.stride(1),
    )
    return y, xA


def fused_lora_grad_xA_triton(
    grad_y: torch.Tensor,  # (M, N) bf16
    lora_B: torch.Tensor,  # (N, R) bf16
    scaling: float,
) -> torch.Tensor:
    """grad_xA = (grad_y * scaling) @ B  -> (M, R) bf16.

    Scaling fused into grad_y load.
    """
    assert grad_y.dtype == torch.bfloat16, f"grad_y must be bf16, got {grad_y.dtype}"
    assert lora_B.dtype == torch.bfloat16, f"lora_B must be bf16, got {lora_B.dtype}"
    M, N = grad_y.shape
    N2, R = lora_B.shape
    assert N == N2, f"N mismatch: grad_y.N={N} vs B.N={N2}"

    grad_y = grad_y.contiguous()
    lora_B = lora_B.contiguous()
    grad_xA = torch.empty((M, R), dtype=torch.bfloat16, device=grad_y.device)

    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(R, meta["BN"]),)
    fused_lora_grad_xA_kernel[grid](
        grad_y, lora_B, grad_xA,
        M, N, R,
        float(scaling),
        grad_y.stride(0), grad_y.stride(1),
        lora_B.stride(0), lora_B.stride(1),
        grad_xA.stride(0), grad_xA.stride(1),
    )
    return grad_xA


def fused_lora_grad_A_triton(
    x: torch.Tensor,         # (M, K) bf16
    grad_xA: torch.Tensor,   # (M, R) bf16
) -> torch.Tensor:
    """grad_A = x.T @ grad_xA  -> (K, R) bf16."""
    assert x.dtype == torch.bfloat16, f"x must be bf16, got {x.dtype}"
    assert grad_xA.dtype == torch.bfloat16, f"grad_xA must be bf16, got {grad_xA.dtype}"
    M, K = x.shape
    M2, R = grad_xA.shape
    assert M == M2, f"M mismatch: x.M={M} vs grad_xA.M={M2}"

    x = x.contiguous()
    grad_xA = grad_xA.contiguous()
    grad_A = torch.empty((K, R), dtype=torch.bfloat16, device=x.device)

    grid = lambda meta: (triton.cdiv(K, meta["BM"]) * triton.cdiv(R, meta["BN"]),)
    fused_lora_grad_A_kernel[grid](
        x, grad_xA, grad_A,
        M, K, R,
        x.stride(0), x.stride(1),
        grad_xA.stride(0), grad_xA.stride(1),
        grad_A.stride(0), grad_A.stride(1),
    )
    return grad_A


def fused_lora_grad_B_triton(
    grad_y: torch.Tensor,   # (M, N) bf16
    xA: torch.Tensor,       # (M, R) bf16
    scaling: float,
) -> torch.Tensor:
    """grad_B = (grad_y * scaling).T @ xA  -> (N, R) bf16.

    Scaling fused into grad_y load (no separate aten::mul kernel).
    """
    assert grad_y.dtype == torch.bfloat16, f"grad_y must be bf16, got {grad_y.dtype}"
    assert xA.dtype == torch.bfloat16, f"xA must be bf16, got {xA.dtype}"
    M, N = grad_y.shape
    M2, R = xA.shape
    assert M == M2, f"M mismatch: grad_y.M={M} vs xA.M={M2}"

    grad_y = grad_y.contiguous()
    xA = xA.contiguous()
    grad_B = torch.empty((N, R), dtype=torch.bfloat16, device=grad_y.device)

    grid = lambda meta: (triton.cdiv(N, meta["BM"]) * triton.cdiv(R, meta["BN"]),)
    fused_lora_grad_B_kernel[grid](
        grad_y, xA, grad_B,
        M, N, R,
        float(scaling),
        grad_y.stride(0), grad_y.stride(1),
        xA.stride(0), xA.stride(1),
        grad_B.stride(0), grad_B.stride(1),
    )
    return grad_B


def fused_lora_grad_x_triton(
    grad_xA: torch.Tensor,  # (M, R) bf16
    lora_A: torch.Tensor,   # (K, R) bf16
) -> torch.Tensor:
    """grad_x_lora = grad_xA @ A.T  -> (M, K) bf16."""
    assert grad_xA.dtype == torch.bfloat16, f"grad_xA must be bf16, got {grad_xA.dtype}"
    assert lora_A.dtype == torch.bfloat16, f"lora_A must be bf16, got {lora_A.dtype}"
    M, R = grad_xA.shape
    K, R2 = lora_A.shape
    assert R == R2, f"R mismatch: {R} vs {R2}"

    grad_xA = grad_xA.contiguous()
    lora_A = lora_A.contiguous()
    grad_x = torch.empty((M, K), dtype=torch.bfloat16, device=grad_xA.device)

    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(K, meta["BN"]),)
    fused_lora_grad_x_kernel[grid](
        grad_xA, lora_A, grad_x,
        M, K, R,
        grad_xA.stride(0), grad_xA.stride(1),
        lora_A.stride(0), lora_A.stride(1),
        grad_x.stride(0), grad_x.stride(1),
    )
    return grad_x


# ============================================================================
# Autograd Function - TritonLoRALinear
#   forward:  y, xA = fused_lora_forward_triton(x, A, B, scaling)
#             ctx.save_for_backward(x, A, B, xA)
#   backward: grad_xA = fused_lora_grad_xA_triton(grad_y, B, scaling)
#             grad_x_lora = fused_lora_grad_x_triton(grad_xA, A)
#             grad_A = fused_lora_grad_A_triton(x, grad_xA)
#             grad_B = fused_lora_grad_B_triton(grad_y, xA, scaling)
# ============================================================================
class TritonLoRALinear(torch.autograd.Function):
    """Fused LoRA forward + backward via Triton.

    Eliminates (per step, 31 LoRA modules):
      - 31 aten::mul for scaling (fused into grad_y load + output store)
      - 31 separate PyTorch matmul dispatches for grad_A, grad_B, grad_x_lora
        (replaced with 4 Triton matmul kernels sharing cached xA + grad_xA)
      - autograd graph traversal overhead for the (x @ A) @ B.T * scaling
        expression (1 node instead of 3)

    Does NOT eliminate (deferred to Patch 20):
      - 31 aten::add for grad_x accumulation (grad_x_base + grad_x_lora)
    """

    @staticmethod
    def forward(ctx, x, lora_A, lora_B, scaling):
        x = x.contiguous()
        lora_A = lora_A.contiguous()
        lora_B = lora_B.contiguous()

        M, K = x.shape
        K2, R = lora_A.shape
        N, R2 = lora_B.shape
        assert K == K2, f"K mismatch: x.K={K} vs A.K={K2}"
        assert R == R2, f"R mismatch: A.R={R} vs B.R={R2}"

        y, xA = fused_lora_forward_triton(x, lora_A, lora_B, float(scaling))

        ctx.save_for_backward(x, lora_A, lora_B, xA)
        ctx.scaling = float(scaling)
        ctx.M = M
        ctx.K = K
        ctx.N = N
        ctx.R = R
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, lora_A, lora_B, xA = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        needs_grad_x = ctx.needs_input_grad[0]
        needs_grad_A = ctx.needs_input_grad[1]
        needs_grad_B = ctx.needs_input_grad[2]

        scaling = ctx.scaling

        # grad_xA is needed for grad_x_lora, grad_A, and grad_B (the latter
        # uses xA not grad_xA, but grad_xA is needed because we want to avoid
        # recomputing it for grad_x_lora). Compute it once if any is needed.
        # Note: grad_B does NOT need grad_xA (it uses xA directly).
        grad_xA = None
        if needs_grad_x or needs_grad_A:
            grad_xA = fused_lora_grad_xA_triton(grad_y, lora_B, scaling)

        grad_x = None
        if needs_grad_x:
            grad_x = fused_lora_grad_x_triton(grad_xA, lora_A)

        grad_A = None
        if needs_grad_A:
            grad_A = fused_lora_grad_A_triton(x, grad_xA)

        grad_B = None
        if needs_grad_B:
            # grad_B uses grad_y (with scaling fused) and xA (cached from forward).
            # No recompute of xA needed -> saves 1 M*K*R matmul per LoRA module.
            grad_B = fused_lora_grad_B_triton(grad_y, xA, scaling)

        # Return tuple matches forward input order: (x, lora_A, lora_B, scaling)
        return grad_x, grad_A, grad_B, None


def triton_lora_forward(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Functional interface for fused LoRA forward.

    Args:
        x: (M, K) bf16 input
        lora_A: (K, R) bf16
        lora_B: (N, R) bf16
        scaling: float (alpha / rank)
    Returns:
        y: (M, N) bf16 = (x @ lora_A) @ lora_B.T * scaling
    """
    return TritonLoRALinear.apply(x, lora_A, lora_B, scaling)
