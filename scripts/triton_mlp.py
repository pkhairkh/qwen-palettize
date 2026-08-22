"""Triton fused kernels for the Qwen3.5 MLP block (Patch 12).

WAVE 2 — Patch 12 (layer-fusion):
  Fuses `gate_proj(x) * SiLU(up_proj(x)) → down_proj(...)` into a single
  autograd Function.  The three PalettizedLinears stay separate (they have
  different palettes, indices, and biases), but the SiLU + elementwise multiply
  + the down_proj input materialization are fused.

  Math (SwiGLU / Shazeer 2020):
      gate_out = gate_proj(x)         # (M, N_inner)  PalettizedLinear
      up_out   = up_proj(x)           # (M, N_inner)  PalettizedLinear
      act      = gate_out * SiLU(up_out)   = gate_out * (up_out * sigmoid(up_out))
      y        = down_proj(act)       # (M, K)        PalettizedLinear

  In the FUSED forward, we:
    1. Run triton_soft_forward.triton_soft_linear for gate_proj  →  gate_out
    2. Run triton_soft_forward.triton_soft_linear for up_proj    →  up_out
    3. Run fused_silu_mul_triton(gate_out, up_out)               →  act
       This Triton kernel fuses SiLU + multiply into ONE elementwise kernel
       (was 2 PyTorch kernels: silu + mul).
    4. Run triton_soft_forward.triton_soft_linear for down_proj(act) →  y

  In the FUSED backward, we fuse:
    grad_act = grad_y @ W_ste_down.T  (Triton TC matmul)
    grad_gate, grad_up = fused_silu_mul_backward_triton(grad_act, gate_out, up_out)
      — one Triton kernel that computes:
          grad_gate = grad_act * SiLU(up_out)
          grad_up   = grad_act * gate_out * SiLU'(up_out)
                   = grad_act * gate_out * (SiLU(up_out) * (1 - sigmoid(up_out)))
      — was 4 PyTorch kernels (sigmoid, mul, sigmoid_grad, mul, mul).
    grad_x = grad_gate @ W_ste_gate.T + grad_up @ W_ste_up.T  (two Triton matmuls + aten::add_)

References:
  - SwiGLU (Shazeer 2020, arXiv:2002.05202) — fused gate+up+activation.
  - research-kernel-efficiency/00_overview.md §3 — fused MLP pattern.

File ownership:
  - This file is NEW and owned by layer-fusion (Patch 12).
  - We IMPORT from triton_soft_forward (owned by triton-kernels).
  - We IMPORT from triton_soft_backward (owned by triton-kernels).
  - We do NOT touch qwen_model.py.

Layouts:
  x:            (M, K)        bf16 — input to gate_proj / up_proj
  gate_out:     (M, N_inner)  bf16 — gate_proj output
  up_out:       (M, N_inner)  bf16 — up_proj output
  act:          (M, N_inner)  bf16 — gate_out * SiLU(up_out)
  y:            (M, K)        bf16 — down_proj output (back to hidden dim)
  W_ste_gate:   (K, N_inner)  bf16  — saved for backward
  W_ste_up:     (K, N_inner)  bf16
  W_ste_down:   (N_inner, K)  bf16
  P_aos_*:      (K, N_inner, 4) / (N_inner, K, 4) fp16  — saved for backward
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL — fused SiLU + multiply:  act = gate * SiLU(up)
# ═════════════════════════════════════════════════════════════════════════════
# SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
#
# Numerically stable sigmoid:  sigmoid(x) = 1 / (1 + exp(-x))
#   For x >= 0:  sigmoid(x) = 1 / (1 + exp(-x))
#   For x <  0:  sigmoid(x) = exp(x) / (1 + exp(x))
#   We use the Triton-friendly form:  s = 1 / (1 + exp(-x)), branching on sign
#   via tl.where to avoid overflow.
@triton.jit
def fused_silu_mul_kernel(
    gate_ptr,   # (M, N) bf16
    up_ptr,     # (M, N) bf16
    out_ptr,    # (M, N) bf16 — OUTPUT: gate * SiLU(up)
    M, N,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr,
):
    """act = gate * SiLU(up) = gate * (up * sigmoid(up))."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    g = tl.load(
        gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
        mask=mask, other=0.0,
    ).to(tl.float32)  # (BM, BN)
    u = tl.load(
        up_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un,
        mask=mask, other=0.0,
    ).to(tl.float32)  # (BM, BN)

    # Numerically stable sigmoid
    # For u >= 0: s = 1 / (1 + exp(-u))
    # For u <  0: s = exp(u) / (1 + exp(u))    (avoids overflow of exp(-u))
    # Compute both and select via tl.where
    pos_u = tl.maximum(u, 0.0)               # for the negative branch
    neg_u = tl.minimum(u, 0.0)               # for the negative branch (negated)
    # exp(-u) for u >= 0
    exp_neg_u = tl.exp(-pos_u)
    s_pos = 1.0 / (1.0 + exp_neg_u)
    # exp(u) for u < 0
    exp_pos_u = tl.exp(neg_u)
    s_neg = exp_pos_u / (1.0 + exp_pos_u)
    s = tl.where(u >= 0.0, s_pos, s_neg)

    silu_u = u * s
    act = g * silu_u

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        act.to(tl.bfloat16),
        mask=mask,
    )


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL — fused SiLU + multiply backward:  grad_gate, grad_up
# ═════════════════════════════════════════════════════════════════════════════
# Math (derivation):
#   act = gate * SiLU(up) = gate * silu_u  where silu_u = up * sigmoid(up)
#   grad_gate = grad_act * SiLU(up)                    = grad_act * silu_u
#   grad_up   = grad_act * gate * SiLU'(up)
#   SiLU'(up) = sigmoid(up) + up * sigmoid'(up)
#             = sigmoid(up) * (1 + up * (1 - sigmoid(up)))
#             = sigmoid(up) * (1 - up * sigmoid(up) + up)
#   Simplest form:
#     SiLU'(up) = sigmoid(up) * (1 + up * (1 - sigmoid(up)))
#   Or (more standard):
#     SiLU'(up) = sigmoid(up) + up * sigmoid(up) * (1 - sigmoid(up))
#   Both forms are algebraically equivalent.  We use the first.
@triton.jit
def fused_silu_mul_backward_kernel(
    grad_act_ptr,  # (M, N) bf16
    gate_ptr,      # (M, N) bf16 — saved from forward
    up_ptr,        # (M, N) bf16 — saved from forward
    grad_gate_ptr, # (M, N) bf16 — OUTPUT
    grad_up_ptr,   # (M, N) bf16 — OUTPUT
    M, N,
    stride_gam, stride_gan,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_ggm, stride_ggn,
    stride_gum, stride_gun,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    ga = tl.load(
        grad_act_ptr + offs_m[:, None] * stride_gam + offs_n[None, :] * stride_gan,
        mask=mask, other=0.0,
    ).to(tl.float32)
    g = tl.load(
        gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
        mask=mask, other=0.0,
    ).to(tl.float32)
    u = tl.load(
        up_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un,
        mask=mask, other=0.0,
    ).to(tl.float32)

    # Stable sigmoid (same form as forward)
    pos_u = tl.maximum(u, 0.0)
    neg_u = tl.minimum(u, 0.0)
    exp_neg_u = tl.exp(-pos_u)
    s_pos = 1.0 / (1.0 + exp_neg_u)
    exp_pos_u = tl.exp(neg_u)
    s_neg = exp_pos_u / (1.0 + exp_pos_u)
    s = tl.where(u >= 0.0, s_pos, s_neg)

    silu_u = u * s
    # SiLU'(up) = s * (1 + u * (1 - s))
    one_minus_s = 1.0 - s
    silu_grad = s * (1.0 + u * one_minus_s)

    grad_gate = ga * silu_u
    grad_up = ga * g * silu_grad

    tl.store(
        grad_gate_ptr + offs_m[:, None] * stride_ggm + offs_n[None, :] * stride_ggn,
        grad_gate.to(tl.bfloat16),
        mask=mask,
    )
    tl.store(
        grad_up_ptr + offs_m[:, None] * stride_gum + offs_n[None, :] * stride_gun,
        grad_up.to(tl.bfloat16),
        mask=mask,
    )


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL — fused grad_x accumulation:  grad_x += grad_a @ W_ste_a.T (gate)
#                                     + grad_b @ W_ste_b.T (up)
# ═════════════════════════════════════════════════════════════════════════════
# This kernel accumulates the gradient contributions from BOTH gate_proj and
# up_proj into a single grad_x tensor, eliminating the aten::add_ that
# currently accumulates them in the PyTorch autograd path.
#
# Math:
#   grad_x_gate = grad_gate @ W_ste_gate.T    (M, K) from (M, N) @ (K, N).T
#   grad_x_up   = grad_up   @ W_ste_up.T      (M, K)
#   grad_x      = grad_x_gate + grad_x_up      ← THIS IS THE FUSION POINT
#
# We implement this as a single Triton kernel that loads a (BM, BK) tile of
# grad_x, computes BOTH matmuls, and writes the sum.  Both W_ste_gate and
# W_ste_up share the same K dimension (hidden dim), so the K blocking is
# shared and the L2 cache reuse is good.
@triton.autotune(
    configs=[
        triton.Config({"BM": 64,  "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 256, "BK": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 4}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_dual_grad_x_kernel(
    grad_a_ptr,    # (M, N) bf16 — grad_gate (for gate_proj backward)
    grad_b_ptr,    # (M, N) bf16 — grad_up   (for up_proj backward)
    W_a_ptr,       # (K, N) bf16 — W_ste_gate (for gate_proj backward)
    W_b_ptr,       # (K, N) bf16 — W_ste_up   (for up_proj backward)
    grad_x_ptr,    # (M, K) bf16 — OUTPUT (sum of two matmuls)
    M, N, K,
    stride_gam, stride_gan,
    stride_gbm, stride_gbn,
    stride_wak, stride_wan,
    stride_wbk, stride_wbn,
    stride_gxm, stride_gxk,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """grad_x[m, k] = Σ_n grad_a[m, n] * W_a[k, n] + grad_b[m, n] * W_b[k, n].

    Fuses two (M,N)@(K,N).T matmuls into one kernel.  Eliminates the
    `aten::add_` for residual grad accumulation that appears in the autograd
    profile (167ms / step).
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_k = tl.cdiv(K, BK)
    # L2 cache swizzle (from Triton matmul tutorial)
    num_pid_in_group = GROUP_M * grid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_n = tl.arange(0, BN)

    # grad_a, grad_b pointers (M, N) — we walk N in chunks of BN
    ga_ptrs = grad_a_ptr + offs_m[:, None] * stride_gam + offs_n[None, :] * stride_gan
    gb_ptrs = grad_b_ptr + offs_m[:, None] * stride_gbm + offs_n[None, :] * stride_gbn
    # W_a, W_b pointers (K, N) — load TRANSPOSED by swapping strides so we
    # get (BN, BK) tiles directly (avoids tl.trans shared-mem transpose).
    wa_ptrs = W_a_ptr + offs_n[:, None] * stride_wan + offs_k[None, :] * stride_wak
    wb_ptrs = W_b_ptr + offs_n[:, None] * stride_wbn + offs_k[None, :] * stride_wbk

    acc = tl.zeros((BM, BK), dtype=tl.float32)
    for n_iter in range(0, tl.cdiv(N, BN)):
        n_off = n_iter * BN
        mask_ga = (offs_m[:, None] < M) & ((offs_n[None, :] + n_off) < N)
        mask_gb = (offs_m[:, None] < M) & ((offs_n[None, :] + n_off) < N)
        mask_wa = ((offs_n[:, None] + n_off) < N) & (offs_k[None, :] < K)
        mask_wb = ((offs_n[:, None] + n_off) < N) & (offs_k[None, :] < K)

        ga_tile = tl.load(ga_ptrs, mask=mask_ga, other=0.0)  # (BM, BN) bf16
        gb_tile = tl.load(gb_ptrs, mask=mask_gb, other=0.0)  # (BM, BN) bf16
        wa_tile = tl.load(wa_ptrs, mask=mask_wa, other=0.0)  # (BN, BK) bf16
        wb_tile = tl.load(wb_ptrs, mask=mask_wb, other=0.0)  # (BN, BK) bf16

        # Fused dual matmul:  acc += ga @ wa + gb @ wb
        acc += tl.dot(ga_tile, wa_tile)
        acc += tl.dot(gb_tile, wb_tile)

        ga_ptrs += BN * stride_gan
        gb_ptrs += BN * stride_gbn
        wa_ptrs += BN * stride_wan
        wb_ptrs += BN * stride_wbn

    gx_ptrs = grad_x_ptr + offs_m[:, None] * stride_gxm + offs_k[None, :] * stride_gxk
    mask_gx = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(gx_ptrs, acc.to(tl.bfloat16), mask=mask_gx)


# ═════════════════════════════════════════════════════════════════════════════
# Python launchers
# ═════════════════════════════════════════════════════════════════════════════
def fused_silu_mul_triton(
    gate: torch.Tensor,  # (M, N) bf16
    up: torch.Tensor,    # (M, N) bf16
) -> torch.Tensor:
    """act = gate * SiLU(up).  Returns (M, N) bf16.

    One Triton kernel (was 2 PyTorch kernels: silu + mul).
    """
    assert gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16
    assert gate.shape == up.shape
    M, N = gate.shape
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    # BM/BN tuned for typical N_inner = 2*K = 5120..8192
    BM, BN = 128, 128
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    fused_silu_mul_kernel[grid](
        gate, up, out,
        M, N,
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        out.stride(0), out.stride(1),
        BM=BM, BN=BN,
        num_warps=8,
        num_stages=1,
    )
    return out


def fused_silu_mul_backward_triton(
    grad_act: torch.Tensor,  # (M, N) bf16
    gate: torch.Tensor,      # (M, N) bf16 — saved from forward
    up: torch.Tensor,        # (M, N) bf16 — saved from forward
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (grad_gate, grad_up).  Both (M, N) bf16.

    One Triton kernel (was 4+ PyTorch kernels: sigmoid, mul, sigmoid_grad, mul, mul).
    """
    assert grad_act.dtype == torch.bfloat16
    assert gate.dtype == torch.bfloat16
    assert up.dtype == torch.bfloat16
    assert gate.shape == up.shape == grad_act.shape
    M, N = gate.shape
    grad_act = grad_act.contiguous()
    gate = gate.contiguous()
    up = up.contiguous()
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    BM, BN = 128, 128
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    fused_silu_mul_backward_kernel[grid](
        grad_act, gate, up, grad_gate, grad_up,
        M, N,
        grad_act.stride(0), grad_act.stride(1),
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        grad_gate.stride(0), grad_gate.stride(1),
        grad_up.stride(0), grad_up.stride(1),
        BM=BM, BN=BN,
        num_warps=8,
        num_stages=1,
    )
    return grad_gate, grad_up


def fused_dual_grad_x_triton(
    grad_a: torch.Tensor,   # (M, N) bf16 — grad w.r.t. gate_proj output
    grad_b: torch.Tensor,   # (M, N) bf16 — grad w.r.t. up_proj output
    W_a: torch.Tensor,      # (K, N) bf16 — W_ste for gate_proj
    W_b: torch.Tensor,      # (K, N) bf16 — W_ste for up_proj
) -> torch.Tensor:
    """grad_x = grad_a @ W_a.T + grad_b @ W_b.T  → (M, K) bf16.

    Fuses two matmuls + their sum.  Eliminates the aten::add_ in the autograd
    path (the 167ms of add_ calls documented in the profiler).
    """
    assert grad_a.dtype == grad_b.dtype == W_a.dtype == W_b.dtype == torch.bfloat16
    assert grad_a.shape == grad_b.shape
    M, N = grad_a.shape
    K, N2 = W_a.shape
    assert N == N2
    assert W_b.shape == (K, N)
    grad_a = grad_a.contiguous()
    grad_b = grad_b.contiguous()
    W_a = W_a.contiguous()
    W_b = W_b.contiguous()
    grad_x = torch.empty((M, K), dtype=torch.bfloat16, device=grad_a.device)
    grid = lambda meta: (
        triton.cdiv(M, meta["BM"]) * triton.cdiv(K, meta["BK"]),
    )
    fused_dual_grad_x_kernel[grid](
        grad_a, grad_b, W_a, W_b, grad_x,
        M, N, K,
        grad_a.stride(0), grad_a.stride(1),
        grad_b.stride(0), grad_b.stride(1),
        W_a.stride(0), W_a.stride(1),
        W_b.stride(0), W_b.stride(1),
        grad_x.stride(0), grad_x.stride(1),
    )
    return grad_x


# ═════════════════════════════════════════════════════════════════════════════
# Autograd Function — fused SwiGLU MLP
# ═════════════════════════════════════════════════════════════════════════════
class FusedMLP(torch.autograd.Function):
    """Fused SwiGLU MLP: gate * SiLU(up) → down.

    Forward graph:
        x ──┬─► gate_proj ──┐
            │                ├─► fused_silu_mul ──► act ──► down_proj ──► y
            └─► up_proj ─────┘

    The three PalettizedLinears stay separate (different palettes, indices,
    biases).  The fusion is at the SiLU+mul elementwise + grad_x accumulation.

    Backward graph:
        grad_y ──► (down_proj backward) ──► grad_act
                                              │
                                              ├──► fused_silu_mul_backward ──► grad_gate, grad_up
                                              │
                                              ├──► (gate_proj backward) ──► grad_palette_g, grad_logits_g
                                              └──► (up_proj backward)   ──► grad_palette_u, grad_logits_u
        grad_x = fused_dual_grad_x(grad_gate, grad_up, W_ste_gate, W_ste_up)
    """

    @staticmethod
    def forward(ctx, x,
                gate_palette, gate_logits, gate_bias,
                up_palette, up_logits, up_bias,
                down_palette, down_logits, down_bias,
                group_size, tau):
        from triton_soft_forward import (
            compute_P_W_ste_triton,
            fused_soft_matmul_triton,
            _next_soft_step_seed,
        )

        x = x.contiguous()
        for t in (gate_palette, gate_logits, up_palette, up_logits,
                  down_palette, down_logits):
            t_c = t.contiguous()
            if t is gate_palette: gate_palette = t_c
            elif t is gate_logits: gate_logits = t_c
            elif t is up_palette: up_palette = t_c
            elif t is up_logits: up_logits = t_c
            elif t is down_palette: down_palette = t_c
            elif t is down_logits: down_logits = t_c
        if gate_bias is not None: gate_bias = gate_bias.contiguous()
        if up_bias is not None: up_bias = up_bias.contiguous()
        if down_bias is not None: down_bias = down_bias.contiguous()

        M, K = x.shape
        n_planes, K_g, N_inner = gate_logits.shape
        assert n_planes == 4 and K_g == K
        assert up_logits.shape == (4, K, N_inner)
        assert down_logits.shape == (4, N_inner, K), \
            f"down_logits must be (4, N_inner, K), got {down_logits.shape}"

        # ── 1. gate_proj  →  gate_out ────────────────────────────────────
        step_seed_g = _next_soft_step_seed()
        P_aos_g, _, W_ste_g = compute_P_W_ste_triton(
            gate_logits, gate_palette, group_size, float(tau), step_seed_g,
        )
        gate_out = fused_soft_matmul_triton(x, W_ste_g, gate_bias)

        # ── 2. up_proj    →  up_out ──────────────────────────────────────
        step_seed_u = _next_soft_step_seed()
        P_aos_u, _, W_ste_u = compute_P_W_ste_triton(
            up_logits, up_palette, group_size, float(tau), step_seed_u,
        )
        up_out = fused_soft_matmul_triton(x, W_ste_u, up_bias)

        # ── 3. fused SiLU + mul  →  act ──────────────────────────────────
        act = fused_silu_mul_triton(gate_out, up_out)

        # ── 4. down_proj  →  y ───────────────────────────────────────────
        step_seed_d = _next_soft_step_seed()
        P_aos_d, _, W_ste_d = compute_P_W_ste_triton(
            down_logits, down_palette, group_size, float(tau), step_seed_d,
        )
        y = fused_soft_matmul_triton(act, W_ste_d, down_bias)

        # ── save for backward ────────────────────────────────────────────
        # We save x, gate_out, up_out, act, P_aos_*, W_ste_*
        # The PalettizedLinear parameters (palette, logits) are saved by
        # autograd via ctx.save_for_backward (we save the actual tensors).
        ctx.save_for_backward(
            x, gate_out, up_out, act,
            gate_palette, gate_logits, P_aos_g, W_ste_g,
            up_palette, up_logits, P_aos_u, W_ste_u,
            down_palette, down_logits, P_aos_d, W_ste_d,
        )
        ctx.group_size = group_size
        ctx.tau = tau
        ctx.has_gate_bias = gate_bias is not None
        ctx.has_up_bias = up_bias is not None
        ctx.has_down_bias = down_bias is not None
        ctx.K = K
        ctx.N_inner = N_inner
        return y

    @staticmethod
    def backward(ctx, grad_y):
        from triton_soft_backward import (
            fused_soft_bwd_grad_x_triton,
            fused_soft_bwd_grad_W_triton,
            fused_soft_bwd_elementwise_triton,
        )

        (x, gate_out, up_out, act,
         gate_palette, gate_logits, P_aos_g, W_ste_g,
         up_palette, up_logits, P_aos_u, W_ste_u,
         down_palette, down_logits, P_aos_d, W_ste_d) = ctx.saved_tensors

        grad_y = grad_y.contiguous()
        M, K = x.shape
        N_inner = ctx.N_inner
        GS = ctx.group_size

        needs_grad_x = ctx.needs_input_grad[0]
        needs_grad_gate_palette = ctx.needs_input_grad[1]
        needs_grad_gate_logits = ctx.needs_input_grad[2]
        needs_grad_gate_bias = ctx.has_gate_bias and ctx.needs_input_grad[3]
        needs_grad_up_palette = ctx.needs_input_grad[4]
        needs_grad_up_logits = ctx.needs_input_grad[5]
        needs_grad_up_bias = ctx.has_up_bias and ctx.needs_input_grad[6]
        needs_grad_down_palette = ctx.needs_input_grad[7]
        needs_grad_down_logits = ctx.needs_input_grad[8]
        needs_grad_down_bias = ctx.has_down_bias and ctx.needs_input_grad[9]

        # ── 1. down_proj backward: grad_act = grad_y @ W_ste_down.T ──────
        grad_act = fused_soft_bwd_grad_x_triton(grad_y, W_ste_d)

        # ── 2. fused SiLU+mul backward: grad_gate, grad_up ────────────────
        grad_gate, grad_up = fused_silu_mul_backward_triton(grad_act, gate_out, up_out)

        # ── 3. grad_x = grad_gate @ W_ste_gate.T + grad_up @ W_ste_up.T ──
        # FUSED into a single kernel (eliminates aten::add_).
        grad_x = None
        if needs_grad_x:
            grad_x = fused_dual_grad_x_triton(grad_gate, grad_up, W_ste_g, W_ste_u)

        # ── 4. down_proj grad_palette / grad_logits ─────────────────────
        grad_down_palette = None
        grad_down_logits = None
        if needs_grad_down_palette or needs_grad_down_logits:
            grad_W_d = fused_soft_bwd_grad_W_triton(act, grad_y)
            grad_down_logits, grad_down_palette = fused_soft_bwd_elementwise_triton(
                grad_W_d, P_aos_d, down_palette, GS,
            )
            if not needs_grad_down_logits:
                grad_down_logits = None
            if not needs_grad_down_palette:
                grad_down_palette = None

        # ── 5. gate_proj grad_palette / grad_logits ─────────────────────
        grad_gate_palette = None
        grad_gate_logits = None
        if needs_grad_gate_palette or needs_grad_gate_logits:
            grad_W_g = fused_soft_bwd_grad_W_triton(x, grad_gate)
            grad_gate_logits, grad_gate_palette = fused_soft_bwd_elementwise_triton(
                grad_W_g, P_aos_g, gate_palette, GS,
            )
            if not needs_grad_gate_logits:
                grad_gate_logits = None
            if not needs_grad_gate_palette:
                grad_gate_palette = None

        # ── 6. up_proj grad_palette / grad_logits ────────────────────────
        grad_up_palette = None
        grad_up_logits = None
        if needs_grad_up_palette or needs_grad_up_logits:
            grad_W_u = fused_soft_bwd_grad_W_triton(x, grad_up)
            grad_up_logits, grad_up_palette = fused_soft_bwd_elementwise_triton(
                grad_W_u, P_aos_u, up_palette, GS,
            )
            if not needs_grad_up_logits:
                grad_up_logits = None
            if not needs_grad_up_palette:
                grad_up_palette = None

        # ── 7. biases ────────────────────────────────────────────────────
        grad_gate_bias = grad_y.new_zeros(K) if False else None  # never compute
        # bias grads are simple reductions — kept in PyTorch (cheap, one
        # reduction per layer per step).  Could be fused into a future kernel.
        grad_gate_bias = None
        grad_up_bias = None
        grad_down_bias = None
        if needs_grad_gate_bias:
            grad_gate_bias = grad_gate.sum(dim=0)
        if needs_grad_up_bias:
            grad_up_bias = grad_up.sum(dim=0)
        if needs_grad_down_bias:
            grad_down_bias = grad_y.sum(dim=0)

        # Return tuple matches forward input order:
        # (x, gate_palette, gate_logits, gate_bias,
        #  up_palette, up_logits, up_bias,
        #  down_palette, down_logits, down_bias, group_size, tau)
        return (
            grad_x,
            grad_gate_palette, grad_gate_logits, grad_gate_bias,
            grad_up_palette, grad_up_logits, grad_up_bias,
            grad_down_palette, grad_down_logits, grad_down_bias,
            None, None,
        )


def fused_mlp(
    x: torch.Tensor,
    gate_palette: torch.Tensor, gate_logits: torch.Tensor, gate_bias: torch.Tensor | None,
    up_palette: torch.Tensor, up_logits: torch.Tensor, up_bias: torch.Tensor | None,
    down_palette: torch.Tensor, down_logits: torch.Tensor, down_bias: torch.Tensor | None,
    group_size: int = 256,
    tau: float = 1.0,
) -> torch.Tensor:
    """Functional interface for the fused SwiGLU MLP soft path.

    Args:
        x: (M, K) bf16
        gate_palette: (G, 4) bf16  — gate_proj palette
        gate_logits:  (4, K, N_inner) fp16  — gate_proj logits
        gate_bias:    (N_inner,) bf16 or None
        up_palette:   (G, 4) bf16  — up_proj palette
        up_logits:    (4, K, N_inner) fp16
        up_bias:      (N_inner,) bf16 or None
        down_palette: (G', 4) bf16 — down_proj palette (N_inner groups)
        down_logits:  (4, N_inner, K) fp16  — down_proj logits
        down_bias:    (K,) bf16 or None
        group_size:   palette group size
        tau:          Gumbel-Softmax temperature

    Returns: y (M, K) bf16 — same shape as x (MLP returns to hidden dim).
    """
    return FusedMLP.apply(
        x,
        gate_palette, gate_logits, gate_bias,
        up_palette, up_logits, up_bias,
        down_palette, down_logits, down_bias,
        group_size, tau,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Self-test
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("triton_mlp.py — Patch 12: fused SwiGLU MLP (gate + up + SiLU + down)")
    print("=" * 70)
    print("Kernels:")
    print("  - fused_silu_mul_kernel              (1 elementwise kernel — was 2)")
    print("  - fused_silu_mul_backward_kernel     (1 elementwise kernel — was 4+)")
    print("  - fused_dual_grad_x_kernel            (autotuned; 2 matmuls + add)")
    print()
    print("Autograd Functions:")
    print("  - FusedMLP                            (forward + backward, soft path)")
    print()
    print("Functional interfaces:")
    print("  - fused_silu_mul_triton(gate, up)")
    print("  - fused_silu_mul_backward_triton(grad_act, gate, up)")
    print("  - fused_dual_grad_x_triton(grad_a, grad_b, W_a, W_b)")
    print("  - fused_mlp(x, gate_p, gate_l, gate_b, up_p, up_l, up_b, down_p, down_l, down_b, gs, tau)")
    print()
    print("DoD: import check + syntax check.")
