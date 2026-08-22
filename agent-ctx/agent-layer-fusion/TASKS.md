# TASKS: layer-fusion

## Branch
`agent/layer-fusion`

## Overview
You fuse the ENTIRE Qwen3.5 layer into Triton. This is the most complex agent — 4 patches covering RMSNorm, MLP, attention, and GatedDeltaNet fusion.

## Patch Inventory
| # | Patch | Wave | Effort | Status |
|---|-------|------|--------|--------|
| 10 | Fused RMSNorm + Linear | 2 | 1 day | ⬜ |
| 12 | Fused MLP (gate+up+SiLU+down) | 2 | 1.5 days | ⬜ |
| 13 | Fused Attention (FlashAttention-style) | 3 | 3 days | ⬜ |
| 14 | Fused GatedDeltaNet (conv1d + delta-rule) | 3 | 3 days | ⬜ |

---

## WAVE 2

### Sub-task 10a: Fused RMSNorm + PalettizedLinear
**Research:** `research-kernel-efficiency/00_overview.md` §3 (fused layernorm)
**Paper:** `docs/papers/2307.08691` (FlashAttention2 — fused layernorm pattern)

**File:** NEW `scripts/triton_rmsnorm.py`

**Math:**
- Forward: `x_normed = x / sqrt(mean(x^2, dim=-1) + eps) * weight`, then `y = x_normed @ W_ste + bias`
- Backward: `grad_x = grad_y @ W_ste.T * (weight / rstd) * (1 - x_normed^2 / M)`, `grad_weight = sum(grad_x_normed * x_normed, dim=0)`

**Implementation:**
```python
@triton.jit
def fused_rmsnorm_linear_forward_kernel(
    x_ptr, weight_ptr, palette_ptr, logits_ptr, bias_ptr,
    y_ptr, P_aos_ptr, W_ste_ptr, rstd_ptr,
    M, N, K, G, group_size, tau, step_seed, eps,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # 1. Load x tile (BM, BK)
    # 2. Compute mean(x^2) per row → rstd = 1/sqrt(mean+eps)
    # 3. x_normed = x * rstd * weight
    # 4. compute_P_W_ste (Gumbel + softmax + STE) → W_ste tile
    # 5. y = tl.dot(x_normed, W_ste) + bias
    # 6. Save rstd for backward, P_aos + W_ste for backward
```

**Dependency:** Wait for nn-module-foundation's Patch 11 (forward signature with `out_norm` parameter).

**Commit:** `Patch 10: fused RMSNorm + PalettizedLinear (forward + backward)`

### Sub-task 12a: Fused MLP (gate + up + SiLU + down)
**Research:** `research-kernel-efficiency/00_overview.md` §3 (fused MLP)
**Paper:** `docs/papers/2002.05202` (SwiGLU — fused gate+up+activation)

**File:** NEW `scripts/triton_mlp.py`

**Math:**
- Forward: `gate_out = gate_proj(x)` (PalettizedLinear), `up_out = up_proj(x)` (PalettizedLinear), `act = gate_out * SiLU(up_out)`, `y = down_proj(act)` (PalettizedLinear)
- The three PalettizedLinears stay separate (different palettes), but the SiLU + elementwise multiply + the down_proj input are fused into ONE kernel.

**Implementation:**
```python
class FusedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_palette, gate_logits, up_palette, up_logits,
                down_palette, down_logits, bias_gate, bias_up, bias_down,
                group_size, tau):
        # 1. gate_out = compute_P_W_ste + matmul(x, W_ste_gate)  [Triton kernel]
        # 2. up_out = compute_P_W_ste + matmul(x, W_ste_up)        [Triton kernel]
        # 3. act = gate_out * silu(up_out)                          [fused elementwise]
        # 4. y = compute_P_W_ste + matmul(act, W_ste_down)         [Triton kernel]
        # 5. Save all P_aos, W_ste, x, act for backward
        ...

    @staticmethod
    def backward(ctx, grad_y):
        # 1. grad_act = grad_y @ W_ste_down.T  [Triton matmul]
        # 2. grad_gate, grad_up = elementwise backward of SiLU*gate
        # 3. grad_x = grad_gate @ W_ste_gate.T + grad_up @ W_ste_up.T  [fused]
        # 4. grad_palette, grad_logits for all three  [elementwise]
        ...
```

**Commit:** `Patch 12: fused MLP (gate + up + SiLU + down) — eliminates 3 elementwise kernels per layer`

### Sub-task 12b: Send messages + push
- Send to cuda-graphs: "MLP fusion ready — triton_mlp.py can be captured in CUDA Graph"
- Update PROGRESS.md.
- Push.

**Commit:** `Wave 2 closeout: PROGRESS.md + inbox msg to cuda-graphs`

---

## WAVE 3

### Sub-task 13a: Fused Attention (FlashAttention-style)
**Research:** `research-kernel-efficiency/00_overview.md` §3 (fused attention)
**Paper:** `docs/papers/2307.08691` (FlashAttention2 — fused attention pattern)

**File:** NEW `scripts/triton_layer.py`

**Math (for full-attention layers — layer 3, 7, 11, ...):**
- Forward: `Q = q_proj(x)`, `K = k_proj(x)`, `V = v_proj(x)` (PalettizedLinears), `rotary(Q, K)`, `scores = QK^T / sqrt(d)`, `attn = softmax(scores)`, `out = attn @ V`, `y = o_proj(out)` (PalettizedLinear)
- The QKV projections and o_proj stay as PalettizedLinears. The attention computation (rotary + QK^T + softmax + AV) is fused into ONE FlashAttention-style kernel.

**Implementation:**
```python
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr,  # (M, n_heads, d_head) bf16
    out_ptr,              # (M, n_heads, d_head) bf16
    cos_ptr, sin_ptr,     # rotary tables
    M, n_heads, d_head, seq_len,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # FlashAttention2 algorithm:
    # 1. Tile Q into blocks of BM rows
    # 2. For each Q block, iterate over K/V blocks of BN columns
    # 3. Apply rotary embedding
    # 4. Compute QK^T (BM, BN) in fp32
    # 5. Online softmax: track max, sum
    # 6. Compute attn @ V, accumulate
    # 7. Write output
    ...
```

**Commit:** `Patch 13: fused FlashAttention (rotary + QK^T + softmax + AV) — eliminates 15 attention kernels per full-attn layer`

### Sub-task 14a: Fused GatedDeltaNet (conv1d + delta-rule)
**Research:** `research-kernel-accuracy/00_overview.md` (GatedDeltaNet architecture)
**Paper:** `docs/papers/2312.00752` (Mamba — SSM-style fused kernel pattern)

**File:** NEW `scripts/triton_layer.py` (same file as attention)

**Math (for linear-attention layers — layer 0,1,2, 4,5,6, ...):**
- Forward: `qkv = in_proj_qkv(x)` (PalettizedLinear), `z = in_proj_z(x)` (PalettizedLinear), `conv1d(qkv)`, `delta_update(S, q, k, v)`, `out = o_proj(...)` (PalettizedLinear)
- The conv1d is a depth-1 temporal convolution. The delta rule is `S = S + Δ(A @ x @ v^T)`. This is the most complex kernel — the recurrent state S must be updated sequentially along the sequence dimension.

**Implementation:**
```python
@triton.jit
def gated_delta_net_kernel(
    qkv_ptr, z_ptr, A_log_ptr, dt_bias_ptr, conv_weight_ptr,
    out_ptr, state_ptr,
    M, d_state, d_head, seq_len,
    BM: tl.constexpr,
):
    # 1. Load qkv tile
    # 2. Apply conv1d (depth-1, causal)
    # 3. Compute delta = softplus(A_log * x + dt_bias)
    # 4. Update state: S = S + delta * (q @ k^T)  (sequential along seq)
    # 5. Compute out = S @ v * gate(z)
    # 6. Write output
    ...
```

**Commit:** `Patch 14: fused GatedDeltaNet (conv1d + delta-rule + state update)`

### Sub-task 14b: Send messages + push
- Send to cuda-graphs: "All layer kernels ready — triton_layer.py + triton_rmsnorm.py + triton_mlp.py can be captured in CUDA Graph"
- Update PROGRESS.md.
- Push.

**Commit:** `Wave 3 closeout: PROGRESS.md + inbox msg to cuda-graphs`

---

## DoD
- [ ] All syntax checks pass
- [ ] Import check: `python3 -c "import sys; sys.path.insert(0,'scripts'); import triton_rmsnorm, triton_mlp, triton_layer"` passes
- [ ] Fused RMSNorm + Linear (Patch 10)
- [ ] Fused MLP (Patch 12)
- [ ] Fused Attention (Patch 13)
- [ ] Fused GatedDeltaNet (Patch 14)
- [ ] Branch pushed to origin
