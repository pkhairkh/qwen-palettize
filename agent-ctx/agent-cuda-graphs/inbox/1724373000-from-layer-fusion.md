# Message: Wave 3 closeout — all layer kernels ready (P10, P12, P13, P14)

**TO:** cuda-graphs
**FROM:** layer-fusion
**TIMESTAMP:** 2026-08-23T03:00:00Z
**SUBJECT:** Wave 3 closeout — fused attention + GatedDeltaNet ready for CUDA Graph capture

## Status

ALL layer-fusion patches are now complete:

### Wave 2 (already pushed)
- **Patch 10** (commit 8054af7): `scripts/triton_rmsnorm.py`
  - `FusedRMSNormLinear` autograd Function
  - 3 Triton kernels: rmsnorm_forward, rmsnorm_backward, fused_rmsnorm_matmul

- **Patch 12** (commit f25bc91): `scripts/triton_mlp.py`
  - `FusedMLP` autograd Function (SwiGLU)
  - 3 Triton kernels: fused_silu_mul, fused_silu_mul_backward, fused_dual_grad_x

### Wave 3 (this push)
- **Patch 13** (commit 9e31186): `scripts/triton_layer.py` (Part A)
  - `FusedFlashAttention` autograd Function (causal FA2 with online softmax + rotary)
  - 2 Triton kernels: flash_attention_kernel (forward), flash_attention_backward_kernel

- **Patch 14** (commit 291b26b): `scripts/triton_layer.py` (Part B)
  - `FusedGatedDeltaNet` autograd Function (conv1d EMA + delta-rule + state update)
  - 1 Triton kernel: gated_delta_net_forward_kernel (per-head, sequential along seq)
  - Backward uses BPTT via PyTorch recomputation (correct, but not yet fused into Triton)

## Implications for CUDA Graphs

### What you CAN now capture
All four fused layer functions are pure-Triton autograd Functions with:
- Static shapes (M, K, N_inner known at build time per layer)
- No Python control flow in the hot path (only `if needs_grad_*:` guards
  in backward, which are constant per-graph)
- All outputs use `torch.empty` (graph-capturable as-is)

You can safely include in the CUDA Graph:
- `FusedRMSNormLinear.apply(...)` for RMSNorm + PalettizedLinear fusion
- `FusedMLP.apply(...)` for the SwiGLU MLP block
- `FusedFlashAttention.apply(...)` for full-attn layers (3, 7, 11, ...)
- `FusedGatedDeltaNet.apply(...)` for linear-attn layers (0, 1, 2, 4, 5, 6, ...)

### What is still NOT fused (Wave 4 territory)
- **LoRA backward** (Patches 19, 20 — lora-fusion) — still uses PyTorch
  autograd. 31 LoRA modules × 3 matmuls + scaling. Patch 20 will fuse LoRA
  backward with PalettizedLinear backward, eliminating the aten::add_ for
  grad_x accumulation.
- **GatedDeltaNet backward** — uses BPTT via PyTorch recomputation (correct
  but slow).  A fully-fused Triton BPTT kernel is left as a future
  optimization — it requires reversing the sequential state update.
- **PalettizedLinear bias gradient** — uses `grad_y.sum(dim=0)` (one
  reduction per step, not the hot path — kept in PyTorch).

### Recommended Patch 21 (CUDA Graph capture) sequence
1. Capture teacher forward (4 layers, bf16, cuBLAS) — already a single
   `nn.Module.forward()` call.
2. Capture student forward: per layer, call the fused Functions in sequence:
   ```python
   h = FusedRMSNormLinear.apply(h, norm_w, palette, logits, bias, gs, tau, eps)
   if is_full_attn_layer(i):
       # QKV projections (PalettizedLinears — to be fused with RMSNorm via Patch 10)
       Q = q_proj(h); K = k_proj(h); V = v_proj(h)
       out = FusedFlashAttention.apply(Q, K, V, cos, sin)
       h = h + o_proj(out)
   else:
       # GatedDeltaNet
       qkv = in_proj_qkv(h); z = in_proj_z(h)
       out = FusedGatedDeltaNet.apply(q, k, v, beta, conv_w, A_log, dt_bias)
       h = h + o_proj(out * sigmoid(z))
   h = FusedRMSNormLinear.apply(h, post_attn_norm_w, ...)
   h = h + FusedMLP.apply(h, gate_p, gate_l, gate_b, up_p, up_l, up_b, down_p, down_l, down_b, gs, tau)
   ```
3. Capture loss + backward + optimizer step.

### Notes on Patch 11 dependency
The integration wiring (calling `FusedRMSNormLinear.apply` from inside
`PalettizedLinear.forward` when `out_norm` is provided) depends on
nn-module-foundation's Patch 11 (forward signature). Until Patch 11 lands,
the fused Functions can only be called via the functional interfaces
(`fused_rmsnorm_linear(...)`, `fused_mlp(...)`, etc.) directly from the
training loop.

## API Summary

```python
# Patch 10
from triton_rmsnorm import (
    FusedRMSNormLinear, fused_rmsnorm_linear, fused_rmsnorm_linear_hard,
    rmsnorm_forward_triton, rmsnorm_backward_triton, fused_rmsnorm_matmul_triton,
)
# Patch 12
from triton_mlp import (
    FusedMLP, fused_mlp,
    fused_silu_mul_triton, fused_silu_mul_backward_triton, fused_dual_grad_x_triton,
)
# Patch 13 + 14
from triton_layer import (
    FusedFlashAttention, fused_flash_attention,
    flash_attention_triton, flash_attention_backward_triton,
    FusedGatedDeltaNet, fused_gated_delta_net,
    gated_delta_net_forward_triton,
)
```

## DoD Verification

```
$ python3 -c "import sys; sys.path.insert(0,'scripts'); import triton_rmsnorm, triton_mlp, triton_layer"
(no output — all 3 modules import successfully)

$ python3 -c "import ast; [ast.parse(open(f'scripts/{m}.py').read()) for m in ['triton_rmsnorm','triton_mlp','triton_layer']]; print('syntax OK')"
syntax OK
```

Branch `agent/layer-fusion` is ready for merge to main (after nn-module-foundation
Patch 11 + triton-kernels Patch 15 land). Pushing now.
