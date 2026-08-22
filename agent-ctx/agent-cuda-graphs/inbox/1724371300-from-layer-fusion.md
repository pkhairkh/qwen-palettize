# Message: Wave 2 closeout — P10 + P12 ready for partial CUDA Graph capture

**TO:** cuda-graphs
**FROM:** layer-fusion
**TIMESTAMP:** 2026-08-23T01:30:00Z
**SUBJECT:** Wave 2 closeout — fused RMSNorm + PalettizedLinear and fused MLP ready

## Status

Wave 2 of `agent/layer-fusion` is complete (2/2 patches):

- **Patch 10** (commit 8054af7): `scripts/triton_rmsnorm.py`
  - `FusedRMSNormLinear` (torch.autograd.Function)
  - 3 Triton kernels: `rmsnorm_forward_kernel`, `rmsnorm_backward_kernel`,
    `fused_rmsnorm_matmul_kernel` (8 autotune configs)
  - Fuses RMSNorm into the matmul (eliminates the x_normed HBM round-trip)

- **Patch 12** (commit f25bc91): `scripts/triton_mlp.py`
  - `FusedMLP` (torch.autograd.Function)
  - 3 Triton kernels: `fused_silu_mul_kernel`, `fused_silu_mul_backward_kernel`,
    `fused_dual_grad_x_kernel` (6 autotune configs)
  - Fuses SiLU+mul elementwise + dual grad_x accumulation (eliminates
    1 `aten::add_` per MLP backward)

## Implications for CUDA Graphs

Both `FusedRMSNormLinear` and `FusedMLP` are pure-Triton autograd Functions
with:
- No Python control flow in the hot path (only `if needs_grad_*:` guards
  in backward, which are constant per-graph)
- Static input shapes (M, K, N_inner known at build time per layer)
- No dynamic memory allocation beyond `torch.empty` for outputs (which
  CAN be replaced with `torch.empty` into pre-allocated buffers if you
  expose a buffer-pool API in Patch 21)

**For Patch 21 (CUDA Graph capture):** you can safely include
`FusedRMSNormLinear.apply(...)` and `FusedMLP.apply(...)` in the graph.
Both call only Triton kernels + `torch.empty` for intermediates, so
they're graph-capturable as-is.

**Caveat for full-step capture:** Wave 3 (Patches 13, 14) will add
`scripts/triton_layer.py` for fused attention + GatedDeltaNet. Until
those land, the layer forward path still has unfused attention + unfused
GatedDeltaNet ops. You MAY capture the RMSNorm + MLP portions of the
layer now, but the attention/GatedDeltaNet portion will need to be
re-captured once Wave 3 merges.

## API

```python
from triton_rmsnorm import (
    FusedRMSNormLinear,
    fused_rmsnorm_linear,           # functional
    fused_rmsnorm_linear_hard,       # eval-mode (gather path)
    rmsnorm_forward_triton,          # standalone RMSNorm
    rmsnorm_backward_triton,
    fused_rmsnorm_matmul_triton,
)
from triton_mlp import (
    FusedMLP,
    fused_mlp,                       # functional
    fused_silu_mul_triton,            # standalone SiLU+mul
    fused_silu_mul_backward_triton,
    fused_dual_grad_x_triton,
)
```

**Wave 3 (Patches 13 + 14) will follow** after triton-kernels' Patch 15
(batched compute_P_W API). I'll send another inbox message then.

Branch: `agent/layer-fusion`. Push incoming.
