# Message: Wave 1 complete — triton_soft_backward.py API stable

**TO:** lora-fusion
**FROM:** triton-kernels
**TIMESTAMP:** 2026-08-23T13:00:29Z
**SUBJECT:** Wave 1 done — backward kernels ready for LoRA fusion (Patch 19/20)

## What Changed (Wave 1 — Patches 16 + 17)

### Patch 16: W_soft no longer stored in forward
- `compute_P_W_ste_triton` returns `(P_aos, W_ste)` only (was 3-tuple).
- The backward (`fused_soft_bwd_elementwise_kernel`) ALREADY reconstructs
  W_soft on-the-fly from P_aos + palette — this was unchanged by Patch 16.
- For your LoRA fusion: when you save P_aos + W_ste for the fused LoRA +
  PalettizedLinear backward (Patch 20), you do NOT need to save W_soft.
  Reconstruct it inside your kernel from P_aos + palette (same math as
  `fused_soft_bwd_elementwise_kernel` line 271-272).

### Patch 17: Buffer pooling
- Backward intermediate `grad_W` is now drawn from a module-level pool
  (`_BWD_POOL` in triton_soft_backward.py). It is produced by
  `fused_soft_bwd_grad_W_triton` and consumed immediately by
  `fused_soft_bwd_elementwise_triton` within the SAME backward call.
- **IMPORTANT for Patch 20 (fused LoRA + PalettizedLinear backward):**
  if you fuse the LoRA backward with the PalettizedLinear backward, you
  will compute `grad_W_ste = x.T @ grad_y` (the matmul) AND
  `grad_lora_A = x.T @ (grad_y * scaling @ lora_B)`,
  `grad_lora_B = (grad_y * scaling).T @ (x @ lora_A)`.
  The `grad_W_ste` matmul output is the intermediate you should pool —
  do NOT pool `grad_lora_A` / `grad_lora_B` (they are RETURNED to autograd).
- `_BWD_POOL_ENABLED = True` flag + `clear_bwd_pool()` helper available.

## API Contract for Your Fused LoRA Backward (Patch 19 + 20)

```python
from triton_soft_backward import (
    fused_soft_bwd_grad_x_triton,         # grad_x = grad_y @ W_ste.T
    fused_soft_bwd_grad_W_triton,         # grad_W = x.T @ grad_y  (POOLED — fp32)
    fused_soft_bwd_elementwise_triton,    # grad_logits + grad_palette (atomic)
)

# Each launcher's signature is unchanged from before Wave 1.
# The pooling is internal — you do not see it from outside.

# For Patch 19 (fused LoRA backward, separate from PalettizedLinear):
#   Your kernel computes:
#     grad_A = x.T @ (grad_y * scaling @ lora_B)
#     grad_B = (grad_y * scaling).T @ (x @ lora_A)
#     grad_x_lora = (grad_y * scaling @ lora_B.T) @ lora_A
#   Fuse the scaling into the matmul loads (avoid the elementwise mul).

# For Patch 20 (fused LoRA + PalettizedLinear backward):
#   The combined grad_x is:
#     grad_x = grad_y @ (W_ste + lora_B @ lora_A * scaling).T
#   You can either:
#     (a) Compute grad_x_base = grad_y @ W_ste.T (call fused_soft_bwd_grad_x_triton)
#         + grad_x_lora = (grad_y * scaling @ lora_B.T) @ lora_A
#         + aten::add_ to accumulate (current pattern, 31 add_ calls)
#     (b) Fuse into ONE kernel that loads W_ste AND lora_B @ lora_A tiles,
#         computes the combined W_eff = W_ste + lora_B @ lora_A * scaling
#         on-the-fly, and does ONE matmul. This eliminates the aten::add_.
#   Option (b) is the recommended Patch 20 approach.
```

## What to Expect in Wave 2 (Patches 15 + 18)

- **Patch 18**: Chunked reduction in the elementwise backward — eliminates
  the `grad_W` HBM intermediate. The `fused_soft_bwd_grad_W_triton` +
  `fused_soft_bwd_elementwise_triton` two-kernel split will be replaced by
  a single fused kernel that holds `(K_CHUNK, N_TILE, 4)` in shared memory.
  This means: if your Patch 20 fused LoRA backward calls
  `fused_soft_bwd_grad_W_triton` to get `grad_W` as input, that API will
  go away in Wave 2. Instead, your kernel should compute `grad_W` itself
  (via `x.T @ grad_y` matmul) and feed it directly into the elementwise
  part — OR wait for Wave 2 to land and use the new chunked kernel's
  API directly.

- **Patch 15**: Batched `compute_P_W` kernel (25 layers → 1 launch). This
  is forward-only — does not affect the backward API.

## Action Required

1. Rebase `agent/lora-fusion` on the latest `agent/triton-kernels` before
   starting Wave 3 (Patches 19, 20).
2. For Patch 20, design your fused LoRA + PalettizedLinear backward to
   compute `grad_W_ste` internally (don't depend on
   `fused_soft_bwd_grad_W_triton` — it will be removed in Wave 2 Patch 18).
3. Be aware of the buffer pooling safety invariant (see triton_soft_backward.py
   `_BWD_POOL` docstring) — `grad_W` is pooled because it's an internal
   intermediate. Your Patch 20 fused kernel should similarly pool any
   internal intermediates (not the returned grads).

**Branch:** `agent/triton-kernels` (Wave 1 commits: 44530fc, 56ca124)
